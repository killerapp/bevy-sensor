//! Headless rendering implementation using Bevy.
//!
//! This module provides two rendering modes:
//!
//! 1. **Headless mode** (default): Renders to an image texture without requiring
//!    a window or display. Works on WSL2, CI servers, and any environment without
//!    GPU windowing support.
//!
//! 2. **Windowed mode** (fallback): Uses a visible window for rendering when
//!    headless mode fails. Requires a display (X11/Wayland).
//!
//! # Current Status
//!
//! - **RGBA**: Working via render-to-texture + GPU readback
//! - **Depth**: Working via ViewDepthTexture + reverse-Z conversion
//!
//! # Headless Rendering Architecture
//!
//! The headless renderer:
//! 1. Creates a Bevy app without window plugins (uses ScheduleRunnerPlugin)
//! 2. Sets up a render-to-texture pipeline with RenderTarget::Image
//! 3. Extracts RGBA data via ImageCopyDriver
//! 4. Extracts depth via DepthReadbackNode
//!
//! # Running Requirements
//!
//! Headless mode should work without any display. For windowed fallback:
//! ```bash
//! DISPLAY=:0 cargo run --example test_render
//! ```
//!
//! # Architecture Notes
//!
//! Bevy's `App::run()` does not return cleanly in all configurations. This
//! implementation uses a watchdog thread that monitors for completion and
//! calls `std::process::exit(0)` once the render output is serialized to
//! a temp file. The main thread reads this file after the process would
//! normally exit.

use bevy::app::{ScheduleRunnerPlugin, TerminalCtrlCHandlerPlugin};
use bevy::asset::{LoadState, RenderAssetUsages};
use bevy::camera::RenderTarget;
use bevy::core_pipeline::prepass::{DepthPrepass, NormalPrepass};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::ecs::query::QueryItem;
use bevy::light::GlobalAmbientLight;
use bevy::log::LogPlugin;
use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_graph::{
    Node, NodeRunError, RenderGraphContext, RenderGraphExt, RenderLabel, ViewNode, ViewNodeRunner,
};
use bevy::render::render_resource::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Extent3d, MapMode, Origin3d,
    TexelCopyBufferInfo, TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect,
    TextureDimension, TextureFormat, TextureUsages,
};
use bevy::render::renderer::RenderQueue;
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::texture::GpuImage;
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
use bevy::render::view::{ExtractedView, Hdr, ViewDepthTexture};
use bevy::render::{Extract, Render, RenderApp, RenderSystems};
use bevy::window::{ExitCondition, WindowPlugin};
use bevy_obj::ObjPlugin;
use std::fs::File;
use std::io::Read as IoRead;
use std::path::{Path, PathBuf};
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use crate::{
    backend::BackendConfig, ObjectRotation, RenderConfig, RenderError, RenderOutput,
    TargetingPolicy,
};
use ycbust::{GOOGLE_16K_MESH_RELATIVE, GOOGLE_16K_TEXTURE_RELATIVE};

/// Watchdog timeout for a single render, in seconds.
///
/// Bounds how long any single render path waits before declaring failure.
/// 180s accommodates first-run wgpu shader compilation on Windows, which
/// can take well over 60s on a cold GPU cache (see commit 9cd1d11).
const RENDER_TIMEOUT_SECS: u64 = 180;

/// Warmup frames after each camera move in `render_headless_sequence`.
///
/// After writing a new camera `Transform`, Bevy needs at least one frame for
/// transform propagation + render-world extract before the next capture is
/// valid. Historically set to 3 as a conservative cushion; reducing directly
/// shortens per-viewpoint wall-clock since `app.update()` in the batch path
/// is not rate-limited. Validated against the pixel-exact hardware test
/// `test_batch_render_matches_sequential_episode_outputs`.
const BATCH_WARMUP_FRAMES: u32 = 1;

/// Warmup frames at the start of each `PersistentRenderer::render()` call.
///
/// `BATCH_WARMUP_FRAMES = 1` works for inter-viewpoint advancement inside a
/// batch because `extract_and_continue_headless_batch` writes the next
/// camera transform *and* clears the shared GPU readback buffers in the
/// same tick — so the in-flight copy from the previous viewpoint has
/// already drained by the time the next capture is gated.
///
/// In the persistent per-call path, the previous render's output may still
/// be sitting in `shared_rgba`/`shared_depth` (we clear them before the
/// loop, but the pipeline still needs ticks to propagate the new camera/
/// scene-rotation `Transform` writes through `PostUpdate` →
/// `transform_propagate` → `Extract` → render graph → `ImageCopyDriver`
/// before the capture we request actually reflects the new transforms.
///
/// Validated by `test_persistent_renderer_matches_render_to_buffer`. Three
/// ticks of warmup gives Windows/DX12 enough room to drain the previous
/// readback and capture the post-propagation color target:
///   - tick 0: transforms propagate, render runs (no copy enabled)
///   - tick 1: previous in-flight readback drains (no copy enabled)
///   - tick 2: warmup hits 0, capture fires, render runs with copy enabled
///   - tick 3: shared buffers populated → captured → batch finalized
const PERSISTENT_WARMUP_FRAMES: u32 = 3;

fn persistent_warmup_camera_transform() -> Transform {
    crate::generate_viewpoints(&crate::ViewpointConfig::default())
        .into_iter()
        .next()
        .unwrap_or_else(|| Transform::from_xyz(0.0, 0.0, 0.5).looking_at(Vec3::ZERO, Vec3::Y))
}

/// Check the render-trace env var. Cheap enough (single HashMap lookup) to call
/// from per-frame systems; gate all tracing output behind this.
#[inline]
fn render_trace_enabled() -> bool {
    std::env::var("BEVY_SENSOR_RENDER_TRACE").is_ok()
}

/// Convert a filesystem path into a Bevy asset-path string.
///
/// `std::fs::canonicalize` on Windows returns a `\\?\C:\...` verbatim-prefixed
/// path. Bevy's `AssetPath` parser cannot handle that prefix, so the asset
/// would silently never load. Strip the verbatim prefix and normalize
/// separators to `/` so the absolute path resolves through the default file
/// asset source on every platform.
fn fs_path_to_asset_string(path: &std::path::Path) -> String {
    let s = path.display().to_string();
    let s = s.strip_prefix(r"\\?\").map(str::to_string).unwrap_or(s);
    s.replace('\\', "/")
}

/// Check if a display is available for windowed rendering.
///
/// Returns true if DISPLAY or WAYLAND_DISPLAY environment variable is set.
#[allow(dead_code)]
fn display_available() -> bool {
    std::env::var("DISPLAY").is_ok() || std::env::var("WAYLAND_DISPLAY").is_ok()
}

/// Check if we're running on WSL2 (which doesn't support Vulkan window surfaces).
#[allow(dead_code)]
fn is_wsl2() -> bool {
    if let Ok(version) = std::fs::read_to_string("/proc/version") {
        return version.to_lowercase().contains("microsoft")
            || version.to_lowercase().contains("wsl");
    }
    false
}

/// Internal state for tracking render progress
#[derive(Resource, Default)]
struct RenderState {
    frame_count: u32,
    scene_loaded: bool,
    texture_loaded: bool,
    materials_applied: bool,
    /// `frame_count` at the moment materials were applied; used to gate
    /// `capture_ready` on N frames of render-graph propagation rather than
    /// a legacy llvmpipe-era 60-frame wait.
    materials_applied_frame: u32,
    /// `frame_count` when the texture finished loading. Capture waits a small
    /// margin past this for GPU image preparation. The material (and therefore
    /// the main-pass pipeline) is applied earlier, so by the time the texture is
    /// ready the pipeline has already compiled.
    texture_ready_frame: u32,
    capture_ready: bool,
    screenshot_requested: bool,
    /// Number of frames spent waiting for a *valid* (non-blank / valid-depth)
    /// readback. The one-shot GPU capture is nondeterministic and occasionally
    /// reads a uniform clear-color frame; we reject those and keep capturing
    /// until a real frame lands, bounded by this counter.
    capture_retries: u32,
    /// Previous frame's RGBA readback. The capture is accepted only once two
    /// consecutive readbacks are identical (the render has settled), so partial
    /// in-progress frames aren't captured and every render path yields the same
    /// fully-drawn image (required for byte-exact cross-path parity).
    prev_rgba: Option<Vec<u8>>,
    /// Previous frame's depth readback, for the same settle-detection as
    /// `prev_rgba` (depth parity is asserted to ~1e-9, i.e. bit-exact).
    prev_depth: Option<Vec<f64>>,
    captured: bool,
    exit_requested: bool,
    #[allow(dead_code)]
    exit_frame_count: u32,
    rgba_data: Option<Vec<u8>>,
    depth_data: Option<Vec<f64>>,
    image_width: u32,
    image_height: u32,
}

#[cfg(test)]
static HEADLESS_SCENE_SETUP_COUNT: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
fn reset_headless_scene_setup_count() {
    HEADLESS_SCENE_SETUP_COUNT.store(0, Ordering::SeqCst);
}

#[cfg(test)]
fn headless_scene_setup_count() -> usize {
    HEADLESS_SCENE_SETUP_COUNT.load(Ordering::SeqCst)
}

/// Shared buffer for screenshot callback to write into
#[derive(Resource, Clone)]
#[allow(clippy::type_complexity)]
#[allow(dead_code)]
struct SharedImageBuffer(Arc<Mutex<Option<(Vec<u8>, u32, u32)>>>);

/// Shared buffer for depth data from GPU readback
/// Contains: (linear_depth_values, width, height)
/// Uses f64 for TBP numerical precision compatibility.
#[derive(Resource, Clone, Default)]
#[allow(clippy::type_complexity)]
struct SharedDepthBuffer(Arc<Mutex<Option<(Vec<f64>, u32, u32)>>>);

// ============================================================================
// Depth Readback Infrastructure
// ============================================================================

/// Request to capture depth - extracted from main world to render world
#[derive(Resource, Default, Clone)]
struct DepthCaptureRequest {
    requested: bool,
    near: f32,
    far: f32,
}

/// Pending depth capture info for async processing.
///
/// `m22`/`m32` are the relevant entries of the view's reverse-Z projection
/// matrix (`clip_from_view`), captured at copy time so the CPU-side
/// linearization matches the exact projection the GPU rendered with. This keeps
/// depth output robust if projection construction or backend behavior changes.
struct PendingDepthCapture {
    buffer: Buffer,
    width: u32,
    height: u32,
    m22: f32,
    m32: f32,
    far: f32,
}

fn render_projection(config: &RenderConfig) -> Projection {
    let near = config.near_plane;
    Projection::Perspective(PerspectiveProjection {
        fov: config.fov_radians(),
        near,
        far: config.far_plane,
        near_clip_plane: Vec4::new(0.0, 0.0, -1.0, -near),
        ..default()
    })
}

/// Queue for pending depth captures (written by render node, read by cleanup system)
#[derive(Resource, Default)]
struct PendingDepthCaptureQueue(Arc<Mutex<Vec<PendingDepthCapture>>>);

#[cfg(test)]
mod projection_tests {
    use super::*;

    #[test]
    fn render_projection_uses_configured_near_plane_for_effective_clip_matrix() {
        let mut config = RenderConfig::tbp_default();
        config.near_plane = 0.025;
        config.far_plane = 12.0;

        let projection = render_projection(&config);
        let Projection::Perspective(perspective) = &projection else {
            panic!("render_projection should create a perspective projection");
        };

        assert_eq!(perspective.near, config.near_plane);
        assert_eq!(
            perspective.near_clip_plane,
            Vec4::new(0.0, 0.0, -1.0, -config.near_plane)
        );
        assert_eq!(perspective.far, config.far_plane);

        let clip_from_view = projection.get_clip_from_view();
        assert!(
            (clip_from_view.w_axis.z - config.near_plane).abs() < 1e-6,
            "reverse-Z projection matrix should encode configured near plane; got {}",
            clip_from_view.w_axis.z
        );
    }
}

// ============================================================================
// Depth Buffer Helpers
// ============================================================================

mod depth_helpers {
    /// wgpu requires buffer row alignment of 256 bytes
    pub const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;

    /// Align byte size to wgpu's COPY_BYTES_PER_ROW_ALIGNMENT
    pub fn align_byte_size(value: u32) -> u32 {
        let remainder = value % COPY_BYTES_PER_ROW_ALIGNMENT;
        if remainder == 0 {
            value
        } else {
            value + (COPY_BYTES_PER_ROW_ALIGNMENT - remainder)
        }
    }

    /// Calculate aligned buffer size for an image
    #[allow(dead_code)]
    pub fn get_aligned_size(width: u32, height: u32, pixel_size: u32) -> u32 {
        height * align_byte_size(width * pixel_size)
    }

    /// Convert reverse-Z NDC depth to linear depth in meters.
    ///
    /// Bevy uses reverse-Z depth buffer: near plane maps to depth=1, far plane to depth=0.
    /// This provides better precision for distant objects.
    ///
    /// Formula derivation:
    /// - At near plane (z = near): ndc = 1
    /// - At far plane (z = far): ndc = 0
    /// - linear = far / (1 + ndc * (far/near - 1))
    ///
    /// Superseded in the render path by [`ndc_to_linear_with_matrix`], which
    /// reads the actual projection near from the view matrix instead of trusting
    /// a passed-in near (the source of the #92 10x depth error). Retained for its
    /// tests and as a reference formula.
    #[allow(dead_code)]
    pub fn reverse_z_to_linear_depth(ndc_depth: f32, near: f32, far: f32) -> f32 {
        // Handle edge cases
        if ndc_depth <= 0.0 {
            return far; // Background (infinite distance in reverse-Z)
        }
        if ndc_depth >= 1.0 {
            return near; // At or beyond near plane
        }
        // Reverse-Z formula: linear = far / (1 + ndc * (far/near - 1))
        far / (1.0 + ndc_depth * (far / near - 1.0))
    }

    /// Extract depth values from aligned buffer, handling row padding
    pub fn extract_depth_with_alignment(data: &[u8], width: u32, height: u32) -> Vec<f32> {
        let pixel_size = 4u32; // f32 = 4 bytes
        let aligned_row_bytes = align_byte_size(width * pixel_size) as usize;
        let actual_row_bytes = (width * pixel_size) as usize;

        let mut depth_values = Vec::with_capacity((width * height) as usize);

        for y in 0..height as usize {
            let row_start = y * aligned_row_bytes;
            let row_data = &data[row_start..row_start + actual_row_bytes];

            for x in 0..width as usize {
                let offset = x * 4;
                let bytes: [u8; 4] = row_data[offset..offset + 4].try_into().unwrap();
                let depth_value = f32::from_le_bytes(bytes);
                depth_values.push(depth_value);
            }
        }

        depth_values
    }

    /// Convert all NDC depth values to linear meters (as f64 for TBP precision).
    /// Superseded by [`convert_depth_to_linear_with_matrix`]; retained for tests.
    #[allow(dead_code)]
    pub fn convert_depth_to_linear(raw_depth: &[f32], near: f32, far: f32) -> Vec<f64> {
        raw_depth
            .iter()
            .map(|&ndc| reverse_z_to_linear_depth(ndc, near, far) as f64)
            .collect()
    }

    /// Linearize a reverse-Z NDC depth using the view's actual projection matrix,
    /// rather than a hand-supplied near/far.
    ///
    /// For a perspective right-handed projection, the relevant clip-space rows are
    /// `clip_z = m22 * z + m32` and `clip_w = -z` (camera looks down -Z), so
    /// `ndc = clip_z / clip_w = (m22*z + m32) / (-z)`. Solving for the positive
    /// view-space distance `d = -z` gives **`d = m32 / (ndc + m22)`**. This holds
    /// for both finite and infinite reverse-Z and is correct regardless of which
    /// near plane the renderer actually used — the previous fixed-near formula
    /// produced depths 10x too small when the effective projection near plane
    /// drifted from `RenderConfig::near_plane` (issue #86/#92/#95).
    ///
    /// `m22 = clip_from_view[col=2][row=2]`, `m32 = clip_from_view[col=3][row=2]`.
    /// `ndc <= 0` is the reverse-Z far plane (background) and maps to `far`.
    pub fn ndc_to_linear_with_matrix(ndc: f32, m22: f32, m32: f32, far: f32) -> f32 {
        if ndc <= 0.0 {
            return far; // background / at-or-beyond far plane in reverse-Z
        }
        let denom = ndc + m22;
        if denom.abs() <= f32::EPSILON {
            return far;
        }
        let linear = m32 / denom;
        if !linear.is_finite() || linear <= 0.0 {
            far
        } else {
            linear.min(far)
        }
    }

    /// Convert all NDC depth values to linear meters using the view projection
    /// matrix (f64 for TBP precision). See [`ndc_to_linear_with_matrix`].
    pub fn convert_depth_to_linear_with_matrix(
        raw_depth: &[f32],
        m22: f32,
        m32: f32,
        far: f32,
    ) -> Vec<f64> {
        raw_depth
            .iter()
            .map(|&ndc| ndc_to_linear_with_matrix(ndc, m22, m32, far) as f64)
            .collect()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_align_byte_size() {
            assert_eq!(align_byte_size(256), 256);
            assert_eq!(align_byte_size(257), 512);
            assert_eq!(align_byte_size(1), 256);
            assert_eq!(align_byte_size(512), 512);
            assert_eq!(align_byte_size(0), 0);
        }

        #[test]
        fn test_reverse_z_to_linear_depth() {
            let near = 0.01;
            let far = 10.0;

            // Near plane (ndc=1 in reverse-Z)
            let linear_near = reverse_z_to_linear_depth(1.0, near, far);
            assert!((linear_near - near).abs() < 0.001);

            // Mid-range depth (ndc=0.5 should give geometric mean area)
            let linear_mid = reverse_z_to_linear_depth(0.5, near, far);
            // At ndc=0.5: linear = 10 / (1 + 0.5 * (1000-1)) = 10 / 500.5 ≈ 0.02
            assert!(linear_mid > near && linear_mid < far);

            // Very close to far plane (ndc very small)
            let linear_almost_far = reverse_z_to_linear_depth(0.0001, near, far);
            // At ndc=0.0001: linear = 10 / (1 + 0.0001 * 999) ≈ 10 / 1.0999 ≈ 9.09
            assert!(linear_almost_far > 9.0);

            // Background (ndc=0)
            let background = reverse_z_to_linear_depth(0.0, near, far);
            assert_eq!(background, far);
        }

        #[test]
        fn test_extract_depth_with_alignment() {
            // 2x2 image, 4 bytes per pixel
            // Aligned row = 256 bytes, but actual = 8 bytes
            let width = 2u32;
            let height = 2u32;

            let mut data = vec![0u8; 256 * 2]; // 2 aligned rows

            // Write test depth values
            // Row 0: [0.5, 0.6]
            data[0..4].copy_from_slice(&0.5f32.to_le_bytes());
            data[4..8].copy_from_slice(&0.6f32.to_le_bytes());
            // Row 1: [0.7, 0.8]
            data[256..260].copy_from_slice(&0.7f32.to_le_bytes());
            data[260..264].copy_from_slice(&0.8f32.to_le_bytes());

            let depth = extract_depth_with_alignment(&data, width, height);
            assert_eq!(depth.len(), 4);
            assert!((depth[0] - 0.5).abs() < 0.001);
            assert!((depth[1] - 0.6).abs() < 0.001);
            assert!((depth[2] - 0.7).abs() < 0.001);
            assert!((depth[3] - 0.8).abs() < 0.001);
        }

        #[test]
        fn test_reverse_z_depth_at_near_plane() {
            // Near plane should give near value
            let near = 0.01;
            let far = 100.0;
            let depth = reverse_z_to_linear_depth(1.0, near, far);
            assert!((depth - near).abs() < 0.0001);
        }

        #[test]
        fn test_reverse_z_depth_at_far_plane() {
            // Far plane (ndc=0) should give far value
            let near = 0.01;
            let far = 100.0;
            let depth = reverse_z_to_linear_depth(0.0, near, far);
            assert!((depth - far).abs() < 0.0001);
        }

        #[test]
        fn test_reverse_z_monotonic() {
            // Depth should increase as NDC decreases (reverse-Z)
            let near = 0.01;
            let far = 10.0;

            let mut prev_depth = 0.0;
            for i in (0..=100).rev() {
                let ndc = i as f32 / 100.0;
                let depth = reverse_z_to_linear_depth(ndc, near, far);
                assert!(
                    depth >= prev_depth,
                    "Depth should be monotonic: ndc={}, depth={}, prev={}",
                    ndc,
                    depth,
                    prev_depth
                );
                prev_depth = depth;
            }
        }

        #[test]
        fn test_ndc_to_linear_with_matrix_infinite_reverse_z() {
            // Infinite reverse-Z (Bevy `perspective_infinite_reverse_rh`):
            // m22 = 0, m32 = near. d = near / ndc.
            let (m22, m32, far) = (0.0f32, 0.1f32, 10.0f32);

            // The exact regression from #92: ndc 0.366504 must linearize to
            // ~0.273 m (near 0.1), NOT ~0.027 m (the old fixed near = 0.01).
            let d = ndc_to_linear_with_matrix(0.366504, m22, m32, far);
            assert!((d as f64 - 0.272849).abs() < 1e-4, "got {d}");

            // Background (reverse-Z far plane) and clamping.
            assert_eq!(ndc_to_linear_with_matrix(0.0, m22, m32, far), far);
            assert_eq!(ndc_to_linear_with_matrix(-0.5, m22, m32, far), far);
            // Very small ndc -> very far -> clamped to far.
            assert_eq!(ndc_to_linear_with_matrix(1e-9, m22, m32, far), far);
        }

        #[test]
        fn test_ndc_to_linear_with_matrix_finite_reverse_z() {
            // Finite reverse-Z maps near->ndc 1, far->ndc 0. Construct the matrix
            // entries for near=0.5, far=20: m22 = near/(far-near), m32 = far*m22.
            let (near, far) = (0.5f32, 20.0f32);
            let m22 = near / (far - near);
            let m32 = far * m22;
            // ndc = 1 -> near; ndc = 0 -> far (background sentinel also returns far).
            assert!((ndc_to_linear_with_matrix(1.0, m22, m32, far) - near).abs() < 1e-4);
            assert_eq!(ndc_to_linear_with_matrix(0.0, m22, m32, far), far);
        }

        #[test]
        fn test_convert_depth_to_linear_batch() {
            let near = 0.01f32;
            let far = 10.0f32;
            let ndc_depths = vec![1.0f32, 0.5, 0.1, 0.0];

            let linear = convert_depth_to_linear(&ndc_depths, near, far);

            assert_eq!(linear.len(), 4);
            // Near plane
            assert!((linear[0] - near as f64).abs() < 0.001);
            // Far plane
            assert!((linear[3] - far as f64).abs() < 0.001);
            // All should be in range [near, far]
            for d in &linear {
                assert!(*d >= near as f64 && *d <= far as f64);
            }
        }

        #[test]
        fn test_align_byte_size_edge_cases() {
            // Powers of two should stay the same if multiple of 256
            assert_eq!(align_byte_size(256), 256);
            assert_eq!(align_byte_size(512), 512);
            assert_eq!(align_byte_size(1024), 1024);

            // Just under 256 should round up to 256
            assert_eq!(align_byte_size(255), 256);
            assert_eq!(align_byte_size(128), 256);

            // Just over 256 should round up to 512
            assert_eq!(align_byte_size(300), 512);
        }

        #[test]
        fn test_extract_depth_64x64() {
            // Test with TBP default resolution
            let width = 64u32;
            let height = 64u32;
            let bytes_per_pixel = 4u32;
            let padded_row = align_byte_size(width * bytes_per_pixel);

            // Create aligned buffer
            let mut data = vec![0u8; (padded_row * height) as usize];

            // Fill with incrementing values
            for y in 0..height {
                for x in 0..width {
                    let value = (y * width + x) as f32 / (width * height) as f32;
                    let offset = (y * padded_row + x * bytes_per_pixel) as usize;
                    data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
                }
            }

            let depth = extract_depth_with_alignment(&data, width, height);
            assert_eq!(depth.len(), (width * height) as usize);

            // Verify first and last values
            assert!((depth[0] - 0.0).abs() < 0.001);
            let expected_last = (width * height - 1) as f32 / (width * height) as f32;
            assert!((depth[(width * height - 1) as usize] - expected_last).abs() < 0.001);
        }
    }
}

// ============================================================================
// Depth Readback Render Node
// ============================================================================

/// Label for the depth readback render graph node.
#[derive(Debug, Hash, PartialEq, Eq, Clone, bevy::render::render_graph::RenderLabel)]
struct DepthReadbackLabel;

/// Render node that copies the main camera's depth texture to a staging buffer.
/// This runs after the main pass completes, using ViewDepthTexture.
#[derive(Default)]
struct DepthReadbackNode;

impl ViewNode for DepthReadbackNode {
    type ViewQuery = (
        &'static ViewDepthTexture,
        &'static ExtractedCamera,
        &'static ExtractedView,
    );

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (view_depth_texture, camera, view): QueryItem<'w, '_, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
        let trace = render_trace_enabled();
        let t0 = trace.then(std::time::Instant::now);

        // Check if depth capture is requested
        let Some(request) = world.get_resource::<DepthCaptureRequest>() else {
            return Ok(());
        };
        if !request.requested {
            return Ok(());
        }

        // Get the pending queue
        let Some(queue) = world.get_resource::<PendingDepthCaptureQueue>() else {
            return Ok(());
        };

        // Get texture size from camera viewport or physical size
        let Some(physical_size) = camera.physical_target_size else {
            return Ok(());
        };
        let width = physical_size.x;
        let height = physical_size.y;

        let render_device = world.resource::<RenderDevice>();

        // Calculate aligned buffer size (wgpu requires 256-byte row alignment)
        let bytes_per_pixel = 4u32; // f32 = 4 bytes (Depth32Float)
        let unpadded_bytes_per_row = width * bytes_per_pixel;
        let padded_bytes_per_row = depth_helpers::align_byte_size(unpadded_bytes_per_row);
        let buffer_size = (padded_bytes_per_row * height) as u64;

        // Create staging buffer for CPU readback
        let staging_buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("depth_staging_buffer"),
            size: buffer_size,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Copy depth texture to staging buffer
        let encoder = render_context.command_encoder();
        encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
                texture: &view_depth_texture.texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::DepthOnly,
            },
            TexelCopyBufferInfo {
                buffer: &staging_buffer,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_bytes_per_row),
                    rows_per_image: Some(height),
                },
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );

        // Push to queue for async processing (queue is Arc<Mutex<Vec>>).
        // Capture the projection-matrix entries used for linearization: for a
        // perspective RH matrix, clip_z = m22*z + m32 and clip_w = -z, so the
        // positive view-space distance is d = m32 / (ndc + m22).
        let clip_from_view = view.clip_from_view;
        if let Ok(mut pending) = queue.0.lock() {
            pending.push(PendingDepthCapture {
                buffer: staging_buffer,
                width,
                height,
                m22: clip_from_view.z_axis.z,
                m32: clip_from_view.w_axis.z,
                far: request.far,
            });
        }

        if let Some(t0) = t0 {
            eprintln!(
                "[render_trace][node] DepthReadbackNode ms={:.3}",
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }

        Ok(())
    }
}

// ============================================================================
// Depth Readback Plugin
// ============================================================================

/// Plugin that sets up depth buffer readback from the GPU.
struct DepthReadbackPlugin {
    shared_depth: SharedDepthBuffer,
    near: f32,
    far: f32,
}

impl Plugin for DepthReadbackPlugin {
    fn build(&self, app: &mut App) {
        use bevy::core_pipeline::core_3d::graph::Core3d;
        use bevy::core_pipeline::core_3d::graph::Node3d;

        // Insert shared depth buffer in main app
        app.insert_resource(self.shared_depth.clone());
        app.insert_resource(DepthCaptureRequest {
            requested: false,
            near: self.near,
            far: self.far,
        });

        // Get render app
        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            eprintln!("Failed to get RenderApp for depth readback");
            return;
        };

        // Insert resources in render world
        render_app.insert_resource(self.shared_depth.clone());
        render_app.init_resource::<PendingDepthCaptureQueue>();

        // Add extraction system to copy request from main world
        render_app.add_systems(ExtractSchedule, extract_depth_request);

        // Add system to process completed depth captures
        render_app.add_systems(
            Render,
            collect_depth_captures.in_set(RenderSystems::Cleanup),
        );

        // Register the depth readback node in the render graph
        // Run after main pass completes (depth buffer is ready) but before tonemapping
        render_app
            .add_render_graph_node::<ViewNodeRunner<DepthReadbackNode>>(Core3d, DepthReadbackLabel)
            .add_render_graph_edges(
                Core3d,
                (Node3d::EndMainPass, DepthReadbackLabel, Node3d::Tonemapping),
            );
    }
}

/// Extract depth capture request from main world to render world
fn extract_depth_request(mut commands: Commands, request: Extract<Res<DepthCaptureRequest>>) {
    commands.insert_resource(DepthCaptureRequest {
        requested: request.requested,
        near: request.near,
        far: request.far,
    });
}

/// Process completed depth buffer captures (synchronous GPU-to-CPU readback with device polling)
fn collect_depth_captures(
    queue: Res<PendingDepthCaptureQueue>,
    shared_depth: Res<SharedDepthBuffer>,
    render_device: Res<RenderDevice>,
) {
    let trace = render_trace_enabled();
    let t_sys = trace.then(std::time::Instant::now);

    // Take all pending captures from the queue
    let pending_captures = {
        let Ok(mut pending) = queue.0.lock() else {
            return;
        };
        std::mem::take(&mut *pending)
    };

    if pending_captures.is_empty() {
        if let Some(t0) = t_sys {
            eprintln!(
                "[render_trace][sys] collect_depth_captures empty ms={:.3}",
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }
        return;
    }

    let pending_count = pending_captures.len();

    // Process each pending capture synchronously with device polling
    for pending in pending_captures {
        let width = pending.width;
        let height = pending.height;
        let m22 = pending.m22;
        let m32 = pending.m32;
        let far = pending.far;
        let buffer = pending.buffer;
        let shared = shared_depth.0.clone();

        // Use blocking sync approach with device polling (same as RGBA capture)
        let buffer_slice = buffer.slice(..);

        // Request mapping
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        let t_wait = trace.then(std::time::Instant::now);
        let mut poll_iters: u32 = 0;

        // Poll the device until mapping completes
        loop {
            let _ =
                render_device.poll(bevy::render::render_resource::PollType::wait_indefinitely());
            poll_iters += 1;
            match rx.try_recv() {
                Ok(Ok(())) => {
                    let data = buffer_slice.get_mapped_range();

                    // Extract depth values with alignment handling
                    let ndc_depth =
                        depth_helpers::extract_depth_with_alignment(&data, width, height);

                    drop(data);
                    buffer.unmap();

                    // Convert reverse-Z NDC to linear depth (meters) using the
                    // view's actual projection matrix entries. See
                    // `convert_depth_to_linear_with_matrix`.
                    let linear_depth = depth_helpers::convert_depth_to_linear_with_matrix(
                        &ndc_depth, m22, m32, far,
                    );

                    // Store in shared buffer
                    if let Ok(mut guard) = shared.lock() {
                        *guard = Some((linear_depth, width, height));
                    }
                    break;
                }
                Ok(Err(e)) => {
                    eprintln!("Failed to map depth buffer: {:?}", e);
                    break;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    // Keep polling
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    eprintln!("Depth buffer mapping channel disconnected");
                    break;
                }
            }
        }

        if let Some(t_wait) = t_wait {
            eprintln!(
                "[render_trace][sys] collect_depth_captures mapping_wait poll_iters={} ms={:.3}",
                poll_iters,
                t_wait.elapsed().as_secs_f64() * 1000.0
            );
        }
    }

    if let Some(t0) = t_sys {
        eprintln!(
            "[render_trace][sys] collect_depth_captures done pending={} ms={:.3}",
            pending_count,
            t0.elapsed().as_secs_f64() * 1000.0
        );
    }
}

// ============================================================================
// Image Copy Infrastructure (for headless rendering)
// ============================================================================

/// Label for the image copy render graph node
#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct ImageCopyLabel;

/// Component that marks an image for GPU-to-CPU copying
#[derive(Component, Clone)]
struct ImageCopier {
    /// Handle to the source image (render target)
    src_image: Handle<Image>,
    /// Whether to capture on this frame
    enabled: bool,
}

/// Resource containing all ImageCopiers for the render world
#[derive(Resource, Default)]
struct ImageCopiers(Vec<ImageCopier>);

/// Pending image capture for async processing
struct PendingImageCapture {
    buffer: Buffer,
    width: u32,
    height: u32,
    padded_bytes_per_row: u32,
}

/// Queue for pending image captures
#[derive(Resource, Default)]
struct PendingImageCaptureQueue(Arc<Mutex<Vec<PendingImageCapture>>>);

/// Shared buffer for captured RGBA data
#[derive(Resource, Clone, Default)]
#[allow(clippy::type_complexity)]
struct SharedRgbaBuffer(Arc<Mutex<Option<(Vec<u8>, u32, u32)>>>);

/// Render graph node that copies render target images to staging buffers
struct ImageCopyDriver;

impl Node for ImageCopyDriver {
    fn run(
        &self,
        _graph: &mut RenderGraphContext,
        _render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), NodeRunError> {
        let trace = render_trace_enabled();
        let t0 = trace.then(std::time::Instant::now);

        let Some(image_copiers) = world.get_resource::<ImageCopiers>() else {
            return Ok(());
        };

        let Some(gpu_images) = world.get_resource::<RenderAssets<GpuImage>>() else {
            return Ok(());
        };

        let Some(queue) = world.get_resource::<PendingImageCaptureQueue>() else {
            return Ok(());
        };

        let render_device = world.resource::<RenderDevice>();

        let Some(render_queue) = world.get_resource::<RenderQueue>() else {
            return Ok(());
        };

        for image_copier in image_copiers.0.iter() {
            if !image_copier.enabled {
                continue;
            }

            let Some(gpu_image) = gpu_images.get(&image_copier.src_image) else {
                continue;
            };

            let width = gpu_image.size.width;
            let height = gpu_image.size.height;

            // Calculate padded bytes per row (wgpu requires 256-byte alignment)
            let block_dimensions = gpu_image.texture_format.block_dimensions();
            let block_size = gpu_image.texture_format.block_copy_size(None).unwrap_or(4); // Default to 4 bytes for RGBA8

            let padded_bytes_per_row = RenderDevice::align_copy_bytes_per_row(
                (width as usize / block_dimensions.0 as usize) * block_size as usize,
            );

            let buffer_size = (padded_bytes_per_row * height as usize) as u64;

            // Create staging buffer for CPU readback
            let staging_buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some("image_copy_staging_buffer"),
                size: buffer_size,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            // Create command encoder for the copy operation
            let mut encoder =
                render_device.create_command_encoder(&CommandEncoderDescriptor::default());

            let texture_extent = Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            };

            // Copy texture to buffer
            encoder.copy_texture_to_buffer(
                gpu_image.texture.as_image_copy(),
                TexelCopyBufferInfo {
                    buffer: &staging_buffer,
                    layout: TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded_bytes_per_row as u32),
                        rows_per_image: None,
                    },
                },
                texture_extent,
            );

            // Submit the copy command
            render_queue.submit(std::iter::once(encoder.finish()));

            // Queue for async processing
            if let Ok(mut pending) = queue.0.lock() {
                pending.push(PendingImageCapture {
                    buffer: staging_buffer,
                    width,
                    height,
                    padded_bytes_per_row: padded_bytes_per_row as u32,
                });
            }
        }

        if let Some(t0) = t0 {
            eprintln!(
                "[render_trace][node] ImageCopyDriver ms={:.3}",
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }

        Ok(())
    }
}

/// Extract ImageCopier components to render world
fn extract_image_copiers(mut commands: Commands, query: Extract<Query<&ImageCopier>>) {
    commands.insert_resource(ImageCopiers(query.iter().cloned().collect()));
}

/// Process completed image captures
fn collect_image_captures(
    queue: Res<PendingImageCaptureQueue>,
    shared_rgba: Res<SharedRgbaBuffer>,
    render_device: Res<RenderDevice>,
) {
    let trace = render_trace_enabled();
    let t_sys = trace.then(std::time::Instant::now);

    let pending_captures = {
        let Ok(mut pending) = queue.0.lock() else {
            return;
        };
        std::mem::take(&mut *pending)
    };

    if pending_captures.is_empty() {
        if let Some(t0) = t_sys {
            eprintln!(
                "[render_trace][sys] collect_image_captures empty ms={:.3}",
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }
        return;
    }

    let pending_count = pending_captures.len();

    for pending in pending_captures {
        let width = pending.width;
        let height = pending.height;
        let padded_bytes_per_row = pending.padded_bytes_per_row;
        let buffer = pending.buffer;
        let shared = shared_rgba.0.clone();

        // Use blocking sync approach with device polling
        let buffer_slice = buffer.slice(..);

        // Request mapping
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        // Poll the device until mapping completes (with timeout)
        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(10);
        let mut poll_iters: u32 = 0;
        loop {
            let _ =
                render_device.poll(bevy::render::render_resource::PollType::wait_indefinitely());
            poll_iters += 1;

            if start.elapsed() > timeout {
                eprintln!(
                    "Warning: Buffer mapping timeout after {:?}",
                    start.elapsed()
                );
                break;
            }

            match rx.try_recv() {
                Ok(Ok(())) => {
                    let data = buffer_slice.get_mapped_range();

                    // Extract pixels with alignment handling
                    let bytes_per_pixel = 4u32;
                    let actual_row_bytes = (width * bytes_per_pixel) as usize;
                    let padded_row_bytes = padded_bytes_per_row as usize;

                    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
                    for y in 0..height as usize {
                        let row_start = y * padded_row_bytes;
                        rgba.extend_from_slice(&data[row_start..row_start + actual_row_bytes]);
                    }

                    drop(data);
                    buffer.unmap();

                    if let Ok(mut guard) = shared.lock() {
                        *guard = Some((rgba, width, height));
                    }
                    break;
                }
                Ok(Err(e)) => {
                    eprintln!("Failed to map image buffer: {:?}", e);
                    break;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => {
                    // Keep polling
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    eprintln!("Image buffer mapping channel disconnected");
                    break;
                }
            }
        }

        if trace {
            eprintln!(
                "[render_trace][sys] collect_image_captures mapping_wait poll_iters={} ms={:.3}",
                poll_iters,
                start.elapsed().as_secs_f64() * 1000.0
            );
        }
    }

    if let Some(t0) = t_sys {
        eprintln!(
            "[render_trace][sys] collect_image_captures done pending={} ms={:.3}",
            pending_count,
            t0.elapsed().as_secs_f64() * 1000.0
        );
    }
}

/// Plugin for headless image copy
struct ImageCopyPlugin {
    shared_rgba: SharedRgbaBuffer,
}

impl Plugin for ImageCopyPlugin {
    fn build(&self, app: &mut App) {
        use bevy::render::render_graph::RenderGraph;

        app.insert_resource(self.shared_rgba.clone());

        let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
            return;
        };

        render_app.insert_resource(self.shared_rgba.clone());
        render_app.init_resource::<ImageCopiers>();
        render_app.init_resource::<PendingImageCaptureQueue>();

        render_app.add_systems(ExtractSchedule, extract_image_copiers);
        render_app.add_systems(
            Render,
            collect_image_captures.in_set(RenderSystems::Cleanup),
        );

        // Add image copy node to render graph (runs after camera driver)
        let mut graph = render_app.world_mut().resource_mut::<RenderGraph>();
        graph.add_node(ImageCopyLabel, ImageCopyDriver);
        graph.add_node_edge(bevy::render::graph::CameraDriverLabel, ImageCopyLabel);
    }
}

// ============================================================================
// Render Request and Components
// ============================================================================

/// Configuration passed to the Bevy app
#[derive(Resource, Clone)]
struct RenderRequest {
    mesh_path: String,
    texture_path: String,
    camera_transform: Transform,
    object_rotation: ObjectRotation,
    object_translation: Vec3,
    object_scale: Vec3,
    config: RenderConfig,
}

/// Marker for the rendered object
#[derive(Component)]
struct RenderedObject;

/// Marker for the render camera
#[derive(Component)]
struct RenderCamera;

/// Handle for the loaded texture
#[derive(Resource)]
struct LoadedTexture(Handle<Image>);

/// Handle for the loaded scene
#[derive(Resource)]
struct LoadedScene(Handle<Scene>);

/// Shared output for extracting render results
#[derive(Resource, Clone)]
struct SharedOutput(Arc<Mutex<Option<RenderOutput>>>);

/// Handle for the render target image
#[derive(Resource)]
#[allow(dead_code)]
struct RenderTargetImage(Handle<Image>);

/// Tracks progress for a homogeneous batch of viewpoints rendered in one app.
#[derive(Resource)]
struct HeadlessBatchSequence {
    viewpoints: Vec<Transform>,
    current_index: usize,
    outputs: Vec<RenderOutput>,
    warmup_frames_remaining: u32,
    done: bool,
}

impl HeadlessBatchSequence {
    fn new(viewpoints: Vec<Transform>) -> Self {
        let capacity = viewpoints.len();
        Self {
            viewpoints,
            current_index: 0,
            outputs: Vec::with_capacity(capacity),
            warmup_frames_remaining: 0,
            done: capacity == 0,
        }
    }

    fn current_viewpoint(&self) -> Option<Transform> {
        self.viewpoints.get(self.current_index).cloned()
    }
}

/// Perform headless rendering of a YCB object.
///
/// This uses true headless GPU rendering via `RenderTarget::Image`, which does NOT
/// require any window surfaces. This should work on WSL2 and other environments
/// without display servers.
///
/// Note: Bevy's App::run() does not return cleanly. A watchdog thread monitors
/// for results and terminates the process once the render is complete.
#[allow(dead_code)]
pub fn render_headless(
    object_dir: &Path,
    camera_transform: &Transform,
    object_rotation: &ObjectRotation,
    object_translation: Vec3,
    object_scale: Vec3,
    config: &RenderConfig,
) -> Result<RenderOutput, RenderError> {
    // Canonicalize paths so Bevy's asset server can find them regardless of
    // caller working directory. Relative paths like "../../ycb" pass the
    // exists() check but Bevy resolves assets against its own root.
    let object_dir = std::fs::canonicalize(object_dir).map_err(|e| {
        RenderError::RenderFailed(format!(
            "Cannot canonicalize object directory {}: {}",
            object_dir.display(),
            e
        ))
    })?;
    let mesh_path = object_dir.join(GOOGLE_16K_MESH_RELATIVE);
    let texture_path = object_dir.join(GOOGLE_16K_TEXTURE_RELATIVE);

    if !mesh_path.exists() {
        return Err(RenderError::MeshNotFound(fs_path_to_asset_string(
            &mesh_path,
        )));
    }
    if !texture_path.exists() {
        return Err(RenderError::TextureNotFound(fs_path_to_asset_string(
            &texture_path,
        )));
    }

    let request = RenderRequest {
        mesh_path: fs_path_to_asset_string(&mesh_path),
        texture_path: fs_path_to_asset_string(&texture_path),
        camera_transform: *camera_transform,
        object_rotation: object_rotation.clone(),
        object_translation,
        object_scale,
        config: config.clone(),
    };

    let shared_output: SharedOutput = SharedOutput(Arc::new(Mutex::new(None)));
    let output_clone = shared_output.clone();

    // Shared buffer for RGBA data from headless render target
    let shared_rgba: SharedRgbaBuffer = SharedRgbaBuffer::default();

    // Shared buffer for depth readback
    let shared_depth: SharedDepthBuffer = SharedDepthBuffer::default();

    // Create a temp file path for fallback output serialization
    let temp_path =
        std::env::temp_dir().join(format!("bevy_sensor_render_{}.bin", std::process::id()));

    // Spawn watchdog thread that monitors for timeout (don't exit - let Bevy exit gracefully)
    let output_poll_for_timeout = shared_output.clone();
    std::thread::spawn(move || {
        let timeout = std::time::Duration::from_secs(RENDER_TIMEOUT_SECS);
        let start = std::time::Instant::now();
        let poll_interval = std::time::Duration::from_millis(100);

        loop {
            // Check if we have a result
            if let Ok(guard) = output_poll_for_timeout.0.lock() {
                if guard.is_some() {
                    // Output is ready, Bevy will exit via AppExit event
                    return; // Exit watchdog thread, Bevy will handle exit
                }
            }

            if start.elapsed() > timeout {
                eprintln!(
                    "Error: Render timeout after {} seconds",
                    RENDER_TIMEOUT_SECS
                );
                eprintln!("Debug info: This may indicate GPU issues, missing assets, or insufficient system resources.");
                // Force exit on timeout (this is a failure case)
                std::process::exit(1);
            }

            std::thread::sleep(poll_interval);
        }
    });

    // Run Bevy app with HEADLESS configuration (no window surfaces!)
    // Uses ScheduleRunnerPlugin instead of WinitPlugin
    build_headless_app(request, output_clone, shared_rgba, shared_depth).run();

    // App::run() returned - check shared_output for result
    if let Ok(guard) = shared_output.0.lock() {
        if let Some(output) = guard.as_ref() {
            return Ok(output.clone());
        }
    }

    // Fallback: try to read from temp file (for legacy compatibility)
    if temp_path.exists() {
        if let Ok(output) = read_output_from_file(&temp_path) {
            let _ = std::fs::remove_file(&temp_path);
            return Ok(output);
        }
    }

    Err(RenderError::RenderFailed(
        "Render did not complete".to_string(),
    ))
}

/// Render a homogeneous sequence of viewpoints in a single headless Bevy app.
///
/// All captures share the same object, object rotation, and render configuration.
/// This is the fast path used by the batch API for episode-style workloads.
pub fn render_headless_sequence(
    object_dir: &Path,
    viewpoints: &[Transform],
    object_rotation: &ObjectRotation,
    object_translation: Vec3,
    object_scale: Vec3,
    config: &RenderConfig,
) -> Result<Vec<RenderOutput>, RenderError> {
    if viewpoints.is_empty() {
        return Ok(Vec::new());
    }

    let object_dir = std::fs::canonicalize(object_dir).map_err(|e| {
        RenderError::RenderFailed(format!(
            "Cannot canonicalize object directory {}: {}",
            object_dir.display(),
            e
        ))
    })?;
    let mesh_path = object_dir.join(GOOGLE_16K_MESH_RELATIVE);
    let texture_path = object_dir.join(GOOGLE_16K_TEXTURE_RELATIVE);

    if !mesh_path.exists() {
        return Err(RenderError::MeshNotFound(fs_path_to_asset_string(
            &mesh_path,
        )));
    }
    if !texture_path.exists() {
        return Err(RenderError::TextureNotFound(fs_path_to_asset_string(
            &texture_path,
        )));
    }

    let request = RenderRequest {
        mesh_path: fs_path_to_asset_string(&mesh_path),
        texture_path: fs_path_to_asset_string(&texture_path),
        camera_transform: viewpoints[0],
        object_rotation: object_rotation.clone(),
        object_translation,
        object_scale,
        config: config.clone(),
    };

    let shared_rgba: SharedRgbaBuffer = SharedRgbaBuffer::default();
    let rgba_clone = shared_rgba.clone();

    let shared_depth: SharedDepthBuffer = SharedDepthBuffer::default();
    let depth_clone = shared_depth.clone();

    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(bevy::asset::AssetPlugin {
                // Bevy 0.17+ forbids loading from absolute / `..` asset paths by
                // default (UnapprovedPathMode::Forbid → load() silently returns a
                // default handle). YCB meshes load from absolute paths, so allow them.
                unapproved_path_mode: bevy::asset::UnapprovedPathMode::Allow,
                ..default()
            })
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                ..default()
            })
            .disable::<bevy::winit::WinitPlugin>()
            .disable::<LogPlugin>()
            .disable::<TerminalCtrlCHandlerPlugin>(),
    )
    .add_plugins(ObjPlugin)
    // bevy_obj's Scene contains Mesh3d + MeshMaterial3d entities; reflection-based
    // Scene spawning panics unless those component types are registered. The
    // minimal headless plugin set doesn't register them, so do it explicitly.
    .register_type::<Mesh3d>()
    .register_type::<MeshMaterial3d<StandardMaterial>>()
    .register_type::<bevy::prelude::Transform>()
    .register_type::<bevy::prelude::GlobalTransform>()
    .register_type::<bevy::transform::components::TransformTreeChanged>()
    .register_type::<bevy::prelude::Visibility>()
    .register_type::<bevy::prelude::InheritedVisibility>()
    .register_type::<bevy::prelude::ViewVisibility>()
    .add_plugins(ImageCopyPlugin {
        shared_rgba: rgba_clone,
    })
    .add_plugins(DepthReadbackPlugin {
        shared_depth: depth_clone,
        near: config.near_plane,
        far: config.far_plane,
    })
    .insert_resource(request)
    .insert_resource(shared_rgba)
    .insert_resource(HeadlessBatchSequence::new(viewpoints.to_vec()))
    .init_resource::<RenderState>()
    .add_systems(Startup, setup_headless_scene)
    .add_systems(
        Update,
        (
            check_assets_loaded,
            apply_materials,
            tick_headless_batch_warmup,
            request_headless_capture,
            check_headless_capture_ready,
            extract_and_continue_headless_batch,
        )
            .chain(),
    );

    // Manual app.update() loops do not run plugin finish/cleanup hooks automatically.
    // Bevy's screenshot plugin inserts CapturedScreenshots during finish(), so run the
    // normal startup phases before driving the headless batch loop ourselves.
    let trace_outer = render_trace_enabled();
    let t_finish = std::time::Instant::now();
    app.finish();
    let finish_ms = t_finish.elapsed().as_secs_f64() * 1000.0;
    let t_cleanup = std::time::Instant::now();
    app.cleanup();
    let cleanup_ms = t_cleanup.elapsed().as_secs_f64() * 1000.0;
    if trace_outer {
        eprintln!(
            "[render_trace][coldinit] app.finish ms={:.3} app.cleanup ms={:.3}",
            finish_ms, cleanup_ms
        );
    }

    let timeout = std::time::Duration::from_secs(RENDER_TIMEOUT_SECS);
    let start = std::time::Instant::now();

    let trace = std::env::var("BEVY_SENSOR_RENDER_TRACE").is_ok();
    let mut update_idx: u32 = 0;
    let mut last_completed_outputs: usize = 0;
    let mut viewpoint_start = std::time::Instant::now();

    loop {
        if start.elapsed() > timeout {
            return Err(RenderError::RenderTimeout {
                duration_secs: RENDER_TIMEOUT_SECS,
            });
        }

        let update_start = std::time::Instant::now();
        app.update();
        let update_elapsed_ms = update_start.elapsed().as_secs_f64() * 1000.0;

        if trace {
            let batch = app.world().resource::<HeadlessBatchSequence>();
            let warmup = batch.warmup_frames_remaining;
            let current = batch.current_index;
            let completed = batch.outputs.len();
            let vp_ms = viewpoint_start.elapsed().as_secs_f64() * 1000.0;
            eprintln!(
                "[render_trace] update={update_idx} vp={current} warmup={warmup} \
                 completed={completed} update_ms={update_elapsed_ms:.2} vp_ms={vp_ms:.2}"
            );
            if completed > last_completed_outputs {
                eprintln!(
                    "[render_trace] viewpoint {} finished in {:.2} ms",
                    completed - 1,
                    vp_ms
                );
                last_completed_outputs = completed;
                viewpoint_start = std::time::Instant::now();
            }
        }

        update_idx += 1;

        if app.world().resource::<HeadlessBatchSequence>().done {
            break;
        }
    }

    if trace {
        eprintln!(
            "[render_trace] total_wall_ms={:.2} updates={update_idx} viewpoints={}",
            start.elapsed().as_secs_f64() * 1000.0,
            viewpoints.len()
        );
    }

    let mut batch = app.world_mut().resource_mut::<HeadlessBatchSequence>();
    if batch.outputs.len() != viewpoints.len() {
        return Err(RenderError::RenderFailed(format!(
            "Batch render produced {} outputs for {} viewpoints",
            batch.outputs.len(),
            viewpoints.len()
        )));
    }

    Ok(std::mem::take(&mut batch.outputs))
}

/// Assemble the shared single-render headless Bevy app.
fn build_headless_app(
    request: RenderRequest,
    shared_output: SharedOutput,
    shared_rgba: SharedRgbaBuffer,
    shared_depth: SharedDepthBuffer,
) -> App {
    let near = request.config.near_plane;
    let far = request.config.far_plane;

    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .set(bevy::asset::AssetPlugin {
                // Bevy 0.17+ forbids loading from absolute / `..` asset paths by
                // default (UnapprovedPathMode::Forbid → load() silently returns a
                // default handle). YCB meshes load from absolute paths, so allow them.
                unapproved_path_mode: bevy::asset::UnapprovedPathMode::Allow,
                ..default()
            })
            .set(WindowPlugin {
                primary_window: None,
                exit_condition: ExitCondition::DontExit,
                ..default()
            })
            .disable::<bevy::winit::WinitPlugin>()
            .disable::<LogPlugin>()
            .disable::<TerminalCtrlCHandlerPlugin>(),
    )
    .add_plugins(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
        1.0 / 60.0,
    )))
    .add_plugins(ObjPlugin)
    // bevy_obj's Scene contains Mesh3d + MeshMaterial3d entities; reflection-based
    // Scene spawning panics unless those component types are registered. The
    // minimal headless plugin set doesn't register them, so do it explicitly.
    .register_type::<Mesh3d>()
    .register_type::<MeshMaterial3d<StandardMaterial>>()
    .register_type::<bevy::prelude::Transform>()
    .register_type::<bevy::prelude::GlobalTransform>()
    .register_type::<bevy::transform::components::TransformTreeChanged>()
    .register_type::<bevy::prelude::Visibility>()
    .register_type::<bevy::prelude::InheritedVisibility>()
    .register_type::<bevy::prelude::ViewVisibility>()
    .add_plugins(ImageCopyPlugin {
        shared_rgba: shared_rgba.clone(),
    })
    .add_plugins(DepthReadbackPlugin {
        shared_depth,
        near,
        far,
    })
    .insert_resource(request)
    .insert_resource(shared_output)
    .insert_resource(shared_rgba)
    .init_resource::<RenderState>()
    .add_systems(Startup, setup_headless_scene)
    .add_systems(
        Update,
        (
            check_assets_loaded,
            apply_materials,
            request_headless_capture,
            check_headless_capture_ready,
            extract_and_exit_headless,
        )
            .chain(),
    );
    app
}

/// Serialize RenderOutput to bytes for IPC (used by subprocess mode)
#[allow(dead_code)]
fn serialize_output(output: &RenderOutput) -> Vec<u8> {
    let mut data = Vec::new();

    // Header: width, height, rgba_len, depth_len
    data.extend_from_slice(&output.width.to_le_bytes());
    data.extend_from_slice(&output.height.to_le_bytes());
    data.extend_from_slice(&(output.rgba.len() as u32).to_le_bytes());
    data.extend_from_slice(&(output.depth.len() as u32).to_le_bytes());

    // RGBA data
    data.extend_from_slice(&output.rgba);

    // Depth data (as f64 bytes for TBP precision)
    for d in &output.depth {
        data.extend_from_slice(&d.to_le_bytes());
    }

    // Intrinsics (f64 for TBP precision)
    data.extend_from_slice(&output.intrinsics.focal_length[0].to_le_bytes());
    data.extend_from_slice(&output.intrinsics.focal_length[1].to_le_bytes());
    data.extend_from_slice(&output.intrinsics.principal_point[0].to_le_bytes());
    data.extend_from_slice(&output.intrinsics.principal_point[1].to_le_bytes());
    data.extend_from_slice(&output.intrinsics.image_size[0].to_le_bytes());
    data.extend_from_slice(&output.intrinsics.image_size[1].to_le_bytes());

    // Camera transform (translation + rotation quaternion)
    let t = output.camera_transform.translation;
    let r = output.camera_transform.rotation;
    data.extend_from_slice(&t.x.to_le_bytes());
    data.extend_from_slice(&t.y.to_le_bytes());
    data.extend_from_slice(&t.z.to_le_bytes());
    data.extend_from_slice(&r.x.to_le_bytes());
    data.extend_from_slice(&r.y.to_le_bytes());
    data.extend_from_slice(&r.z.to_le_bytes());
    data.extend_from_slice(&r.w.to_le_bytes());

    // Object rotation (f64)
    let or = &output.object_rotation;
    data.extend_from_slice(&or.pitch.to_le_bytes());
    data.extend_from_slice(&or.yaw.to_le_bytes());
    data.extend_from_slice(&or.roll.to_le_bytes());

    // Object translation + scale (f32 for Bevy compatibility)
    let ot = output.object_translation;
    let os = output.object_scale;
    data.extend_from_slice(&ot.x.to_le_bytes());
    data.extend_from_slice(&ot.y.to_le_bytes());
    data.extend_from_slice(&ot.z.to_le_bytes());
    data.extend_from_slice(&os.x.to_le_bytes());
    data.extend_from_slice(&os.y.to_le_bytes());
    data.extend_from_slice(&os.z.to_le_bytes());

    data
}

/// Read RenderOutput from serialized file
fn read_output_from_file(path: &std::path::Path) -> Result<RenderOutput, RenderError> {
    let mut file = File::open(path).map_err(|e| RenderError::RenderFailed(e.to_string()))?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)
        .map_err(|e| RenderError::RenderFailed(e.to_string()))?;

    let mut cursor = 0;

    let read_u32 = |data: &[u8], cursor: &mut usize| -> u32 {
        let val = u32::from_le_bytes(data[*cursor..*cursor + 4].try_into().unwrap());
        *cursor += 4;
        val
    };

    let read_f32 = |data: &[u8], cursor: &mut usize| -> f32 {
        let val = f32::from_le_bytes(data[*cursor..*cursor + 4].try_into().unwrap());
        *cursor += 4;
        val
    };

    let read_f64 = |data: &[u8], cursor: &mut usize| -> f64 {
        let val = f64::from_le_bytes(data[*cursor..*cursor + 8].try_into().unwrap());
        *cursor += 8;
        val
    };

    let width = read_u32(&data, &mut cursor);
    let height = read_u32(&data, &mut cursor);
    let rgba_len = read_u32(&data, &mut cursor) as usize;
    let depth_len = read_u32(&data, &mut cursor) as usize;

    let rgba = data[cursor..cursor + rgba_len].to_vec();
    cursor += rgba_len;

    // Depth data (f64 for TBP precision)
    let mut depth = Vec::with_capacity(depth_len);
    for _ in 0..depth_len {
        depth.push(read_f64(&data, &mut cursor));
    }

    // Intrinsics (f64 for TBP precision)
    let focal_length = [read_f64(&data, &mut cursor), read_f64(&data, &mut cursor)];
    let principal_point = [read_f64(&data, &mut cursor), read_f64(&data, &mut cursor)];
    let image_size = [read_u32(&data, &mut cursor), read_u32(&data, &mut cursor)];

    // Camera transform (f32 for Bevy compatibility)
    let tx = read_f32(&data, &mut cursor);
    let ty = read_f32(&data, &mut cursor);
    let tz = read_f32(&data, &mut cursor);
    let rx = read_f32(&data, &mut cursor);
    let ry = read_f32(&data, &mut cursor);
    let rz = read_f32(&data, &mut cursor);
    let rw = read_f32(&data, &mut cursor);

    // Object rotation (f64)
    let pitch = read_f64(&data, &mut cursor);
    let yaw = read_f64(&data, &mut cursor);
    let roll = read_f64(&data, &mut cursor);

    let (object_translation, object_scale) = if cursor + 24 <= data.len() {
        let tx = read_f32(&data, &mut cursor);
        let ty = read_f32(&data, &mut cursor);
        let tz = read_f32(&data, &mut cursor);
        let sx = read_f32(&data, &mut cursor);
        let sy = read_f32(&data, &mut cursor);
        let sz = read_f32(&data, &mut cursor);
        (Vec3::new(tx, ty, tz), Vec3::new(sx, sy, sz))
    } else {
        (Vec3::ZERO, Vec3::ONE)
    };

    Ok(RenderOutput {
        rgba,
        depth,
        width,
        height,
        intrinsics: crate::CameraIntrinsics {
            focal_length,
            principal_point,
            image_size,
        },
        camera_transform: Transform {
            translation: Vec3::new(tx, ty, tz),
            rotation: Quat::from_xyzw(rx, ry, rz, rw),
            scale: Vec3::ONE,
        },
        object_rotation: ObjectRotation { pitch, yaw, roll },
        object_translation,
        object_scale,
        target_point: Vec3::ZERO,
        targeting_policy: TargetingPolicy::Origin,
    })
}

/// Setup the scene with camera, lighting, and object
#[allow(dead_code)]
fn setup_scene(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    request: Res<RenderRequest>,
    mut _materials: ResMut<Assets<StandardMaterial>>,
) {
    // Camera with depth prepass (Bevy 0.15+ uses Camera3d component)
    // Disable MSAA for depth readback compatibility (can't copy from multisampled texture)
    // Apply FOV from RenderConfig so the projection matches TBP's camera intrinsics.
    commands.spawn((
        Camera3d::default(),
        Camera::default(),
        Hdr,
        render_projection(&request.config),
        Msaa::Off,
        request.camera_transform,
        Tonemapping::None, // Accurate colors for software rendering
        DepthPrepass,
        NormalPrepass,
        RenderCamera,
    ));

    // Ambient light (from config). In Bevy 0.18 the global ambient light is the
    // `GlobalAmbientLight` resource (the `AmbientLight` type became a per-camera component).
    let lighting = &request.config.lighting;
    commands.insert_resource(GlobalAmbientLight {
        color: Color::WHITE,
        brightness: lighting.ambient_brightness,
        ..default()
    });

    // Key light (from config) - Bevy 0.15+ uses PointLight component directly
    if lighting.key_light_intensity > 0.0 {
        commands.spawn((
            PointLight {
                intensity: lighting.key_light_intensity,
                shadows_enabled: lighting.shadows_enabled,
                ..default()
            },
            Transform::from_xyz(
                lighting.key_light_position[0],
                lighting.key_light_position[1],
                lighting.key_light_position[2],
            ),
        ));
    }

    // Fill light (from config)
    if lighting.fill_light_intensity > 0.0 {
        commands.spawn((
            PointLight {
                intensity: lighting.fill_light_intensity,
                shadows_enabled: lighting.shadows_enabled,
                ..default()
            },
            Transform::from_xyz(
                lighting.fill_light_position[0],
                lighting.fill_light_position[1],
                lighting.fill_light_position[2],
            ),
        ));
    }

    // Load the scene
    let scene_handle: Handle<Scene> = asset_server.load(&request.mesh_path);
    commands.insert_resource(LoadedScene(scene_handle.clone()));

    // Load the texture
    let texture_handle: Handle<Image> = asset_server.load(&request.texture_path);
    commands.insert_resource(LoadedTexture(texture_handle.clone()));

    // Create material with texture (will be applied later)
    let _material = _materials.add(StandardMaterial {
        base_color_texture: Some(texture_handle),
        unlit: true,
        ..default()
    });

    // Spawn the scene with the requested object transform (Bevy 0.15+ uses SceneRoot)
    commands.spawn((
        SceneRoot(scene_handle),
        request
            .object_rotation
            .to_transform_with_translation_scale(request.object_translation, request.object_scale),
        RenderedObject,
    ));

    println!("Scene setup complete");
}

/// Check if assets are loaded
fn check_assets_loaded(
    mut state: ResMut<RenderState>,
    asset_server: Res<AssetServer>,
    scene: Option<Res<LoadedScene>>,
    texture: Option<Res<LoadedTexture>>,
) {
    let trace = render_trace_enabled();
    let was_scene_loaded = state.scene_loaded;
    let was_texture_loaded = state.texture_loaded;

    state.frame_count += 1;

    if state.scene_loaded && state.texture_loaded {
        return;
    }

    if let Some(scene) = scene {
        match asset_server.get_load_state(&scene.0) {
            Some(LoadState::Loaded) => {
                state.scene_loaded = true;
            }
            Some(LoadState::Failed(_)) => {}
            _ => {}
        }
    }

    if let Some(texture) = texture {
        match asset_server.get_load_state(&texture.0) {
            Some(LoadState::Loaded) => {
                state.texture_loaded = true;
            }
            Some(LoadState::Failed(_)) => {}
            _ => {}
        }
    }

    if trace {
        if !was_scene_loaded && state.scene_loaded {
            eprintln!(
                "[render_trace][coldinit] scene_loaded frame_count={}",
                state.frame_count
            );
        }
        if !was_texture_loaded && state.texture_loaded {
            eprintln!(
                "[render_trace][coldinit] texture_loaded frame_count={}",
                state.frame_count
            );
        }
    }
}

/// Apply materials to loaded meshes
fn apply_materials(
    mut state: ResMut<RenderState>,
    texture: Option<Res<LoadedTexture>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    // Bevy 0.15+: Use MeshMaterial3d instead of Handle<StandardMaterial>
    mut mesh_query: Query<&mut MeshMaterial3d<StandardMaterial>, With<Mesh3d>>,
) {
    // NOTE: we intentionally do NOT wait for `texture_loaded` before applying the
    // material. The texture *handle* is valid immediately, so applying the material
    // as soon as the mesh entities exist lets the main-pass `StandardMaterial`
    // pipeline start compiling during the long async texture load. A late material
    // swap (after texture load) would reset the pipeline and capture a blank color
    // frame before it recompiled — the root cause of the 0.18 blank renders.
    if !state.scene_loaded || state.capture_ready {
        return;
    }

    state.frame_count += 1;

    let Some(tex) = texture else { return };

    if !state.materials_applied {
        // The scene hierarchy is instantiated asynchronously after the asset
        // load event fires; wait until mesh entities exist before applying.
        if mesh_query.is_empty() {
            return;
        }

        let textured_material = materials.add(StandardMaterial {
            base_color_texture: Some(tex.0.clone()),
            unlit: true,
            ..default()
        });

        for mut mat in mesh_query.iter_mut() {
            mat.0 = textured_material.clone();
        }

        state.materials_applied = true;
        state.materials_applied_frame = state.frame_count;
    }

    // Record the frame the texture finished loading (once).
    if state.texture_loaded && state.texture_ready_frame == 0 {
        state.texture_ready_frame = state.frame_count;
    }

    // Capture once the texture pixels are loaded (+ a small margin for GPU image
    // preparation) AND the main-pass pipeline has had time to compile since the
    // material was applied. Because the material is applied early, the pipeline is
    // almost always ready well before the texture, so this resolves to a few frames
    // after the texture loads — deterministic and fast (no 60/120-frame cushion).
    let texture_ready =
        state.texture_ready_frame != 0 && state.frame_count >= state.texture_ready_frame + 6;
    let pipeline_ready = state.frame_count >= state.materials_applied_frame + 6;
    if texture_ready && pipeline_ready {
        let was_ready = state.capture_ready;
        state.capture_ready = true;
        if render_trace_enabled() && !was_ready {
            eprintln!(
                "[render_trace][coldinit] capture_ready frame_count={}",
                state.frame_count
            );
        }
    }
}

/// Request a screenshot capture (Bevy 0.15+ uses Screenshot entity + observer)
#[allow(dead_code)]
fn request_screenshot(
    mut commands: Commands,
    mut state: ResMut<RenderState>,
    shared_image: Res<SharedImageBuffer>,
    mut depth_request: ResMut<DepthCaptureRequest>,
) {
    if !state.capture_ready || state.screenshot_requested {
        return;
    }

    // Clone the Arc for the observer closure
    let image_buffer = shared_image.0.clone();

    // Also request depth capture
    depth_request.requested = true;
    println!("Depth capture requested");

    // Spawn Screenshot entity with observer (Bevy 0.15+ API)
    println!("Requesting screenshot via Screenshot entity");
    commands
        .spawn(Screenshot::primary_window())
        .observe(move |trigger: On<ScreenshotCaptured>| {
            // ScreenshotCaptured derefs to Image
            let image: &Image = trigger.event();

            // Get dimensions
            let width = image.texture_descriptor.size.width;
            let height = image.texture_descriptor.size.height;

            // Bevy 0.18: Image.data is now Option<Vec<u8>>; skip if absent.
            let Some(rgba_data) = image.data.clone() else {
                return;
            };

            // Store in shared buffer
            if let Ok(mut guard) = image_buffer.lock() {
                *guard = Some((rgba_data, width, height));
            }
        });

    state.screenshot_requested = true;
    println!("Screenshot requested");
}

/// Check if screenshot callback has completed
#[allow(dead_code)]
fn check_screenshot_ready(
    mut state: ResMut<RenderState>,
    shared_image: Res<SharedImageBuffer>,
    shared_depth: Res<SharedDepthBuffer>,
    request: Res<RenderRequest>,
) {
    if !state.screenshot_requested || state.captured {
        return;
    }

    // Increment frame count while waiting for capture
    state.frame_count += 1;

    // Check if RGBA callback has written data
    let rgba_ready = if let Ok(guard) = shared_image.0.lock() {
        if let Some((rgba_data, width, height)) = guard.as_ref() {
            if state.rgba_data.is_none() {
                state.rgba_data = Some(rgba_data.clone());
                state.image_width = *width;
                state.image_height = *height;
            }
            true
        } else {
            false
        }
    } else {
        false
    };

    // Check if depth readback has completed
    let depth_ready = if let Ok(guard) = shared_depth.0.lock() {
        if let Some((depth_data, _width, _height)) = guard.as_ref() {
            if state.depth_data.is_none() {
                state.depth_data = Some(depth_data.clone());
            }
            true
        } else {
            false
        }
    } else {
        false
    };

    // If depth readback failed or is taking too long, fall back to placeholder.
    // As in check_headless_capture_ready, this uniform plane is a DEGRADED render
    // (flat depth, no real geometry) that must be loud — it silently masked the
    // #92 depth regression. (This fn is currently dead code; kept loud in case it
    // is ever revived.)
    if rgba_ready && !depth_ready && state.frame_count > 60 {
        let camera_dist = request.camera_transform.translation.length() as f64;
        let pixel_count = (state.image_width * state.image_height) as usize;
        eprintln!(
            "[bevy-sensor][WARN] depth readback produced no valid frame; falling back to a \
             UNIFORM {:.4} m camera-distance plane (degraded render, no real 3D geometry). \
             Indicates a depth-readback regression.",
            camera_dist
        );
        state.depth_data = Some(vec![camera_dist; pixel_count]);
    }

    // Mark as captured when both RGBA and depth are ready
    if state.rgba_data.is_some() && state.depth_data.is_some() {
        state.captured = true;
    }
}

/// Extract results and exit
#[allow(dead_code)]
fn extract_and_exit(
    mut state: ResMut<RenderState>,
    request: Res<RenderRequest>,
    shared_output: Res<SharedOutput>,
    mut commands: Commands,
    windows: Query<Entity, With<bevy::window::Window>>,
) {
    // Handle delayed exit after closing window
    if state.exit_requested {
        state.exit_frame_count += 1;
        // After a few frames with no window, Bevy should exit
        return;
    }

    if !state.captured {
        return;
    }

    if let (Some(rgba), Some(depth)) = (&state.rgba_data, &state.depth_data) {
        // Use actual captured dimensions (may differ from config if window was resized)
        let width = state.image_width;
        let height = state.image_height;

        // Compute intrinsics from the same TBP zoom formula as the camera projection.
        let intrinsics = request.config.intrinsics_for_size(width, height);

        let output = RenderOutput {
            rgba: rgba.clone(),
            depth: depth.clone(),
            width,
            height,
            intrinsics,
            camera_transform: request.camera_transform,
            object_rotation: request.object_rotation.clone(),
            object_translation: request.object_translation,
            object_scale: request.object_scale,
            target_point: Vec3::ZERO,
            targeting_policy: TargetingPolicy::Origin,
        };

        if let Ok(mut guard) = shared_output.0.lock() {
            *guard = Some(output);
            drop(guard); // Release lock immediately

            // Small delay to allow watchdog to detect output before window close
            std::thread::sleep(std::time::Duration::from_millis(200));
        }

        // Close all windows to trigger app exit
        // eprintln!("Closing windows to trigger exit...");
        for window_entity in windows.iter() {
            commands.entity(window_entity).despawn();
        }
        state.exit_requested = true;
    }
}

// ============================================================================
// Headless Rendering Systems (no window surfaces)
// ============================================================================

/// Setup the scene for headless rendering with RenderTarget::Image
fn setup_headless_scene(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    asset_server: Res<AssetServer>,
    request: Res<RenderRequest>,
    mut _materials: ResMut<Assets<StandardMaterial>>,
) {
    let trace = render_trace_enabled();
    let t0 = trace.then(std::time::Instant::now);

    #[cfg(test)]
    HEADLESS_SCENE_SETUP_COUNT.fetch_add(1, Ordering::SeqCst);

    let width = request.config.width;
    let height = request.config.height;

    // Create render target image with proper texture usages
    let size = Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    let mut render_target_image = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0, 0, 0, 255], // Initialize with opaque black
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );

    // Add required texture usages for headless rendering
    render_target_image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC | TextureUsages::RENDER_ATTACHMENT;

    let render_target_handle = images.add(render_target_image);

    // Store handle for later access
    commands.insert_resource(RenderTargetImage(render_target_handle.clone()));

    // Camera rendering to the image texture (NO window!)
    commands.spawn((
        Camera3d::default(),
        Camera::default(),
        Hdr,
        // In Bevy 0.18 the render target is a separate `RenderTarget` component,
        // and `RenderTarget::Image` wraps an `ImageRenderTarget` (via `From<Handle<Image>>`).
        RenderTarget::Image(render_target_handle.clone().into()),
        render_projection(&request.config),
        Msaa::Off,
        request.camera_transform,
        Tonemapping::None,
        DepthPrepass,
        NormalPrepass,
        RenderCamera,
        // Add ImageCopier to trigger RGBA extraction
        ImageCopier {
            src_image: render_target_handle,
            enabled: false, // Will enable when ready to capture
        },
    ));

    // Ambient light (global resource in Bevy 0.18).
    let lighting = &request.config.lighting;
    commands.insert_resource(GlobalAmbientLight {
        color: Color::WHITE,
        brightness: lighting.ambient_brightness,
        ..default()
    });

    // Key light
    if lighting.key_light_intensity > 0.0 {
        commands.spawn((
            PointLight {
                intensity: lighting.key_light_intensity,
                shadows_enabled: lighting.shadows_enabled,
                ..default()
            },
            Transform::from_xyz(
                lighting.key_light_position[0],
                lighting.key_light_position[1],
                lighting.key_light_position[2],
            ),
        ));
    }

    // Fill light
    if lighting.fill_light_intensity > 0.0 {
        commands.spawn((
            PointLight {
                intensity: lighting.fill_light_intensity,
                shadows_enabled: lighting.shadows_enabled,
                ..default()
            },
            Transform::from_xyz(
                lighting.fill_light_position[0],
                lighting.fill_light_position[1],
                lighting.fill_light_position[2],
            ),
        ));
    }

    // Load the scene
    let scene_handle: Handle<Scene> = asset_server.load(&request.mesh_path);
    commands.insert_resource(LoadedScene(scene_handle.clone()));

    // Load the texture
    let texture_handle: Handle<Image> = asset_server.load(&request.texture_path);
    commands.insert_resource(LoadedTexture(texture_handle.clone()));

    // Create material with texture
    let _material = _materials.add(StandardMaterial {
        base_color_texture: Some(texture_handle),
        unlit: true,
        ..default()
    });

    // Spawn the scene with the requested object transform
    commands.spawn((
        SceneRoot(scene_handle),
        request
            .object_rotation
            .to_transform_with_translation_scale(request.object_translation, request.object_scale),
        RenderedObject,
    ));

    if let Some(t0) = t0 {
        eprintln!(
            "[render_trace][startup] setup_headless_scene ms={:.3}",
            t0.elapsed().as_secs_f64() * 1000.0
        );
    }
}

/// Request capture for headless rendering (enable ImageCopier)
fn request_headless_capture(
    mut state: ResMut<RenderState>,
    mut depth_request: ResMut<DepthCaptureRequest>,
    mut query: Query<&mut ImageCopier>,
    batch: Option<Res<HeadlessBatchSequence>>,
) {
    let trace = render_trace_enabled();
    let t0 = trace.then(std::time::Instant::now);

    if !state.capture_ready || state.screenshot_requested {
        if let Some(t0) = t0 {
            eprintln!(
                "[render_trace][sys] request_headless_capture skipped(gate) ms={:.3}",
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }
        return;
    }

    if batch
        .as_ref()
        .is_some_and(|batch| batch.warmup_frames_remaining > 0)
    {
        if let Some(t0) = t0 {
            eprintln!(
                "[render_trace][sys] request_headless_capture skipped(warmup) ms={:.3}",
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }
        return;
    }

    // Enable the ImageCopier to trigger RGBA extraction
    for mut copier in query.iter_mut() {
        copier.enabled = true;
    }

    // Request depth capture
    depth_request.requested = true;

    state.screenshot_requested = true;

    if let Some(t0) = t0 {
        eprintln!(
            "[render_trace][sys] request_headless_capture requested ms={:.3}",
            t0.elapsed().as_secs_f64() * 1000.0
        );
    }
}

/// Check if headless capture has completed
fn check_headless_capture_ready(
    mut state: ResMut<RenderState>,
    shared_rgba: Res<SharedRgbaBuffer>,
    shared_depth: Res<SharedDepthBuffer>,
    request: Res<RenderRequest>,
    mut query: Query<&mut ImageCopier>,
) {
    let trace = render_trace_enabled();
    let t0 = trace.then(std::time::Instant::now);

    if !state.screenshot_requested || state.captured {
        if let Some(t0) = t0 {
            eprintln!(
                "[render_trace][sys] check_headless_capture_ready skipped(gate) ms={:.3}",
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }
        return;
    }

    state.frame_count += 1;
    state.capture_retries += 1;
    // Bounded fallback so a genuinely-uniform scene (or persistent invalid
    // readback) still terminates instead of hanging to the watchdog.
    // Generous bound: slow paths (e.g. RenderSession's retained-render-world
    // settle after a scene swap) can take ~150 frames to produce a stable frame,
    // so force-accepting at 150 would grab a partial frame and break parity. Only
    // force as a true last resort to avoid hanging the watchdog.
    let force_accept = state.capture_retries > 150;

    // RGBA: accept the first non-blank frame. Uniform clear-color frames are
    // pre-geometry reads from the nondeterministic one-shot capture — reject and
    // retry. The copier stays enabled until BOTH RGBA and depth are valid so a
    // late/odd depth frame can still be captured.
    if state.rgba_data.is_none() {
        let captured_rgba = shared_rgba.0.lock().ok().and_then(|mut g| g.take());
        if let Some((rgba_data, width, height)) = captured_rgba {
            let non_blank = rgba_data
                .chunks_exact(4)
                .any(|px| px[0..3] != rgba_data[0..3]);
            // Stable == identical to the previous readback (render has settled).
            let stable = state.prev_rgba.as_deref() == Some(rgba_data.as_slice());
            if (non_blank && stable) || force_accept {
                state.image_width = width;
                state.image_height = height;
                state.rgba_data = Some(rgba_data);
                state.prev_rgba = None;
            } else {
                // Not settled yet: remember this frame and re-read fresh next one.
                state.prev_rgba = Some(rgba_data);
            }
        }
    }

    // Depth: accept the first readback that contains real foreground (the depth
    // readback can also miss the geometry, leaving an all-far-plane buffer).
    if state.depth_data.is_none() {
        let captured_depth = shared_depth.0.lock().ok().and_then(|mut g| g.take());
        if let Some((depth_data, _w, _h)) = captured_depth {
            let far = request.config.far_plane as f64;
            // Require a real object-surface depth, not just any non-far value:
            // near-plane garbage (~0.01) would otherwise be accepted but is not a
            // valid surface, and downstream depth-validity checks require > 0.1m.
            let has_foreground = depth_data.iter().any(|&d| d > 0.1 && d < far * 0.999);
            // Settled == identical to the previous depth readback.
            let stable = state.prev_depth.as_deref() == Some(depth_data.as_slice());
            if has_foreground && stable {
                state.depth_data = Some(depth_data);
                state.prev_depth = None;
            } else {
                state.prev_depth = Some(depth_data);
            }
        }
    }

    // Last-resort fallback so we never hang the watchdog: once RGBA is in hand
    // and we've retried a lot, fill a uniform camera-distance depth placeholder.
    //
    // This is NOT a valid render — it is a flat depth plane that extracts
    // features and passes buffer-equality parity tests yet unprojects every
    // pixel onto one sheet, silently cratering downstream spatial matching
    // (this exact fallback masked the Bevy 0.18 depth regression in #92). It
    // must therefore be LOUD: a future depth-readback regression has to surface
    // in logs/CI instead of looking like a successful render. `tests/
    // spatial_parity.rs` is the geometric guard for the same failure.
    if state.rgba_data.is_some() && state.depth_data.is_none() && force_accept {
        let camera_dist = request.camera_transform.translation.length() as f64;
        let pixel_count = (state.image_width * state.image_height) as usize;
        eprintln!(
            "[bevy-sensor][WARN] depth readback produced no valid frame after {} retries; \
             falling back to a UNIFORM {:.4} m camera-distance plane. This is a degraded \
             render (flat depth -> no real 3D geometry) and indicates a depth-readback \
             regression. See render.rs DepthReadbackNode and tests/spatial_parity.rs.",
            state.capture_retries, camera_dist
        );
        state.depth_data = Some(vec![camera_dist; pixel_count]);
    }

    let rgba_ready = state.rgba_data.is_some();
    let depth_ready = state.depth_data.is_some();

    // Both valid → capture complete; stop the copier.
    if rgba_ready && depth_ready {
        state.captured = true;
        for mut copier in query.iter_mut() {
            copier.enabled = false;
        }
    }

    if let Some(t0) = t0 {
        eprintln!(
            "[render_trace][sys] check_headless_capture_ready rgba_ready={} depth_ready={} captured={} frame_count={} ms={:.3}",
            rgba_ready,
            depth_ready,
            state.captured,
            state.frame_count,
            t0.elapsed().as_secs_f64() * 1000.0
        );
    }
}

/// Extract results and exit for headless rendering
fn extract_and_exit_headless(
    mut state: ResMut<RenderState>,
    request: Res<RenderRequest>,
    shared_output: Res<SharedOutput>,
    mut app_exit: MessageWriter<bevy::app::AppExit>,
    batch: Option<Res<HeadlessBatchSequence>>,
) {
    if batch.is_some() {
        return;
    }

    if state.exit_requested {
        return;
    }

    if !state.captured {
        return;
    }

    if state.rgba_data.is_some() && state.depth_data.is_some() {
        let width = state.image_width;
        let height = state.image_height;
        let rgba = state.rgba_data.take().expect("checked rgba_data");
        let depth = state.depth_data.take().expect("checked depth_data");

        // Compute intrinsics from the same TBP zoom formula as the camera projection.
        let intrinsics = request.config.intrinsics_for_size(width, height);

        let output = RenderOutput {
            rgba,
            depth,
            width,
            height,
            intrinsics,
            camera_transform: request.camera_transform,
            object_rotation: request.object_rotation.clone(),
            object_translation: request.object_translation,
            object_scale: request.object_scale,
            target_point: Vec3::ZERO,
            targeting_policy: TargetingPolicy::Origin,
        };

        if let Ok(mut guard) = shared_output.0.lock() {
            *guard = Some(output);
            drop(guard);
            std::thread::sleep(std::time::Duration::from_millis(200));
        }

        // Send AppExit event (headless apps use this instead of closing windows)
        app_exit.write(bevy::app::AppExit::Success);
        state.exit_requested = true;
    }
}

/// Advance the short post-camera-move warmup for homogeneous batch rendering.
fn tick_headless_batch_warmup(batch: Option<ResMut<HeadlessBatchSequence>>) {
    let Some(mut batch) = batch else {
        return;
    };

    if batch.warmup_frames_remaining > 0 {
        batch.warmup_frames_remaining -= 1;
    }
}

/// Extract one batch output and continue rendering the next viewpoint in the same app.
fn extract_and_continue_headless_batch(
    mut state: ResMut<RenderState>,
    request: Res<RenderRequest>,
    buffers: (Res<SharedRgbaBuffer>, Res<SharedDepthBuffer>),
    batch: Option<ResMut<HeadlessBatchSequence>>,
    mut camera_query: Query<&mut Transform, With<RenderCamera>>,
    mut depth_request: ResMut<DepthCaptureRequest>,
    mut image_copiers: Query<&mut ImageCopier>,
) {
    let trace = render_trace_enabled();
    let t0 = trace.then(std::time::Instant::now);

    let (shared_rgba, shared_depth) = buffers;
    let Some(mut batch) = batch else {
        if let Some(t0) = t0 {
            eprintln!(
                "[render_trace][sys] extract_and_continue_headless_batch skipped(no_batch) ms={:.3}",
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }
        return;
    };

    if state.exit_requested || !state.captured || batch.done {
        if let Some(t0) = t0 {
            eprintln!(
                "[render_trace][sys] extract_and_continue_headless_batch skipped(gate) captured={} done={} ms={:.3}",
                state.captured,
                batch.done,
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }
        return;
    }

    if state.rgba_data.is_some() && state.depth_data.is_some() {
        let width = state.image_width;
        let height = state.image_height;
        let rgba = state.rgba_data.take().expect("checked rgba_data");
        let depth = state.depth_data.take().expect("checked depth_data");

        let intrinsics = request.config.intrinsics_for_size(width, height);

        let output = RenderOutput {
            rgba,
            depth,
            width,
            height,
            intrinsics,
            camera_transform: batch
                .current_viewpoint()
                .unwrap_or(request.camera_transform),
            object_rotation: request.object_rotation.clone(),
            object_translation: request.object_translation,
            object_scale: request.object_scale,
            target_point: Vec3::ZERO,
            targeting_policy: TargetingPolicy::Origin,
        };
        batch.outputs.push(output);

        let next_index = batch.current_index + 1;
        if next_index >= batch.viewpoints.len() {
            batch.done = true;
            state.exit_requested = true;
            return;
        }

        batch.current_index = next_index;
        batch.warmup_frames_remaining = BATCH_WARMUP_FRAMES;

        if let Some(next_viewpoint) = batch.current_viewpoint() {
            for mut camera_transform in camera_query.iter_mut() {
                *camera_transform = next_viewpoint;
            }
        }

        if let Ok(mut guard) = shared_rgba.0.lock() {
            *guard = None;
        }
        if let Ok(mut guard) = shared_depth.0.lock() {
            *guard = None;
        }

        for mut copier in image_copiers.iter_mut() {
            copier.enabled = false;
        }

        depth_request.requested = false;
        state.frame_count = 0;
        state.capture_ready = true;
        state.screenshot_requested = false;
        state.captured = false;
        state.rgba_data = None;
        state.depth_data = None;
        state.image_width = 0;
        state.image_height = 0;
        // Reset the per-capture settle/retry tracking too, otherwise it
        // accumulates across viewpoints and force-accepts an unsettled frame for
        // later viewpoints (breaking parity).
        state.capture_retries = 0;
        state.prev_rgba = None;
        state.prev_depth = None;

        if let Some(t0) = t0 {
            eprintln!(
                "[render_trace][sys] extract_and_continue_headless_batch extracted vp={} next={} done={} ms={:.3}",
                batch.current_index.saturating_sub(1),
                batch.current_index,
                batch.done,
                t0.elapsed().as_secs_f64() * 1000.0
            );
        }
    } else if let Some(t0) = t0 {
        eprintln!(
            "[render_trace][sys] extract_and_continue_headless_batch no_data ms={:.3}",
            t0.elapsed().as_secs_f64() * 1000.0
        );
    }
}

// ============================================================================
// Persistent batch session (RenderSession)
//
// Amortizes wgpu device creation, Bevy app setup, and first-draw pipeline state
// object (PSO) compilation across multiple `render()` calls. Profile data (see
// issues #54 and #55) showed that on a 60-episode parity-gate, ~2.3s per episode
// lives in first-draw DX12 PSO compilation, totalling ~131s of 151s wall-clock.
// Keeping the `App` (and thus the `RenderDevice` and its PSO cache) alive across
// episodes recovers the bulk of that cost.
// ============================================================================

/// Marker for the per-group scene entity so we can despawn it cleanly when the
/// next `RenderSession::render()` call swaps in a different object or rotation.
#[derive(Component)]
struct SessionScene;

/// Session-persistent setup: render target image, camera (with prepass +
/// `ImageCopier`), ambient light, key + fill lights. Everything here lives for
/// the full lifetime of the `RenderSession`; per-group work (mesh/texture load,
/// scene entity spawn) happens outside Startup in `RenderSession::render()`.
fn setup_session_persistent_scene(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    config: Res<SessionRenderConfig>,
) {
    let width = config.0.width;
    let height = config.0.height;

    let size = Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    let mut render_target_image = Image::new_fill(
        size,
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    render_target_image.texture_descriptor.usage =
        TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_SRC | TextureUsages::RENDER_ATTACHMENT;

    let render_target_handle = images.add(render_target_image);
    commands.insert_resource(RenderTargetImage(render_target_handle.clone()));

    commands.spawn((
        Camera3d::default(),
        Camera::default(),
        Hdr,
        RenderTarget::Image(render_target_handle.clone().into()),
        render_projection(&config.0),
        Msaa::Off,
        Transform::default(),
        Tonemapping::None,
        DepthPrepass,
        NormalPrepass,
        RenderCamera,
        ImageCopier {
            src_image: render_target_handle,
            enabled: false,
        },
    ));

    let lighting = &config.0.lighting;
    commands.insert_resource(GlobalAmbientLight {
        color: Color::WHITE,
        brightness: lighting.ambient_brightness,
        ..default()
    });

    if lighting.key_light_intensity > 0.0 {
        commands.spawn((
            PointLight {
                intensity: lighting.key_light_intensity,
                shadows_enabled: lighting.shadows_enabled,
                ..default()
            },
            Transform::from_xyz(
                lighting.key_light_position[0],
                lighting.key_light_position[1],
                lighting.key_light_position[2],
            ),
        ));
    }

    if lighting.fill_light_intensity > 0.0 {
        commands.spawn((
            PointLight {
                intensity: lighting.fill_light_intensity,
                shadows_enabled: lighting.shadows_enabled,
                ..default()
            },
            Transform::from_xyz(
                lighting.fill_light_position[0],
                lighting.fill_light_position[1],
                lighting.fill_light_position[2],
            ),
        ));
    }
}

/// Resource carrying the `RenderConfig` that was fixed at session construction.
/// Used by `setup_session_persistent_scene` to size the render target.
#[derive(Resource)]
struct SessionRenderConfig(RenderConfig);

/// Persistent batch render session. Keeps a Bevy `App` (and its `RenderDevice`
/// plus PSO cache) alive across multiple `render()` calls, amortizing per-episode
/// cold-init cost.
///
/// # Thread affinity
///
/// `RenderSession` must be created, used, and dropped on the same thread. It
/// holds a `bevy::App` which owns GPU resources that are not safe to move
/// across threads. The `!Send + !Sync` marker is enforced via
/// `PhantomData<*const ()>`.
///
/// # Config invariant
///
/// The `RenderConfig` (resolution, lighting, near/far, fov) is fixed at
/// `new()`. All `render()` calls must use requests whose `render_config`
/// matches; heterogeneous configs are rejected.
///
/// # Phase 1 limitation
///
/// Each `render()` call must contain homogeneous requests (same `object_dir`
/// and `object_rotation`). Heterogeneous calls return
/// `BatchRenderError::InvalidConfig`. Hold a single `RenderSession` and call
/// `render()` once per episode to amortize setup across episodes.
pub struct RenderSession {
    app: App,
    render_config: RenderConfig,
    shared_rgba: SharedRgbaBuffer,
    shared_depth: SharedDepthBuffer,
    _not_send_sync: std::marker::PhantomData<*const ()>,
}

impl RenderSession {
    /// Build the App, run plugin `finish()`/`cleanup()`, and perform one warmup
    /// `update()` so Startup systems run and the wgpu device + adapter are
    /// initialized. The first `render()` call still pays PSO compilation for
    /// the specific mesh/material combination; subsequent calls reuse the cache.
    pub fn new(render_config: &crate::RenderConfig) -> Result<Self, crate::RenderError> {
        let shared_rgba: SharedRgbaBuffer = SharedRgbaBuffer::default();
        let shared_depth: SharedDepthBuffer = SharedDepthBuffer::default();

        let mut app = App::new();
        app.add_plugins(
            DefaultPlugins
                .set(bevy::asset::AssetPlugin {
                    // Bevy 0.17+ forbids loading from absolute / `..` asset paths by
                    // default (UnapprovedPathMode::Forbid → load() silently returns a
                    // default handle). YCB meshes load from absolute paths, so allow them.
                    unapproved_path_mode: bevy::asset::UnapprovedPathMode::Allow,
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .disable::<bevy::winit::WinitPlugin>()
                .disable::<LogPlugin>()
                .disable::<TerminalCtrlCHandlerPlugin>(),
        )
        .add_plugins(ObjPlugin)
        // bevy_obj's Scene contains Mesh3d + MeshMaterial3d entities; reflection-based
        // Scene spawning panics unless those component types are registered. The
        // minimal headless plugin set doesn't register them, so do it explicitly.
        .register_type::<Mesh3d>()
        .register_type::<MeshMaterial3d<StandardMaterial>>()
        .register_type::<bevy::prelude::Transform>()
        .register_type::<bevy::prelude::GlobalTransform>()
        .register_type::<bevy::transform::components::TransformTreeChanged>()
        .register_type::<bevy::prelude::Visibility>()
        .register_type::<bevy::prelude::InheritedVisibility>()
        .register_type::<bevy::prelude::ViewVisibility>()
        .add_plugins(ImageCopyPlugin {
            shared_rgba: shared_rgba.clone(),
        })
        .add_plugins(DepthReadbackPlugin {
            shared_depth: shared_depth.clone(),
            near: render_config.near_plane,
            far: render_config.far_plane,
        })
        .insert_resource(SessionRenderConfig(render_config.clone()))
        .insert_resource(shared_rgba.clone())
        .init_resource::<RenderState>()
        .add_systems(Startup, setup_session_persistent_scene)
        .add_systems(
            Update,
            (
                check_assets_loaded,
                apply_materials,
                tick_headless_batch_warmup,
                request_headless_capture,
                check_headless_capture_ready,
                extract_and_continue_headless_batch,
            )
                .chain()
                // Gate the capture chain on `RenderRequest` existing. `new()`
                // runs a warmup `app.update()` to execute Startup (which spawns
                // the camera/lights/render target) before the first `render()`
                // call, but does not yet insert `RenderRequest`. Several systems
                // in this chain take `Res<RenderRequest>` (not `Option`) and
                // would panic on SystemState init if the resource were absent.
                .run_if(bevy::ecs::schedule::common_conditions::resource_exists::<RenderRequest>),
        );

        app.finish();
        app.cleanup();

        // One warmup update runs Startup systems (render target, camera, lights)
        // so they exist before the first `render()` call seeds the camera
        // transform. The Update chain is gated by `RenderRequest` existence and
        // is a no-op this tick. PSO compilation for specific mesh/material
        // combinations still happens lazily on the first real render.
        app.update();

        Ok(Self {
            app,
            render_config: render_config.clone(),
            shared_rgba,
            shared_depth,
            _not_send_sync: std::marker::PhantomData,
        })
    }

    /// Render a homogeneous batch of viewpoints (same object + rotation + config).
    /// Returns outputs in request order.
    ///
    /// On `BatchRenderError::DeviceLost`, the returned error signals that the
    /// wgpu device was lost mid-render. This call produced no output; any
    /// outputs from earlier `render()` calls on this session are still valid.
    /// Recovery: drop this `RenderSession` and construct a new one.
    pub fn render(
        &mut self,
        requests: &[crate::BatchRenderRequest],
    ) -> Result<Vec<crate::BatchRenderOutput>, crate::BatchRenderError> {
        use crate::{BatchRenderError, BatchRenderOutput};

        if requests.is_empty() {
            return Ok(Vec::new());
        }

        // Enforce homogeneity and config invariance.
        let first = &requests[0];
        if first.render_config != self.render_config {
            return Err(BatchRenderError::InvalidConfig(
                "RenderSession render_config mismatch: session was constructed with a different \
                 RenderConfig than the first request carries. Session config cannot change after \
                 `new()`; construct a new session if you need a different resolution/camera."
                    .to_string(),
            ));
        }
        for r in &requests[1..] {
            if r.object_dir != first.object_dir
                || r.object_rotation != first.object_rotation
                || r.object_translation != first.object_translation
                || r.object_scale != first.object_scale
                || r.render_config != first.render_config
            {
                return Err(BatchRenderError::InvalidConfig(
                    "Phase 1 RenderSession::render requires homogeneous requests \
                     (same object_dir, object transform, and render_config across the batch). \
                     Call render() once per group instead."
                        .to_string(),
                ));
            }
        }

        // Canonicalize paths and validate mesh/texture presence. This matches
        // `render_headless_sequence`'s preconditions so the error surface stays
        // consistent.
        let object_dir = std::fs::canonicalize(&first.object_dir).map_err(|e| {
            BatchRenderError::InvalidConfig(format!(
                "Cannot canonicalize object directory {}: {}",
                first.object_dir.display(),
                e
            ))
        })?;
        let mesh_path = object_dir.join(GOOGLE_16K_MESH_RELATIVE);
        let texture_path = object_dir.join(GOOGLE_16K_TEXTURE_RELATIVE);
        if !mesh_path.exists() {
            return Err(BatchRenderError::InvalidConfig(format!(
                "Mesh not found: {}",
                mesh_path.display()
            )));
        }
        if !texture_path.exists() {
            return Err(BatchRenderError::InvalidConfig(format!(
                "Texture not found: {}",
                texture_path.display()
            )));
        }

        let viewpoints: Vec<Transform> = requests.iter().map(|r| r.viewpoint).collect();

        // --- per-group scene swap (direct world manipulation) ---
        {
            let world = self.app.world_mut();

            // Despawn any SessionScene entity from the previous group.
            let stale: Vec<Entity> = world
                .query_filtered::<Entity, With<SessionScene>>()
                .iter(world)
                .collect();
            for entity in stale {
                world.entity_mut(entity).despawn();
            }

            // Clear shared RGBA/depth buffers so a stale payload can't leak
            // into the first viewpoint of this call.
            if let Ok(mut guard) = self.shared_rgba.0.lock() {
                *guard = None;
            }
            if let Ok(mut guard) = self.shared_depth.0.lock() {
                *guard = None;
            }

            // Reset RenderState (scene_loaded, texture_loaded, capture_ready,
            // frame_count, materials_applied, etc.). Default() gives all false/0.
            *world.resource_mut::<RenderState>() = RenderState::default();

            // Update RenderRequest so the existing capture systems see the new
            // object paths, rotation, and camera transform (seeded from first vp).
            let new_request = RenderRequest {
                mesh_path: fs_path_to_asset_string(&mesh_path),
                texture_path: fs_path_to_asset_string(&texture_path),
                camera_transform: viewpoints[0],
                object_rotation: first.object_rotation.clone(),
                object_translation: first.object_translation,
                object_scale: first.object_scale,
                config: self.render_config.clone(),
            };
            world.insert_resource(new_request);

            // Kick off asset loads and install the handles under the names the
            // existing `check_assets_loaded` system expects.
            let asset_server = world.resource::<AssetServer>().clone();
            let scene_handle: Handle<Scene> =
                asset_server.load(fs_path_to_asset_string(&mesh_path));
            let texture_handle: Handle<Image> =
                asset_server.load(fs_path_to_asset_string(&texture_path));
            world.insert_resource(LoadedScene(scene_handle.clone()));
            world.insert_resource(LoadedTexture(texture_handle));

            // Spawn the new scene entity tagged so we can find + despawn it next
            // render() call.
            world.spawn((
                SceneRoot(scene_handle),
                first.object_rotation.to_transform_with_translation_scale(
                    first.object_translation,
                    first.object_scale,
                ),
                RenderedObject,
                SessionScene,
            ));

            // Seed the camera transform to the first viewpoint now so the first
            // capture lines up; subsequent viewpoints are advanced by
            // `extract_and_continue_headless_batch`.
            let camera_entity = world
                .query_filtered::<Entity, With<RenderCamera>>()
                .iter(world)
                .next();
            if let Some(cam) = camera_entity {
                if let Some(mut transform) = world.entity_mut(cam).get_mut::<Transform>() {
                    *transform = viewpoints[0];
                }
            }

            // Install the viewpoint sequence for this render() call. The robust
            // settled-frame capture (reject blank/partial readbacks, retry until
            // two consecutive readbacks match) absorbs the despawn/respawn
            // render-world settle, so a separate discarded warmup pass is not
            // needed and the per-object cost stays low.
            world.insert_resource(HeadlessBatchSequence::new(viewpoints.clone()));
        }

        // --- drive the real capture loop ---
        let timeout = std::time::Duration::from_secs(RENDER_TIMEOUT_SECS);
        let start = std::time::Instant::now();
        loop {
            if start.elapsed() > timeout {
                return Err(BatchRenderError::TotalFailure(format!(
                    "RenderSession::render timed out after {}s",
                    RENDER_TIMEOUT_SECS
                )));
            }

            self.app.update();

            if self.app.world().resource::<HeadlessBatchSequence>().done {
                break;
            }
        }

        // Collect outputs and zip with requests to produce BatchRenderOutput in
        // request order.
        let mut sequence = self.app.world_mut().resource_mut::<HeadlessBatchSequence>();
        if sequence.outputs.len() != requests.len() {
            return Err(BatchRenderError::TotalFailure(format!(
                "RenderSession produced {} outputs for {} requests",
                sequence.outputs.len(),
                requests.len()
            )));
        }
        let outputs = std::mem::take(&mut sequence.outputs);

        Ok(requests
            .iter()
            .cloned()
            .zip(outputs)
            .map(|(req, out)| BatchRenderOutput::from_render_output(req, out))
            .collect())
    }
}

// ============================================================================
// Per-step persistent renderer (PersistentRenderer)
//
// `RenderSession` reuses the App across calls but rebuilds the scene on every
// `render()` (despawn SceneRoot, re-issue asset_server.load, respawn). That's
// fine for the parity-gate path (one scene per episode of N viewpoints) but
// wasteful for surface-policy feedback loops where N=1 viewpoint per call and
// the object stays loaded for the whole episode.
//
// `PersistentRenderer` commits to one `object_dir` + `RenderConfig` at
// construction. `new()` loads mesh + texture + spawns the scene root + drives
// one warmup render (output discarded) so PSO compilation and material setup
// are paid up front. `render(camera, rotation)` then only mutates the camera
// `Transform` and (if changed) the scene root rotation, drives the capture
// chain for one frame, and returns. See issue #65.
// ============================================================================

/// Marker for the `PersistentRenderer`'s scene root entity. We keep the
/// entity alive for the whole renderer lifetime and just mutate its
/// `Transform` when the caller-supplied object rotation changes.
#[derive(Component)]
struct PersistentScene;

/// Persistent per-step renderer. Loads the scene once at `new()` and renders
/// one frame per `render()` call by mutating the camera transform and scene
/// root rotation in-place. Built for surface-policy feedback loops where the
/// object stays fixed for the duration of an episode and the camera moves
/// every step. See issue #65.
///
/// # Thread affinity
///
/// `PersistentRenderer` must be created, used, and dropped on the same thread.
/// Holds a `bevy::App` that owns GPU resources not safe to move across
/// threads; `!Send + !Sync` is enforced via `PhantomData<*const ()>`.
///
/// # Object + config invariants
///
/// `object_dir` and `RenderConfig` are fixed at `new()`. To render a different
/// object or change resolution/lighting, drop and rebuild. Rotation may change
/// freely between `render()` calls.
pub struct PersistentRenderer {
    app: App,
    object_dir: PathBuf,
    render_config: RenderConfig,
    shared_rgba: SharedRgbaBuffer,
    shared_depth: SharedDepthBuffer,
    _not_send_sync: std::marker::PhantomData<*const ()>,
}

impl PersistentRenderer {
    /// Build the App, load the scene + texture, spawn the scene root, and drive
    /// one warmup render whose output is discarded. After `new()` returns, the
    /// first user-facing `render()` call benefits from a warm PSO cache and
    /// applied materials.
    pub fn new(
        object_dir: &Path,
        render_config: &RenderConfig,
    ) -> Result<Self, crate::RenderError> {
        let object_dir =
            std::fs::canonicalize(object_dir).map_err(|e| crate::RenderError::FileNotFound {
                path: object_dir.display().to_string(),
                reason: e.to_string(),
            })?;
        let mesh_path = object_dir.join(GOOGLE_16K_MESH_RELATIVE);
        let texture_path = object_dir.join(GOOGLE_16K_TEXTURE_RELATIVE);
        if !mesh_path.exists() {
            return Err(crate::RenderError::MeshNotFound(fs_path_to_asset_string(
                &mesh_path,
            )));
        }
        if !texture_path.exists() {
            return Err(crate::RenderError::TextureNotFound(
                fs_path_to_asset_string(&texture_path),
            ));
        }

        let shared_rgba: SharedRgbaBuffer = SharedRgbaBuffer::default();
        let shared_depth: SharedDepthBuffer = SharedDepthBuffer::default();

        let mut app = App::new();
        app.add_plugins(
            DefaultPlugins
                .set(bevy::asset::AssetPlugin {
                    // Bevy 0.17+ forbids loading from absolute / `..` asset paths by
                    // default (UnapprovedPathMode::Forbid → load() silently returns a
                    // default handle). YCB meshes load from absolute paths, so allow them.
                    unapproved_path_mode: bevy::asset::UnapprovedPathMode::Allow,
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .disable::<bevy::winit::WinitPlugin>()
                .disable::<LogPlugin>()
                .disable::<TerminalCtrlCHandlerPlugin>(),
        )
        .add_plugins(ObjPlugin)
        // bevy_obj's Scene contains Mesh3d + MeshMaterial3d entities; reflection-based
        // Scene spawning panics unless those component types are registered. The
        // minimal headless plugin set doesn't register them, so do it explicitly.
        .register_type::<Mesh3d>()
        .register_type::<MeshMaterial3d<StandardMaterial>>()
        .register_type::<bevy::prelude::Transform>()
        .register_type::<bevy::prelude::GlobalTransform>()
        .register_type::<bevy::transform::components::TransformTreeChanged>()
        .register_type::<bevy::prelude::Visibility>()
        .register_type::<bevy::prelude::InheritedVisibility>()
        .register_type::<bevy::prelude::ViewVisibility>()
        .add_plugins(ImageCopyPlugin {
            shared_rgba: shared_rgba.clone(),
        })
        .add_plugins(DepthReadbackPlugin {
            shared_depth: shared_depth.clone(),
            near: render_config.near_plane,
            far: render_config.far_plane,
        })
        .insert_resource(SessionRenderConfig(render_config.clone()))
        .insert_resource(shared_rgba.clone())
        .init_resource::<RenderState>()
        .add_systems(Startup, setup_session_persistent_scene)
        .add_systems(
            Update,
            (
                check_assets_loaded,
                apply_materials,
                tick_headless_batch_warmup,
                request_headless_capture,
                check_headless_capture_ready,
                extract_and_continue_headless_batch,
            )
                .chain()
                // Same gate as RenderSession: capture chain only runs once
                // RenderRequest is installed. Startup runs first via the
                // warmup `app.update()` below.
                .run_if(bevy::ecs::schedule::common_conditions::resource_exists::<RenderRequest>),
        );

        app.finish();
        app.cleanup();
        // Warmup tick #1: Startup runs (camera, lights, render target spawn).
        app.update();

        // Install scene + warmup render request. The warmup output is discarded
        // — its purpose is to pay PSO compilation and material application
        // upfront so the first user-facing render() is fast. Use a real TBP
        // viewpoint rather than Transform::default(), which places the camera
        // at the object origin and forces a flat-depth fallback before any
        // caller-requested surface-policy render runs.
        let warmup_camera = persistent_warmup_camera_transform();
        let initial_request = RenderRequest {
            mesh_path: fs_path_to_asset_string(&mesh_path),
            texture_path: fs_path_to_asset_string(&texture_path),
            camera_transform: warmup_camera,
            object_rotation: ObjectRotation::identity(),
            object_translation: Vec3::ZERO,
            object_scale: Vec3::ONE,
            config: render_config.clone(),
        };

        {
            let world = app.world_mut();
            let asset_server = world.resource::<AssetServer>().clone();
            let scene_handle: Handle<Scene> =
                asset_server.load(fs_path_to_asset_string(&mesh_path));
            let texture_handle: Handle<Image> =
                asset_server.load(fs_path_to_asset_string(&texture_path));
            world.insert_resource(LoadedScene(scene_handle.clone()));
            world.insert_resource(LoadedTexture(texture_handle));
            world.insert_resource(initial_request);
            world.spawn((
                SceneRoot(scene_handle),
                ObjectRotation::identity()
                    .to_transform_with_translation_scale(Vec3::ZERO, Vec3::ONE),
                RenderedObject,
                PersistentScene,
            ));
            if let Some(cam) = world
                .query_filtered::<Entity, With<RenderCamera>>()
                .iter(world)
                .next()
            {
                if let Some(mut transform) = world.entity_mut(cam).get_mut::<Transform>() {
                    *transform = warmup_camera;
                }
            }
            world.insert_resource(HeadlessBatchSequence::new(vec![warmup_camera]));
        }

        // Drive the warmup render to completion.
        let timeout = std::time::Duration::from_secs(RENDER_TIMEOUT_SECS);
        let start = std::time::Instant::now();
        loop {
            if start.elapsed() > timeout {
                return Err(crate::RenderError::RenderFailed(format!(
                    "PersistentRenderer::new warmup render timed out after {RENDER_TIMEOUT_SECS}s"
                )));
            }
            app.update();
            if app.world().resource::<HeadlessBatchSequence>().done {
                break;
            }
        }
        // Discard the warmup output so it doesn't leak into the first real
        // render() call's output buffer.
        app.world_mut()
            .resource_mut::<HeadlessBatchSequence>()
            .outputs
            .clear();

        Ok(Self {
            app,
            object_dir,
            render_config: render_config.clone(),
            shared_rgba,
            shared_depth,
            _not_send_sync: std::marker::PhantomData,
        })
    }

    /// Render one frame from the given camera transform and object rotation.
    /// Reuses the loaded scene + warm PSO cache from `new()`.
    pub fn render(
        &mut self,
        camera_transform: &Transform,
        object_rotation: &ObjectRotation,
    ) -> Result<RenderOutput, crate::RenderError> {
        self.render_with_object_transform(camera_transform, object_rotation, Vec3::ZERO, Vec3::ONE)
    }

    /// Render one frame with explicit object translation and scale.
    pub fn render_with_object_transform(
        &mut self,
        camera_transform: &Transform,
        object_rotation: &ObjectRotation,
        object_translation: Vec3,
        object_scale: Vec3,
    ) -> Result<RenderOutput, crate::RenderError> {
        let camera_transform = *camera_transform;
        let object_rotation_owned = object_rotation.clone();

        {
            let world = self.app.world_mut();

            // Update the persistent scene root rotation. Always-write avoids
            // the cost of an extra ObjectRotation comparison per call; the
            // mutation itself is a single Transform write.
            let scene_entity = world
                .query_filtered::<Entity, With<PersistentScene>>()
                .iter(world)
                .next();
            if let Some(entity) = scene_entity {
                if let Some(mut transform) = world.entity_mut(entity).get_mut::<Transform>() {
                    *transform = object_rotation_owned
                        .to_transform_with_translation_scale(object_translation, object_scale);
                }
            }

            // Update the camera transform.
            let cam_entity = world
                .query_filtered::<Entity, With<RenderCamera>>()
                .iter(world)
                .next();
            if let Some(cam) = cam_entity {
                if let Some(mut transform) = world.entity_mut(cam).get_mut::<Transform>() {
                    *transform = camera_transform;
                }
            }

            // Reset per-frame state, preserving scene_loaded / texture_loaded
            // / materials_applied / materials_applied_frame. The asset-load
            // and material-apply work was paid in `new()`'s warmup; we only
            // need to clear the per-capture state.
            //
            // `capture_ready = true` short-circuits `apply_materials` on
            // every tick of the render loop (no need to re-check material
            // application — it stays applied for the renderer's lifetime).
            // It does NOT short-circuit `request_headless_capture`, which
            // is gated by `HeadlessBatchSequence::warmup_frames_remaining`
            // below. Bug fix from PR #66 review (off-by-one / blank-step-0):
            // without that warmup gate, request_headless_capture fires same-
            // tick as the transform writes, capturing the previous render's
            // target before the new transforms have propagated.
            {
                let mut state = world.resource_mut::<RenderState>();
                state.exit_requested = false;
                state.screenshot_requested = false;
                state.captured = false;
                state.rgba_data = None;
                state.depth_data = None;
                state.frame_count = 0;
                state.image_width = 0;
                state.image_height = 0;
                state.capture_ready = true;
                state.capture_retries = 0;
                state.prev_rgba = None;
                state.prev_depth = None;
            }

            // Clear shared GPU readback buffers so a stale payload from the
            // previous render() can't leak into this call's output.
            if let Ok(mut guard) = self.shared_rgba.0.lock() {
                *guard = None;
            }
            if let Ok(mut guard) = self.shared_depth.0.lock() {
                *guard = None;
            }

            // Update RenderRequest (used by extract_and_continue_headless_batch
            // to stamp the output with the right intrinsics + rotation).
            {
                let mut req = world.resource_mut::<RenderRequest>();
                req.camera_transform = camera_transform;
                req.object_rotation = object_rotation_owned.clone();
                req.object_translation = object_translation;
                req.object_scale = object_scale;
            }

            // Install fresh single-element batch with warmup frames so
            // `request_headless_capture` is gated until the new transforms
            // have propagated through the render pipeline.
            let mut batch = HeadlessBatchSequence::new(vec![camera_transform]);
            batch.warmup_frames_remaining = PERSISTENT_WARMUP_FRAMES;
            world.insert_resource(batch);
        }

        let timeout = std::time::Duration::from_secs(RENDER_TIMEOUT_SECS);
        let start = std::time::Instant::now();
        loop {
            if start.elapsed() > timeout {
                return Err(crate::RenderError::RenderFailed(format!(
                    "PersistentRenderer::render timed out after {RENDER_TIMEOUT_SECS}s"
                )));
            }
            self.app.update();
            if self.app.world().resource::<HeadlessBatchSequence>().done {
                break;
            }
        }

        let mut sequence = self.app.world_mut().resource_mut::<HeadlessBatchSequence>();
        let mut outputs = std::mem::take(&mut sequence.outputs);
        if outputs.len() != 1 {
            return Err(crate::RenderError::RenderFailed(format!(
                "PersistentRenderer::render expected 1 output, got {}",
                outputs.len()
            )));
        }

        Ok(outputs.remove(0))
    }

    /// Path to the YCB object directory this renderer was bound to.
    pub fn object_dir(&self) -> &Path {
        &self.object_dir
    }

    /// The `RenderConfig` this renderer was constructed with.
    pub fn render_config(&self) -> &RenderConfig {
        &self.render_config
    }

    /// Explicit close. Equivalent to dropping; provided to match the API
    /// proposal in #65 for callers that want lifetime-explicit teardown.
    pub fn close(self) {
        // Drop runs on return.
    }
}

/// Render directly to files (for subprocess mode).
///
/// This function saves RGBA and depth data directly to files before exiting.
/// Designed for subprocess rendering where the process will exit after rendering.
#[allow(clippy::too_many_arguments)]
pub fn render_to_files(
    object_dir: &Path,
    camera_transform: &Transform,
    object_rotation: &ObjectRotation,
    object_translation: Vec3,
    object_scale: Vec3,
    config: &RenderConfig,
    rgba_path: &Path,
    depth_path: &Path,
) -> Result<(), RenderError> {
    let mesh_path = object_dir.join(GOOGLE_16K_MESH_RELATIVE);
    let texture_path = object_dir.join(GOOGLE_16K_TEXTURE_RELATIVE);

    if !mesh_path.exists() {
        return Err(RenderError::MeshNotFound(fs_path_to_asset_string(
            &mesh_path,
        )));
    }
    if !texture_path.exists() {
        return Err(RenderError::TextureNotFound(fs_path_to_asset_string(
            &texture_path,
        )));
    }

    let request = RenderRequest {
        mesh_path: fs_path_to_asset_string(&mesh_path),
        texture_path: fs_path_to_asset_string(&texture_path),
        camera_transform: *camera_transform,
        object_rotation: object_rotation.clone(),
        object_translation,
        object_scale,
        config: config.clone(),
    };

    // Shared state for output
    let shared_output: SharedOutput = SharedOutput(Arc::new(Mutex::new(None)));
    let output_poll = shared_output.clone();

    // Clone paths for watchdog thread
    let rgba_path = rgba_path.to_path_buf();
    let depth_path = depth_path.to_path_buf();

    // Shared buffer for RGBA data from headless render target
    let shared_rgba: SharedRgbaBuffer = SharedRgbaBuffer::default();

    // Shared buffer for depth readback
    let shared_depth: SharedDepthBuffer = SharedDepthBuffer::default();

    // Spawn watchdog thread that saves files and exits
    std::thread::spawn(move || {
        let timeout = std::time::Duration::from_secs(RENDER_TIMEOUT_SECS);
        let start = std::time::Instant::now();
        let poll_interval = std::time::Duration::from_millis(100);

        loop {
            if let Ok(guard) = output_poll.0.lock() {
                if let Some(output) = guard.as_ref() {
                    // Save RGBA as PNG
                    if let Err(e) =
                        save_rgba_to_png(&output.rgba, output.width, output.height, &rgba_path)
                    {
                        eprintln!("Failed to save RGBA: {:?}", e);
                        std::process::exit(1);
                    }

                    // Save depth as binary f32
                    if let Err(e) = save_depth_to_binary(&output.depth, &depth_path) {
                        eprintln!("Failed to save depth: {:?}", e);
                        std::process::exit(1);
                    }

                    std::process::exit(0);
                }
            }

            if start.elapsed() > timeout {
                eprintln!(
                    "Error: Render timeout after {} seconds",
                    RENDER_TIMEOUT_SECS
                );
                eprintln!("Debug info: This may indicate GPU issues, missing assets, or insufficient system resources.");
                std::process::exit(1);
            }

            std::thread::sleep(poll_interval);
        }
    });

    // Configure rendering backend for this environment.
    // Use OnceLock so env vars are only set once per process — repeated calls
    // (e.g. sequential render_to_buffer calls in a parity loop) no longer trigger
    // redundant wgpu backend env writes. Full GPU adapter reuse across App instances
    // requires a persistent renderer (tracked in issue #14).
    static BACKEND_INIT: OnceLock<()> = OnceLock::new();
    BACKEND_INIT.get_or_init(|| {
        let backend_config = BackendConfig::headless();
        backend_config.apply_env();
    });

    // Run Bevy app with HEADLESS configuration
    build_headless_app(request, shared_output, shared_rgba, shared_depth).run();

    // Unreachable - watchdog thread exits the process
    Err(RenderError::RenderFailed(
        "Render did not complete".to_string(),
    ))
}

/// Save RGBA data to PNG file
fn save_rgba_to_png(rgba: &[u8], width: u32, height: u32, path: &Path) -> Result<(), String> {
    use image::{ImageBuffer, Rgba};

    // Create parent directories if needed
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }

    let img: ImageBuffer<Rgba<u8>, Vec<u8>> =
        ImageBuffer::from_raw(width, height, rgba.to_vec())
            .ok_or_else(|| "Failed to create image buffer".to_string())?;

    img.save(path).map_err(|e| e.to_string())
}

/// Save depth data to binary file (f64 for TBP precision)
fn save_depth_to_binary(depth: &[f64], path: &Path) -> Result<(), String> {
    // Create parent directories if needed
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }

    let bytes: Vec<u8> = depth.iter().flat_map(|f| f.to_le_bytes()).collect();
    std::fs::write(path, &bytes).map_err(|e| e.to_string())
}

#[cfg(test)]
mod smoke_tests {
    use super::{
        headless_scene_setup_count, persistent_warmup_camera_transform,
        reset_headless_scene_setup_count,
    };
    use crate::{
        BatchRenderConfig, BatchRenderRequest, ObjectRotation, RenderConfig, TargetingPolicy, Vec3,
        ViewpointConfig,
    };
    use image::{ImageBuffer, Rgba};
    use tempfile::TempDir;

    fn write_synthetic_object() -> TempDir {
        let temp_dir = TempDir::new().expect("create temp dir for synthetic object");
        let object_dir = temp_dir.path().join("synthetic_cube").join("google_16k");
        std::fs::create_dir_all(&object_dir).expect("create synthetic google_16k dir");

        // A small centered cube stays visible from all default TBP viewpoints and does not
        // need any YCB downloads.
        let obj = r#"o SyntheticCube
v -0.10 -0.10  0.10
v  0.10 -0.10  0.10
v  0.10  0.10  0.10
v -0.10  0.10  0.10
v -0.10 -0.10 -0.10
v  0.10 -0.10 -0.10
v  0.10  0.10 -0.10
v -0.10  0.10 -0.10
vt 0.0 0.0
vt 1.0 0.0
vt 1.0 1.0
vt 0.0 1.0
f 1/1 2/2 3/3
f 1/1 3/3 4/4
f 6/1 5/2 8/3
f 6/1 8/3 7/4
f 2/1 6/2 7/3
f 2/1 7/3 3/4
f 5/1 1/2 4/3
f 5/1 4/3 8/4
f 4/1 3/2 7/3
f 4/1 7/3 8/4
f 5/1 6/2 2/3
f 5/1 2/3 1/4
"#;
        std::fs::write(object_dir.join("textured.obj"), obj).expect("write synthetic obj");

        let texture = ImageBuffer::from_fn(2, 2, |x, y| match (x, y) {
            (0, 0) => Rgba([255u8, 48, 48, 255]),
            (1, 0) => Rgba([48u8, 255, 48, 255]),
            (0, 1) => Rgba([48u8, 48, 255, 255]),
            _ => Rgba([255u8, 255, 64, 255]),
        });
        texture
            .save(object_dir.join("texture_map.png"))
            .expect("write synthetic texture");

        temp_dir
    }

    #[test]
    fn persistent_warmup_camera_is_a_real_viewpoint() {
        let transform = persistent_warmup_camera_transform();
        assert!(
            transform.translation.length() > 0.1,
            "persistent warmup must not place the camera at the object origin"
        );

        let forward = transform.rotation * Vec3::NEG_Z;
        let to_origin = -transform.translation.normalize();
        assert!(
            forward.dot(to_origin) > 0.99,
            "persistent warmup camera should look at the object origin"
        );
    }

    #[test]
    #[ignore = "headless throughput smoke check is opt-in because it needs a local render backend"]
    fn test_headless_batch_throughput_smoke() {
        crate::initialize();
        reset_headless_scene_setup_count();

        let object_root = write_synthetic_object();
        let object_dir = object_root.path().join("synthetic_cube");
        let viewpoints = crate::generate_viewpoints(&ViewpointConfig::default());
        let request_count = 5usize;
        let config = RenderConfig::tbp_default();

        let requests: Vec<_> = viewpoints
            .iter()
            .take(request_count)
            .copied()
            .map(|viewpoint| BatchRenderRequest {
                object_dir: object_dir.clone(),
                viewpoint,
                object_rotation: ObjectRotation::identity(),
                object_translation: Vec3::ZERO,
                object_scale: Vec3::ONE,
                render_config: config.clone(),
                target_point: Vec3::ZERO,
                targeting_policy: TargetingPolicy::Origin,
            })
            .collect();

        let start = std::time::Instant::now();
        let outputs = crate::render_batch(requests, &BatchRenderConfig::default())
            .expect("synthetic headless batch render should succeed");
        let elapsed = start.elapsed();

        assert_eq!(outputs.len(), request_count);
        // This is the deterministic churn signal for the smoke check. Adapter log lines vary by
        // backend and logging config, but a homogeneous batch should still set up headless scene
        // state exactly once.
        assert_eq!(
            headless_scene_setup_count(),
            1,
            "homogeneous batch smoke check should reuse one headless app setup"
        );

        for (idx, output) in outputs.iter().enumerate() {
            assert_eq!(output.width, config.width, "output {idx} width mismatch");
            assert_eq!(output.height, config.height, "output {idx} height mismatch");
            assert_eq!(
                output.rgba.len(),
                (config.width * config.height * 4) as usize,
                "output {idx} rgba size mismatch"
            );
            assert_eq!(
                output.depth.len(),
                (config.width * config.height) as usize,
                "output {idx} depth size mismatch"
            );
            assert!(
                output
                    .rgba
                    .chunks_exact(4)
                    .any(|px| px[0] != 0 || px[1] != 0 || px[2] != 0),
                "output {idx} should contain visible color"
            );
        }

        // Acceptance target: under llvmpipe-class CPU rendering, five 64x64 captures should
        // finish in under 8s. Much slower runs usually mean we reintroduced per-capture app
        // churn or another headless startup regression.
        assert!(
            elapsed < std::time::Duration::from_secs(8),
            "5 synthetic headless captures took {:.2}s, expected < 8.0s",
            elapsed.as_secs_f64()
        );
    }
}
