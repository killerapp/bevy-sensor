//! Headless rendering implementation using Bevy.
//!
//! This module provides rendering for capturing RGBA images from YCB objects.
//! A window is briefly opened to render the scene, then `ScreenshotManager`
//! captures the frame before the window closes.
//!
//! # Current Status
//!
//! - **RGBA**: Working via `ScreenshotManager.take_screenshot()` callback
//! - **Depth**: Placeholder only (uniform camera distance per pixel)
//!
//! # Depth Buffer Limitation
//!
//! Real depth buffer extraction is currently blocked because Bevy 0.11's prepass
//! depth texture doesn't have the `COPY_SRC` usage flag, which is required for
//! `copy_texture_to_buffer`. The infrastructure for depth readback exists:
//!
//! - [`DepthReadbackNode`] - A ViewNode (currently disabled) that would copy depth
//! - [`depth_helpers`] - Functions for alignment, extraction, and reverse-Z conversion
//! - [`SharedDepthBuffer`] - Thread-safe buffer for depth data
//! - Graceful fallback to uniform camera distance when depth readback fails
//!
//! ## Solution: Compute Shader Approach
//!
//! To implement real depth extraction, a compute shader approach is needed:
//!
//! 1. Create a compute pipeline that binds prepass depth as a sampled texture
//! 2. Sample each texel and write f32 depth values to a storage buffer
//! 3. Map the storage buffer for CPU readback
//! 4. Convert from reverse-Z NDC to linear depth using [`depth_helpers::reverse_z_to_linear_depth`]
//!
//! Sampling is allowed (unlike copying), so this approach works around the missing
//! `COPY_SRC` flag on prepass textures.
//!
//! # Running Requirements
//!
//! On WSL2 or systems without hardware GPU rendering:
//! ```bash
//! WGPU_BACKEND=vulkan DISPLAY=:0 cargo run --example test_render
//! ```
//!
//! For CI/headless servers, use Xvfb or software rendering (llvmpipe).
//!
//! # Architecture Notes
//!
//! Bevy's `App::run()` does not return cleanly in all configurations. This
//! implementation uses a watchdog thread that monitors for completion and
//! calls `std::process::exit(0)` once the render output is serialized to
//! a temp file. The main thread reads this file after the process would
//! normally exit.

use bevy::asset::LoadState;
use bevy::core_pipeline::prepass::{DepthPrepass, NormalPrepass};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::ecs::query::QueryItem;
use bevy::prelude::*;
use bevy::render::camera::ExtractedCamera;
use bevy::render::render_graph::{
    NodeRunError, RenderGraphApp, RenderGraphContext, ViewNode, ViewNodeRunner,
};
use bevy::render::render_resource::{
    Buffer, BufferDescriptor, BufferUsages, Extent3d, ImageCopyBuffer, ImageCopyTexture,
    ImageDataLayout, MapMode, Origin3d, TextureAspect,
};
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
use bevy::render::view::ViewDepthTexture;
use bevy::render::{Extract, Render, RenderApp, RenderSet};
use bevy::window::{PresentMode, WindowPlugin, WindowResolution};
use bevy_obj::ObjPlugin;
use std::fs::File;
use std::io::{Read as IoRead, Write as IoWrite};
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::{ObjectRotation, RenderConfig, RenderError, RenderOutput};

/// Internal state for tracking render progress
#[derive(Resource, Default)]
struct RenderState {
    frame_count: u32,
    scene_loaded: bool,
    texture_loaded: bool,
    capture_ready: bool,
    screenshot_requested: bool,
    captured: bool,
    exit_requested: bool,
    exit_frame_count: u32,
    rgba_data: Option<Vec<u8>>,
    depth_data: Option<Vec<f32>>,
    image_width: u32,
    image_height: u32,
}

/// Shared buffer for screenshot callback to write into
#[derive(Resource, Clone)]
struct SharedImageBuffer(Arc<Mutex<Option<(Vec<u8>, u32, u32)>>>);

/// Shared buffer for depth data from GPU readback
/// Contains: (linear_depth_values, width, height)
#[derive(Resource, Clone, Default)]
struct SharedDepthBuffer(Arc<Mutex<Option<(Vec<f32>, u32, u32)>>>);

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

/// Pending depth capture info for async processing
struct PendingDepthCapture {
    buffer: Buffer,
    width: u32,
    height: u32,
    near: f32,
    far: f32,
}

/// Queue for pending depth captures (written by render node, read by cleanup system)
#[derive(Resource, Default)]
struct PendingDepthCaptureQueue(Arc<Mutex<Vec<PendingDepthCapture>>>);

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

    /// Convert all NDC depth values to linear meters
    pub fn convert_depth_to_linear(raw_depth: &[f32], near: f32, far: f32) -> Vec<f32> {
        raw_depth
            .iter()
            .map(|&ndc| reverse_z_to_linear_depth(ndc, near, far))
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
        fn test_convert_depth_to_linear_batch() {
            let near = 0.01;
            let far = 10.0;
            let ndc_depths = vec![1.0, 0.5, 0.1, 0.0];

            let linear = convert_depth_to_linear(&ndc_depths, near, far);

            assert_eq!(linear.len(), 4);
            // Near plane
            assert!((linear[0] - near).abs() < 0.001);
            // Far plane
            assert!((linear[3] - far).abs() < 0.001);
            // All should be in range [near, far]
            for d in &linear {
                assert!(*d >= near && *d <= far);
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
    type ViewQuery = (&'static ViewDepthTexture, &'static ExtractedCamera);

    fn run<'w>(
        &self,
        _graph: &mut RenderGraphContext,
        render_context: &mut RenderContext<'w>,
        (view_depth_texture, camera): QueryItem<'w, Self::ViewQuery>,
        world: &'w World,
    ) -> Result<(), NodeRunError> {
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
            ImageCopyTexture {
                texture: &view_depth_texture.texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::DepthOnly,
            },
            ImageCopyBuffer {
                buffer: &staging_buffer,
                layout: ImageDataLayout {
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

        // Push to queue for async processing (queue is Arc<Mutex<Vec>>)
        if let Ok(mut pending) = queue.0.lock() {
            pending.push(PendingDepthCapture {
                buffer: staging_buffer,
                width,
                height,
                near: request.near,
                far: request.far,
            });
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
            error!("Failed to get RenderApp for depth readback");
            return;
        };

        // Insert resources in render world
        render_app.insert_resource(self.shared_depth.clone());
        render_app.init_resource::<PendingDepthCaptureQueue>();

        // Add extraction system to copy request from main world
        render_app.add_systems(ExtractSchedule, extract_depth_request);

        // Add system to process completed depth captures
        render_app.add_systems(Render, collect_depth_captures.in_set(RenderSet::Cleanup));

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

/// Process completed depth buffer captures (async GPU-to-CPU readback)
fn collect_depth_captures(
    queue: Res<PendingDepthCaptureQueue>,
    shared_depth: Res<SharedDepthBuffer>,
) {
    // Take all pending captures from the queue
    let pending_captures = {
        let Ok(mut pending) = queue.0.lock() else {
            return;
        };
        std::mem::take(&mut *pending)
    };

    if pending_captures.is_empty() {
        return;
    }

    // Process each pending capture
    for pending in pending_captures {
        let width = pending.width;
        let height = pending.height;
        let near = pending.near;
        let far = pending.far;
        let buffer = pending.buffer;
        let shared = shared_depth.0.clone();

        // Spawn a thread for blocking GPU-to-CPU buffer mapping
        std::thread::spawn(move || {
            let (tx, rx) = std::sync::mpsc::channel();
            let buffer_slice = buffer.slice(..);

            // Request GPU to map buffer for CPU read
            buffer_slice.map_async(MapMode::Read, move |result| {
                let _ = tx.send(result);
            });

            // Wait for mapping to complete (blocking)
            match rx.recv() {
                Ok(Ok(())) => {
                    // Read mapped data
                    let data = buffer_slice.get_mapped_range();

                    // Extract depth values with alignment handling
                    let ndc_depth =
                        depth_helpers::extract_depth_with_alignment(&data, width, height);

                    drop(data);
                    buffer.unmap();

                    // Convert from reverse-Z NDC to linear depth in meters
                    let linear_depth =
                        depth_helpers::convert_depth_to_linear(&ndc_depth, near, far);

                    // Store in shared buffer
                    if let Ok(mut guard) = shared.lock() {
                        *guard = Some((linear_depth, width, height));
                    }
                }
                Ok(Err(e)) => {
                    eprintln!("Failed to map depth buffer: {:?}", e);
                }
                Err(e) => {
                    eprintln!("Depth buffer mapping channel closed: {:?}", e);
                }
            }
        });
    }
}

/// Configuration passed to the Bevy app
#[derive(Resource, Clone)]
struct RenderRequest {
    mesh_path: String,
    texture_path: String,
    camera_transform: Transform,
    object_rotation: ObjectRotation,
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

/// Perform headless rendering of a YCB object.
///
/// This spins up a minimal Bevy app, renders frames until assets are loaded,
/// then extracts the rendered frame via screenshot.
///
/// Note: Bevy's App::run() may not exit cleanly. A watchdog thread monitors
/// for results and terminates the process once the render is complete.
#[allow(dead_code)]
pub fn render_headless(
    object_dir: &Path,
    camera_transform: &Transform,
    object_rotation: &ObjectRotation,
    config: &RenderConfig,
) -> Result<RenderOutput, RenderError> {
    let mesh_path = object_dir.join("google_16k/textured.obj");
    let texture_path = object_dir.join("google_16k/texture_map.png");

    if !mesh_path.exists() {
        return Err(RenderError::MeshNotFound(mesh_path.display().to_string()));
    }
    if !texture_path.exists() {
        return Err(RenderError::TextureNotFound(
            texture_path.display().to_string(),
        ));
    }

    let request = RenderRequest {
        mesh_path: mesh_path.display().to_string(),
        texture_path: texture_path.display().to_string(),
        camera_transform: *camera_transform,
        object_rotation: object_rotation.clone(),
        config: config.clone(),
    };

    let shared_output: SharedOutput = SharedOutput(Arc::new(Mutex::new(None)));
    let output_clone = shared_output.clone();
    let output_poll = shared_output.clone();

    // Shared buffer for screenshot callback
    let shared_image: SharedImageBuffer = SharedImageBuffer(Arc::new(Mutex::new(None)));
    let image_clone = shared_image.clone();

    // Shared buffer for depth readback
    let shared_depth: SharedDepthBuffer = SharedDepthBuffer::default();
    let depth_clone = shared_depth.clone();

    // Create a temp file path for output serialization
    let temp_path =
        std::env::temp_dir().join(format!("bevy_sensor_render_{}.bin", std::process::id()));
    let temp_path_clone = temp_path.clone();

    // Spawn watchdog thread that monitors for results and exits process when ready
    std::thread::spawn(move || {
        let timeout = std::time::Duration::from_secs(60);
        let start = std::time::Instant::now();
        let poll_interval = std::time::Duration::from_millis(100);

        loop {
            // Check if we have a result
            if let Ok(guard) = output_poll.0.lock() {
                if let Some(output) = guard.as_ref() {
                    // Serialize output to temp file
                    let data = serialize_output(output);
                    if let Ok(mut file) = File::create(&temp_path_clone) {
                        let _ = file.write_all(&data);
                    }
                    // Give a moment for file to flush
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    // Exit the process - App::run() won't return otherwise
                    std::process::exit(0);
                }
            }

            if start.elapsed() > timeout {
                std::process::exit(1);
            }

            std::thread::sleep(poll_interval);
        }
    });

    // Run Bevy app on main thread (required by winit)
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        resolution: WindowResolution::new(
                            config.width as f32,
                            config.height as f32,
                        ),
                        present_mode: PresentMode::AutoNoVsync,
                        title: "bevy-sensor render".into(),
                        ..default()
                    }),
                    ..default()
                })
                .disable::<bevy::log::LogPlugin>(),
        )
        .add_plugins(ObjPlugin)
        .add_plugins(DepthReadbackPlugin {
            shared_depth: depth_clone,
            near: config.near_plane,
            far: config.far_plane,
        })
        .insert_resource(request)
        .insert_resource(output_clone)
        .insert_resource(image_clone)
        .init_resource::<RenderState>()
        .add_systems(Startup, setup_scene)
        .add_systems(
            Update,
            (
                check_assets_loaded,
                apply_materials,
                request_screenshot,
                check_screenshot_ready,
                extract_and_exit,
            )
                .chain(),
        )
        .run();

    // If we get here, try to read from temp file (unlikely since watchdog exits)
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

/// Serialize RenderOutput to bytes for IPC
fn serialize_output(output: &RenderOutput) -> Vec<u8> {
    let mut data = Vec::new();

    // Header: width, height, rgba_len, depth_len
    data.extend_from_slice(&output.width.to_le_bytes());
    data.extend_from_slice(&output.height.to_le_bytes());
    data.extend_from_slice(&(output.rgba.len() as u32).to_le_bytes());
    data.extend_from_slice(&(output.depth.len() as u32).to_le_bytes());

    // RGBA data
    data.extend_from_slice(&output.rgba);

    // Depth data (as f32 bytes)
    for d in &output.depth {
        data.extend_from_slice(&d.to_le_bytes());
    }

    // Intrinsics
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

    // Object rotation
    let or = &output.object_rotation;
    data.extend_from_slice(&or.pitch.to_le_bytes());
    data.extend_from_slice(&or.yaw.to_le_bytes());
    data.extend_from_slice(&or.roll.to_le_bytes());

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

    let width = read_u32(&data, &mut cursor);
    let height = read_u32(&data, &mut cursor);
    let rgba_len = read_u32(&data, &mut cursor) as usize;
    let depth_len = read_u32(&data, &mut cursor) as usize;

    let rgba = data[cursor..cursor + rgba_len].to_vec();
    cursor += rgba_len;

    let mut depth = Vec::with_capacity(depth_len);
    for _ in 0..depth_len {
        depth.push(read_f32(&data, &mut cursor));
    }

    let focal_length = [read_f32(&data, &mut cursor), read_f32(&data, &mut cursor)];
    let principal_point = [read_f32(&data, &mut cursor), read_f32(&data, &mut cursor)];
    let image_size = [read_u32(&data, &mut cursor), read_u32(&data, &mut cursor)];

    let tx = read_f32(&data, &mut cursor);
    let ty = read_f32(&data, &mut cursor);
    let tz = read_f32(&data, &mut cursor);
    let rx = read_f32(&data, &mut cursor);
    let ry = read_f32(&data, &mut cursor);
    let rz = read_f32(&data, &mut cursor);
    let rw = read_f32(&data, &mut cursor);

    let pitch = read_f32(&data, &mut cursor);
    let yaw = read_f32(&data, &mut cursor);
    let roll = read_f32(&data, &mut cursor);

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
    })
}

/// Setup the scene with camera, lighting, and object
fn setup_scene(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    request: Res<RenderRequest>,
    mut _materials: ResMut<Assets<StandardMaterial>>,
) {
    // Camera with depth prepass (Bevy 0.15+ uses Camera3d component)
    // Disable MSAA for depth readback compatibility (can't copy from multisampled texture)
    commands.spawn((
        Camera3d::default(),
        Camera {
            hdr: true,
            ..default()
        },
        Msaa::Off,
        request.camera_transform,
        Tonemapping::None, // Accurate colors for software rendering
        DepthPrepass,
        NormalPrepass,
        RenderCamera,
    ));

    // Ambient light (from config)
    let lighting = &request.config.lighting;
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: lighting.ambient_brightness,
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

    // Spawn the scene with rotation (Bevy 0.15+ uses SceneRoot)
    commands.spawn((
        SceneRoot(scene_handle),
        Transform::from_rotation(request.object_rotation.to_quat()),
        RenderedObject,
    ));

    info!("Scene setup complete");
}

/// Check if assets are loaded
fn check_assets_loaded(
    mut state: ResMut<RenderState>,
    asset_server: Res<AssetServer>,
    scene: Option<Res<LoadedScene>>,
    texture: Option<Res<LoadedTexture>>,
) {
    if state.scene_loaded && state.texture_loaded {
        return;
    }

    if let Some(scene) = scene {
        match asset_server.get_load_state(&scene.0) {
            Some(LoadState::Loaded) => {
                state.scene_loaded = true;
                info!("Scene loaded");
            }
            Some(LoadState::Failed(err)) => {
                error!("Scene failed to load: {:?}", err);
            }
            _ => {}
        }
    }

    if let Some(texture) = texture {
        match asset_server.get_load_state(&texture.0) {
            Some(LoadState::Loaded) => {
                state.texture_loaded = true;
                info!("Texture loaded");
            }
            Some(LoadState::Failed(err)) => {
                error!("Texture failed to load: {:?}", err);
            }
            _ => {}
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
    if !state.scene_loaded || !state.texture_loaded || state.capture_ready {
        return;
    }

    state.frame_count += 1;

    // Wait a few frames for everything to settle
    if state.frame_count < 10 {
        return;
    }

    let Some(tex) = texture else { return };

    // Create textured material
    let textured_material = materials.add(StandardMaterial {
        base_color_texture: Some(tex.0.clone()),
        unlit: true,
        ..default()
    });

    // Apply to all meshes
    let mut count = 0;
    for mut mat in mesh_query.iter_mut() {
        mat.0 = textured_material.clone();
        count += 1;
    }

    if count > 0 {
        info!("Applied texture to {} meshes", count);
    }

    // Wait more frames after applying materials
    if state.frame_count >= 30 {
        state.capture_ready = true;
        info!("Ready to capture");
    }
}

/// Request a screenshot capture (Bevy 0.15+ uses Screenshot entity + observer)
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
    info!("Depth capture requested");

    // Spawn Screenshot entity with observer (Bevy 0.15+ API)
    info!("Requesting screenshot via Screenshot entity");
    commands.spawn(Screenshot::primary_window()).observe(
        move |trigger: Trigger<ScreenshotCaptured>| {
            // ScreenshotCaptured derefs to Image
            let image: &Image = trigger.event();

            // Get dimensions
            let size = image.size();
            let width = size.x;
            let height = size.y;

            // Get raw image data - Bevy 0.15 Image uses .data field
            let rgba_data = image.data.clone();

            // Store in shared buffer
            if let Ok(mut guard) = image_buffer.lock() {
                *guard = Some((rgba_data, width, height));
            }
        },
    );

    state.screenshot_requested = true;
    info!("Screenshot requested");
}

/// Check if screenshot callback has completed
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

    // If depth readback failed or is taking too long, fall back to placeholder
    // (This allows graceful degradation on systems where depth readback fails)
    if rgba_ready && !depth_ready && state.frame_count > 60 {
        let camera_dist = request.camera_transform.translation.length();
        let pixel_count = (state.image_width * state.image_height) as usize;
        state.depth_data = Some(vec![camera_dist; pixel_count]);
    }

    // Mark as captured when both RGBA and depth are ready
    if state.rgba_data.is_some() && state.depth_data.is_some() {
        state.captured = true;
    }
}

/// Extract results and exit
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

        // Compute intrinsics based on actual dimensions
        let config = &request.config;
        let intrinsics = crate::CameraIntrinsics {
            focal_length: [width as f32 * config.zoom, height as f32 * config.zoom],
            principal_point: [width as f32 / 2.0, height as f32 / 2.0],
            image_size: [width, height],
        };

        let output = RenderOutput {
            rgba: rgba.clone(),
            depth: depth.clone(),
            width,
            height,
            intrinsics,
            camera_transform: request.camera_transform,
            object_rotation: request.object_rotation.clone(),
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
