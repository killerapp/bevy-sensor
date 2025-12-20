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

use bevy::app::ScheduleRunnerPlugin;
use bevy::asset::LoadState;
use bevy::core_pipeline::prepass::{DepthPrepass, NormalPrepass};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::ecs::query::QueryItem;
use bevy::prelude::*;
use bevy::render::camera::{ExtractedCamera, RenderTarget};
use bevy::render::render_asset::{RenderAssetUsages, RenderAssets};
use bevy::render::render_graph::{
    Node, NodeRunError, RenderGraphApp, RenderGraphContext, RenderLabel, ViewNode, ViewNodeRunner,
};
use bevy::render::render_resource::{
    Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Extent3d, ImageCopyBuffer,
    ImageCopyTexture, ImageDataLayout, MapMode, Origin3d, TextureAspect, TextureDimension,
    TextureFormat, TextureUsages,
};
use bevy::render::renderer::RenderQueue;
use bevy::render::renderer::{RenderContext, RenderDevice};
use bevy::render::texture::GpuImage;
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
use bevy::render::view::ViewDepthTexture;
use bevy::render::{Extract, Render, RenderApp, RenderSet};
use bevy::window::{ExitCondition, WindowPlugin};
use bevy_obj::ObjPlugin;
use std::fs::File;
use std::io::Read as IoRead;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::{backend::BackendConfig, ObjectRotation, RenderConfig, RenderError, RenderOutput};

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
    capture_ready: bool,
    screenshot_requested: bool,
    captured: bool,
    exit_requested: bool,
    #[allow(dead_code)]
    exit_frame_count: u32,
    rgba_data: Option<Vec<u8>>,
    depth_data: Option<Vec<f64>>,
    image_width: u32,
    image_height: u32,
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

    /// Convert all NDC depth values to linear meters (as f64 for TBP precision)
    pub fn convert_depth_to_linear(raw_depth: &[f32], near: f32, far: f32) -> Vec<f64> {
        raw_depth
            .iter()
            .map(|&ndc| reverse_z_to_linear_depth(ndc, near, far) as f64)
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
            eprintln!("Failed to get RenderApp for depth readback");
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

/// Process completed depth buffer captures (synchronous GPU-to-CPU readback with device polling)
fn collect_depth_captures(
    queue: Res<PendingDepthCaptureQueue>,
    shared_depth: Res<SharedDepthBuffer>,
    render_device: Res<RenderDevice>,
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

    // Process each pending capture synchronously with device polling
    for pending in pending_captures {
        let width = pending.width;
        let height = pending.height;
        let near = pending.near;
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

        // Poll the device until mapping completes
        loop {
            render_device.poll(bevy::render::render_resource::Maintain::Poll);
            match rx.try_recv() {
                Ok(Ok(())) => {
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

            let width = gpu_image.size.x;
            let height = gpu_image.size.y;

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
                ImageCopyBuffer {
                    buffer: &staging_buffer,
                    layout: ImageDataLayout {
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
    let pending_captures = {
        let Ok(mut pending) = queue.0.lock() else {
            return;
        };
        std::mem::take(&mut *pending)
    };

    if pending_captures.is_empty() {
        return;
    }

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
        loop {
            render_device.poll(bevy::render::render_resource::Maintain::Poll);

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
        render_app.add_systems(Render, collect_image_captures.in_set(RenderSet::Cleanup));

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

    // Shared buffer for RGBA data from headless render target
    let shared_rgba: SharedRgbaBuffer = SharedRgbaBuffer::default();
    let rgba_clone = shared_rgba.clone();

    // Shared buffer for depth readback
    let shared_depth: SharedDepthBuffer = SharedDepthBuffer::default();
    let depth_clone = shared_depth.clone();

    // Create a temp file path for fallback output serialization
    let temp_path =
        std::env::temp_dir().join(format!("bevy_sensor_render_{}.bin", std::process::id()));

    // Spawn watchdog thread that monitors for timeout (don't exit - let Bevy exit gracefully)
    let output_poll_for_timeout = shared_output.clone();
    std::thread::spawn(move || {
        let timeout = std::time::Duration::from_secs(60);
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
                eprintln!("Error: Render timeout after 60 seconds");
                eprintln!("Debug info: This may indicate GPU issues, missing assets, or insufficient system resources.");
                // Force exit on timeout (this is a failure case)
                std::process::exit(1);
            }

            std::thread::sleep(poll_interval);
        }
    });

    // Run Bevy app with HEADLESS configuration (no window surfaces!)
    // Uses ScheduleRunnerPlugin instead of WinitPlugin
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: None, // NO WINDOW - true headless
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .disable::<bevy::winit::WinitPlugin>(), // Disable winit entirely
        )
        .add_plugins(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
            1.0 / 60.0,
        )))
        .add_plugins(ObjPlugin)
        .add_plugins(ImageCopyPlugin {
            shared_rgba: rgba_clone,
        })
        .add_plugins(DepthReadbackPlugin {
            shared_depth: depth_clone,
            near: config.near_plane,
            far: config.far_plane,
        })
        .insert_resource(request)
        .insert_resource(output_clone)
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
        )
        .run();

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
#[allow(dead_code)]
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

    println!("Scene setup complete");
}

/// Check if assets are loaded
fn check_assets_loaded(
    mut state: ResMut<RenderState>,
    asset_server: Res<AssetServer>,
    scene: Option<Res<LoadedScene>>,
    texture: Option<Res<LoadedTexture>>,
) {
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
        println!("Applied texture to {} meshes", count);
    }

    // Wait more frames after applying materials
    // Software rendering (llvmpipe) needs more frames to fully render
    if state.frame_count >= 60 {
        state.capture_ready = true;
        println!("Ready to capture (frame {})", state.frame_count);
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
    commands.spawn(Screenshot::primary_window()).observe(
        move |trigger: Trigger<ScreenshotCaptured>| {
            // ScreenshotCaptured derefs to Image
            let image: &Image = trigger.event();

            // Get dimensions
            let width = image.texture_descriptor.size.width;
            let height = image.texture_descriptor.size.height;

            // Get raw image data - Bevy 0.15 Image.data is Vec<u8>
            let rgba_data = image.data.clone();

            // Store in shared buffer
            if let Ok(mut guard) = image_buffer.lock() {
                *guard = Some((rgba_data, width, height));
            }
        },
    );

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

    // If depth readback failed or is taking too long, fall back to placeholder
    // (This allows graceful degradation on systems where depth readback fails)
    if rgba_ready && !depth_ready && state.frame_count > 60 {
        let camera_dist = request.camera_transform.translation.length() as f64;
        let pixel_count = (state.image_width * state.image_height) as usize;
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

        // Compute intrinsics based on actual dimensions (f64 for TBP precision)
        let config = &request.config;
        let intrinsics = crate::CameraIntrinsics {
            focal_length: [
                width as f64 * config.zoom as f64,
                height as f64 * config.zoom as f64,
            ],
            principal_point: [width as f64 / 2.0, height as f64 / 2.0],
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
        Camera {
            hdr: true,
            target: RenderTarget::Image(render_target_handle.clone()),
            ..default()
        },
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

    // Ambient light
    let lighting = &request.config.lighting;
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: lighting.ambient_brightness,
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

    // Spawn the scene with rotation
    commands.spawn((
        SceneRoot(scene_handle),
        Transform::from_rotation(request.object_rotation.to_quat()),
        RenderedObject,
    ));
}

/// Request capture for headless rendering (enable ImageCopier)
fn request_headless_capture(
    mut state: ResMut<RenderState>,
    mut depth_request: ResMut<DepthCaptureRequest>,
    mut query: Query<&mut ImageCopier>,
) {
    if !state.capture_ready || state.screenshot_requested {
        return;
    }

    println!("Requesting capture at frame {}", state.frame_count);

    // Enable the ImageCopier to trigger RGBA extraction
    for mut copier in query.iter_mut() {
        copier.enabled = true;
    }

    // Request depth capture
    depth_request.requested = true;

    state.screenshot_requested = true;
}

/// Check if headless capture has completed
fn check_headless_capture_ready(
    mut state: ResMut<RenderState>,
    shared_rgba: Res<SharedRgbaBuffer>,
    shared_depth: Res<SharedDepthBuffer>,
    request: Res<RenderRequest>,
    mut query: Query<&mut ImageCopier>,
) {
    if !state.screenshot_requested || state.captured {
        return;
    }

    state.frame_count += 1;

    // Check if RGBA data is ready
    let rgba_ready = if let Ok(guard) = shared_rgba.0.lock() {
        if let Some((rgba_data, width, height)) = guard.as_ref() {
            if state.rgba_data.is_none() {
                state.rgba_data = Some(rgba_data.clone());
                state.image_width = *width;
                state.image_height = *height;
                // Disable further captures
                for mut copier in query.iter_mut() {
                    copier.enabled = false;
                }
            }
            true
        } else {
            false
        }
    } else {
        false
    };

    // Check if depth data is ready
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

    // Fallback to placeholder depth after 10 extra frames if depth readback fails
    if rgba_ready && !depth_ready && state.frame_count > 70 {
        let camera_dist = request.camera_transform.translation.length() as f64;
        let pixel_count = (state.image_width * state.image_height) as usize;
        state.depth_data = Some(vec![camera_dist; pixel_count]);
    }

    if state.rgba_data.is_some() && state.depth_data.is_some() {
        state.captured = true;
    }
}

/// Extract results and exit for headless rendering
fn extract_and_exit_headless(
    mut state: ResMut<RenderState>,
    request: Res<RenderRequest>,
    shared_output: Res<SharedOutput>,
    mut app_exit: EventWriter<bevy::app::AppExit>,
) {
    if state.exit_requested {
        return;
    }

    if !state.captured {
        return;
    }

    if let (Some(rgba), Some(depth)) = (&state.rgba_data, &state.depth_data) {
        let width = state.image_width;
        let height = state.image_height;

        // Compute intrinsics (f64 for TBP precision)
        let config = &request.config;
        let intrinsics = crate::CameraIntrinsics {
            focal_length: [
                width as f64 * config.zoom as f64,
                height as f64 * config.zoom as f64,
            ],
            principal_point: [width as f64 / 2.0, height as f64 / 2.0],
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
            drop(guard);
            std::thread::sleep(std::time::Duration::from_millis(200));
        }

        // Send AppExit event (headless apps use this instead of closing windows)
        app_exit.send(bevy::app::AppExit::Success);
        state.exit_requested = true;
    }
}

/// Render directly to files (for subprocess mode).
///
/// This function saves RGBA and depth data directly to files before exiting.
/// Designed for subprocess rendering where the process will exit after rendering.
pub fn render_to_files(
    object_dir: &Path,
    camera_transform: &Transform,
    object_rotation: &ObjectRotation,
    config: &RenderConfig,
    rgba_path: &Path,
    depth_path: &Path,
) -> Result<(), RenderError> {
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

    // Shared state for output
    let shared_output: SharedOutput = SharedOutput(Arc::new(Mutex::new(None)));
    let output_poll = shared_output.clone();

    // Clone paths for watchdog thread
    let rgba_path = rgba_path.to_path_buf();
    let depth_path = depth_path.to_path_buf();

    // Shared buffer for RGBA data from headless render target
    let shared_rgba: SharedRgbaBuffer = SharedRgbaBuffer::default();
    let rgba_clone = shared_rgba.clone();

    // Shared buffer for depth readback
    let shared_depth: SharedDepthBuffer = SharedDepthBuffer::default();
    let depth_clone = shared_depth.clone();

    // Spawn watchdog thread that saves files and exits
    std::thread::spawn(move || {
        let timeout = std::time::Duration::from_secs(60);
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
                eprintln!("Error: Render timeout after 60 seconds");
                eprintln!("Debug info: This may indicate GPU issues, missing assets, or insufficient system resources.");
                std::process::exit(1);
            }

            std::thread::sleep(poll_interval);
        }
    });

    // Configure rendering backend for this environment
    let backend_config = BackendConfig::headless();
    backend_config.apply_env();

    // Run Bevy app with HEADLESS configuration
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: None,
                    exit_condition: ExitCondition::DontExit,
                    ..default()
                })
                .disable::<bevy::winit::WinitPlugin>(),
        )
        .add_plugins(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
            1.0 / 60.0,
        )))
        .add_plugins(ObjPlugin)
        .add_plugins(ImageCopyPlugin {
            shared_rgba: rgba_clone,
        })
        .add_plugins(DepthReadbackPlugin {
            shared_depth: depth_clone,
            near: config.near_plane,
            far: config.far_plane,
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
        )
        .run();

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
