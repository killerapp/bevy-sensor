//! Batch rendering API for multiple viewpoints and objects.
//!
//! Today this module is a queue-oriented wrapper around sequential `render_to_buffer()`
//! calls. It does not yet keep a persistent Bevy app alive across renders; that follow-up
//! remains tracked work. The API is still useful for consumers that want ordered request
//! management and structured batch outputs without promising reuse semantics that do not
//! exist yet.
//!
//! # Example
//!
//! ```ignore
//! use bevy_sensor::{
//!     create_batch_renderer, queue_render_request, render_next_in_batch,
//!     batch::BatchRenderRequest, BatchRenderConfig, RenderConfig, ObjectRotation,
//!     TargetingPolicy, Vec3,
//! };
//! use std::path::PathBuf;
//!
//! // Create a batch helper
//! let config = BatchRenderConfig::default();
//! let mut renderer = create_batch_renderer(&config)?;
//!
//! // Queue multiple renders
//! for rotation in rotations {
//!     for viewpoint in viewpoints {
//!         queue_render_request(&mut renderer, BatchRenderRequest {
//!             object_dir: "/tmp/ycb/003_cracker_box".into(),
//!             viewpoint,
//!             object_rotation: rotation.clone(),
//!             object_translation: Vec3::ZERO,
//!             object_scale: Vec3::ONE,
//!             render_config: RenderConfig::tbp_default(),
//!             target_point: Vec3::ZERO,
//!             targeting_policy: TargetingPolicy::Origin,
//!         })?;
//!     }
//! }
//!
//! // Execute and collect results
//! let mut results = Vec::new();
//! loop {
//!     match render_next_in_batch(&mut renderer, 500)? {
//!         Some(output) => results.push(output),
//!         None => break,
//!     }
//! }
//! ```

use crate::{
    semantic_3d_from_depth, CameraIntrinsics, ObjectRotation, RenderConfig, RenderHealth,
    RenderOutput, TargetingPolicy,
};
use bevy::prelude::{Transform, Vec3};
use std::collections::VecDeque;
use std::path::PathBuf;

/// Configuration for batch rendering.
#[derive(Clone, Debug)]
pub struct BatchRenderConfig {
    /// Maximum number of renders to queue before automatic cleanup
    pub max_batch_size: usize,
    /// Timeout in milliseconds per individual render
    pub frame_timeout_ms: u32,
    /// Enable depth buffer readback
    pub enable_depth_readback: bool,
    /// Enable asset caching for repeated objects
    pub enable_asset_caching: bool,
    /// Number of renders before triggering resource cleanup
    pub resource_cleanup_interval: u32,
}

impl Default for BatchRenderConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 256,
            frame_timeout_ms: 500,
            enable_depth_readback: true,
            enable_asset_caching: true,
            resource_cleanup_interval: 32,
        }
    }
}

/// A single render request in a batch.
#[derive(Clone, Debug)]
pub struct BatchRenderRequest {
    /// Path to YCB object directory (e.g., "/tmp/ycb/003_cracker_box")
    pub object_dir: PathBuf,
    /// Camera transform (position and orientation)
    pub viewpoint: Transform,
    /// Object rotation to apply
    pub object_rotation: ObjectRotation,
    /// Object world translation to apply
    pub object_translation: Vec3,
    /// Object scale to apply
    pub object_scale: Vec3,
    /// Render configuration (resolution, lighting, etc.)
    pub render_config: RenderConfig,
    /// Point the camera was intended to target for this render.
    pub target_point: Vec3,
    /// Policy used to derive `target_point`.
    pub targeting_policy: TargetingPolicy,
}

impl BatchRenderRequest {
    /// Build a request with the current default object transform: origin translation and unit scale.
    pub fn new(
        object_dir: PathBuf,
        viewpoint: Transform,
        object_rotation: ObjectRotation,
        render_config: RenderConfig,
    ) -> Self {
        Self {
            object_dir,
            viewpoint,
            object_rotation,
            object_translation: Vec3::ZERO,
            object_scale: Vec3::ONE,
            render_config,
            target_point: Vec3::ZERO,
            targeting_policy: TargetingPolicy::Origin,
        }
    }

    /// Attach explicit object translation and scale.
    pub fn with_object_transform(mut self, object_translation: Vec3, object_scale: Vec3) -> Self {
        self.object_translation = object_translation;
        self.object_scale = object_scale;
        self
    }

    /// Attach camera-target metadata used to create the request viewpoint.
    pub fn with_targeting(mut self, target_point: Vec3, targeting_policy: TargetingPolicy) -> Self {
        self.target_point = target_point;
        self.targeting_policy = targeting_policy;
        self
    }
}

/// Status of a single render in a batch.
#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum RenderStatus {
    /// Render completed successfully with RGBA and depth
    Success,
    /// Render completed but depth extraction failed
    PartialFailure,
    /// Render failed completely
    Failed,
}

/// Output from a single render in a batch.
#[derive(Clone, Debug)]
pub struct BatchRenderOutput {
    /// Original request for this render
    pub request: BatchRenderRequest,
    /// RGBA pixel data (width * height * 4 bytes, row-major)
    pub rgba: Vec<u8>,
    /// Depth data in meters (width * height f64s)
    pub depth: Vec<f64>,
    /// Image width in pixels
    pub width: u32,
    /// Image height in pixels
    pub height: u32,
    /// Camera intrinsics used
    pub intrinsics: CameraIntrinsics,
    /// Camera transform used for world-space depth unprojection.
    pub camera_transform: Transform,
    /// Object world translation applied during render.
    pub object_translation: Vec3,
    /// Object scale applied during render.
    pub object_scale: Vec3,
    /// Point the camera was intended to target for this render.
    pub target_point: Vec3,
    /// Policy used to derive `target_point`.
    pub targeting_policy: TargetingPolicy,
    /// Cheap diagnostics derived from the rendered depth buffer
    pub health: RenderHealth,
    /// Status of this render
    pub status: RenderStatus,
    /// Error message if status is Failed or PartialFailure
    pub error_message: Option<String>,
}

impl BatchRenderOutput {
    /// Convert to neocortx-compatible RGB format: Vec<Vec<[u8; 3]>>
    pub fn to_rgb_image(&self) -> Vec<Vec<[u8; 3]>> {
        let mut image = Vec::with_capacity(self.height as usize);
        for y in 0..self.height {
            let mut row = Vec::with_capacity(self.width as usize);
            for x in 0..self.width {
                let idx = ((y * self.width + x) * 4) as usize;
                if idx + 2 < self.rgba.len() {
                    row.push([self.rgba[idx], self.rgba[idx + 1], self.rgba[idx + 2]]);
                } else {
                    row.push([0, 0, 0]);
                }
            }
            image.push(row);
        }
        image
    }

    /// Convert depth to neocortx-compatible format: Vec<Vec<f64>>
    pub fn to_depth_image(&self) -> Vec<Vec<f64>> {
        let mut image = Vec::with_capacity(self.height as usize);
        for y in 0..self.height {
            let mut row = Vec::with_capacity(self.width as usize);
            for x in 0..self.width {
                let idx = (y * self.width + x) as usize;
                if idx < self.depth.len() {
                    row.push(self.depth[idx]);
                } else {
                    row.push(0.0);
                }
            }
            image.push(row);
        }
        image
    }

    /// Build TBP-style `semantic_3d` rows using this request's far plane.
    ///
    /// The returned vector is row-major with one `[x, y, z, semantic_id]` row
    /// per pixel. Foreground pixels are unprojected into world space and use
    /// `object_semantic_id`; background/far pixels are `[0, 0, 0, 0]`.
    pub fn semantic_3d(&self, object_semantic_id: u32) -> Vec<[f64; 4]> {
        self.semantic_3d_with_far_plane(
            object_semantic_id,
            self.request.render_config.far_plane as f64,
        )
    }

    /// Build TBP-style `semantic_3d` rows using a caller-provided far plane.
    pub fn semantic_3d_with_far_plane(
        &self,
        object_semantic_id: u32,
        far_plane: f64,
    ) -> Vec<[f64; 4]> {
        semantic_3d_from_depth(
            &self.depth,
            self.width,
            self.height,
            &self.intrinsics,
            self.camera_transform,
            object_semantic_id,
            far_plane,
        )
    }

    /// Convert from RenderOutput, carrying request-level target metadata.
    pub fn from_render_output(request: BatchRenderRequest, output: RenderOutput) -> Self {
        let health = output.health_with_far_plane(request.render_config.far_plane as f64);
        let camera_transform = output.camera_transform;
        let object_translation = output.object_translation;
        let object_scale = output.object_scale;
        let target_point = request.target_point;
        let targeting_policy = request.targeting_policy.clone();
        Self {
            request,
            rgba: output.rgba,
            depth: output.depth,
            width: output.width,
            height: output.height,
            intrinsics: output.intrinsics,
            camera_transform,
            object_translation,
            object_scale,
            target_point,
            targeting_policy,
            health,
            status: RenderStatus::Success,
            error_message: None,
        }
    }
}

/// Error types for batch rendering.
#[derive(Debug, Clone)]
pub enum BatchRenderError {
    /// Some renders succeeded, others failed
    PartialFailure { successful: usize, failed: usize },
    /// All renders failed
    TotalFailure(String),
    /// Invalid configuration
    InvalidConfig(String),
    /// Queue is full
    QueueFull,
    /// No renders queued
    EmptyQueue,
    /// The wgpu device was lost mid-render. The current `RenderSession::render()`
    /// call produced no output; any outputs returned by earlier calls remain valid.
    /// Recovery: drop the session and construct a new one.
    ///
    /// `reason` is a string form of `wgpu::DeviceLostReason` so callers can branch
    /// on recoverable vs. adapter-evicted without taking a direct wgpu dependency.
    /// Phase 1 ships the string form; a typed variant may follow once the Bevy
    /// re-export surface is clearer.
    DeviceLost { reason: String, message: String },
}

impl std::fmt::Display for BatchRenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BatchRenderError::PartialFailure { successful, failed } => {
                write!(
                    f,
                    "Batch render partial failure: {} succeeded, {} failed",
                    successful, failed
                )
            }
            BatchRenderError::TotalFailure(msg) => write!(f, "Batch render total failure: {}", msg),
            BatchRenderError::InvalidConfig(msg) => write!(f, "Invalid batch config: {}", msg),
            BatchRenderError::QueueFull => write!(f, "Batch queue is full"),
            BatchRenderError::EmptyQueue => write!(f, "No renders queued"),
            BatchRenderError::DeviceLost { reason, message } => {
                write!(f, "wgpu device lost ({}): {}", reason, message)
            }
        }
    }
}

impl std::error::Error for BatchRenderError {}

/// State machine for batch rendering lifecycle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BatchState {
    /// Idle, waiting for requests to queue
    Idle,
    /// Loading object assets (mesh, texture)
    LoadingAssets,
    /// Rendering frame to GPU buffer
    RenderingFrame,
    /// Extracting RGBA and depth from GPU
    ExtractingResults,
    /// Cleaning up resources
    Cleanup,
    /// Shutting down
    Shutdown,
}

/// Manages queued render requests and completed outputs for batch-style workflows.
pub struct BatchRenderer {
    /// Queued render requests
    pub pending_requests: VecDeque<BatchRenderRequest>,
    /// Completed results
    pub completed_results: Vec<BatchRenderOutput>,
    /// Current request being processed
    pub current_request: Option<BatchRenderRequest>,
    /// Current render output being built
    pub current_output: Option<BatchRenderOutput>,
    /// Frame counter for timeout management
    pub frame_count: u32,
    /// Current state
    pub state: BatchState,
    /// Configuration
    pub config: BatchRenderConfig,
    /// Total renders processed
    pub renders_processed: usize,
}

impl BatchRenderer {
    /// Create a new batch renderer with default configuration.
    pub fn new(config: BatchRenderConfig) -> Self {
        Self {
            pending_requests: VecDeque::new(),
            completed_results: Vec::new(),
            current_request: None,
            current_output: None,
            frame_count: 0,
            state: BatchState::Idle,
            config,
            renders_processed: 0,
        }
    }

    /// Queue a render request for batch processing.
    pub fn queue_request(&mut self, request: BatchRenderRequest) -> Result<(), BatchRenderError> {
        if self.pending_requests.len() >= self.config.max_batch_size {
            return Err(BatchRenderError::QueueFull);
        }
        self.pending_requests.push_back(request);
        Ok(())
    }

    /// Get the number of pending requests.
    pub fn pending_count(&self) -> usize {
        self.pending_requests.len()
    }

    /// Get the number of completed results.
    pub fn completed_count(&self) -> usize {
        self.completed_results.len()
    }

    /// Get all completed results and clear the internal list.
    pub fn take_completed(&mut self) -> Vec<BatchRenderOutput> {
        std::mem::take(&mut self.completed_results)
    }

    /// Check if all work is done (no pending requests and not currently rendering).
    pub fn is_finished(&self) -> bool {
        self.pending_requests.is_empty() && self.current_request.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_config_defaults() {
        let config = BatchRenderConfig::default();
        assert_eq!(config.max_batch_size, 256);
        assert_eq!(config.frame_timeout_ms, 500);
        assert!(config.enable_depth_readback);
        assert!(config.enable_asset_caching);
    }

    #[test]
    fn test_batch_renderer_creation() {
        let config = BatchRenderConfig::default();
        let renderer = BatchRenderer::new(config);
        assert_eq!(renderer.state, BatchState::Idle);
        assert_eq!(renderer.pending_count(), 0);
        assert_eq!(renderer.completed_count(), 0);
        assert!(renderer.is_finished());
    }

    #[test]
    fn test_queue_request() {
        let mut renderer = BatchRenderer::new(BatchRenderConfig::default());
        let request = BatchRenderRequest {
            object_dir: "/tmp/test".into(),
            viewpoint: Transform::default(),
            object_rotation: ObjectRotation::identity(),
            object_translation: Vec3::ZERO,
            object_scale: Vec3::ONE,
            render_config: RenderConfig::tbp_default(),
            target_point: Vec3::ZERO,
            targeting_policy: TargetingPolicy::Origin,
        };
        assert!(renderer.queue_request(request).is_ok());
        assert_eq!(renderer.pending_count(), 1);
    }

    #[test]
    fn test_queue_full() {
        let config = BatchRenderConfig {
            max_batch_size: 1,
            ..BatchRenderConfig::default()
        };
        let mut renderer = BatchRenderer::new(config);

        let request = BatchRenderRequest {
            object_dir: "/tmp/test".into(),
            viewpoint: Transform::default(),
            object_rotation: ObjectRotation::identity(),
            object_translation: Vec3::ZERO,
            object_scale: Vec3::ONE,
            render_config: RenderConfig::tbp_default(),
            target_point: Vec3::ZERO,
            targeting_policy: TargetingPolicy::Origin,
        };

        assert!(renderer.queue_request(request.clone()).is_ok());
        assert!(matches!(
            renderer.queue_request(request),
            Err(BatchRenderError::QueueFull)
        ));
    }

    #[test]
    fn test_batch_render_output_rgb_conversion() {
        let request = BatchRenderRequest {
            object_dir: "/tmp/test".into(),
            viewpoint: Transform::default(),
            object_rotation: ObjectRotation::identity(),
            object_translation: Vec3::ZERO,
            object_scale: Vec3::ONE,
            render_config: RenderConfig::tbp_default(),
            target_point: Vec3::ZERO,
            targeting_policy: TargetingPolicy::Origin,
        };

        // Create minimal output: 2x2 image
        let mut rgba = vec![0u8; 2 * 2 * 4];
        // Pixel (0,0) = red
        rgba[0] = 255;
        rgba[1] = 0;
        rgba[2] = 0;
        rgba[3] = 255;

        let output = BatchRenderOutput {
            request,
            rgba,
            depth: vec![1.0; 4],
            width: 2,
            height: 2,
            intrinsics: RenderConfig::tbp_default().intrinsics(),
            camera_transform: Transform::default(),
            object_translation: Vec3::ZERO,
            object_scale: Vec3::ONE,
            target_point: Vec3::ZERO,
            targeting_policy: TargetingPolicy::Origin,
            health: RenderHealth {
                center_pixel: Some([1, 1]),
                center_depth: Some(1.0),
                center_foreground: true,
                foreground_pixel_count: 4,
                foreground_coverage: 1.0,
                center_5x5_foreground_count: 4,
                nearest_foreground_pixel: Some([1, 1]),
                nearest_foreground_depth: Some(1.0),
                nearest_foreground_distance_px: Some(0.0),
            },
            status: RenderStatus::Success,
            error_message: None,
        };

        let rgb = output.to_rgb_image();
        assert_eq!(rgb.len(), 2); // 2 rows
        assert_eq!(rgb[0].len(), 2); // 2 cols
        assert_eq!(rgb[0][0], [255, 0, 0]); // Red
    }

    #[test]
    fn test_batch_render_output_carries_request_target_metadata() {
        let target_point = Vec3::new(0.25, -0.125, 0.5);
        let camera_transform = Transform::from_xyz(0.0, 0.0, 2.0).looking_at(Vec3::ZERO, Vec3::Y);
        let request = BatchRenderRequest {
            object_dir: "/tmp/test".into(),
            viewpoint: camera_transform,
            object_rotation: ObjectRotation::identity(),
            object_translation: Vec3::new(0.125, 0.25, -0.5),
            object_scale: Vec3::splat(1.25),
            render_config: RenderConfig::tbp_default(),
            target_point,
            targeting_policy: TargetingPolicy::MeshCenter,
        };
        let output = RenderOutput {
            rgba: vec![0u8; 4],
            depth: vec![1.0],
            width: 1,
            height: 1,
            intrinsics: RenderConfig::tbp_default().intrinsics(),
            camera_transform,
            object_rotation: ObjectRotation::identity(),
            object_translation: Vec3::new(0.125, 0.25, -0.5),
            object_scale: Vec3::splat(1.25),
            target_point: Vec3::ZERO,
            targeting_policy: TargetingPolicy::Origin,
        };

        let batch_output = BatchRenderOutput::from_render_output(request, output);

        assert_eq!(batch_output.target_point, target_point);
        assert_eq!(batch_output.targeting_policy, TargetingPolicy::MeshCenter);
        assert_eq!(batch_output.camera_transform, camera_transform);
        assert_eq!(
            batch_output.object_translation,
            Vec3::new(0.125, 0.25, -0.5)
        );
        assert_eq!(batch_output.object_scale, Vec3::splat(1.25));
        assert_eq!(batch_output.request.target_point, target_point);
        assert_eq!(
            batch_output.request.object_translation,
            Vec3::new(0.125, 0.25, -0.5)
        );
        assert_eq!(batch_output.request.object_scale, Vec3::splat(1.25));
        assert_eq!(
            batch_output.request.targeting_policy,
            TargetingPolicy::MeshCenter
        );
    }

    #[test]
    fn test_batch_render_output_semantic_3d_uses_camera_transform() {
        let camera_transform = Transform::from_xyz(0.0, 0.0, 2.0).looking_at(Vec3::ZERO, Vec3::Y);
        let request = BatchRenderRequest {
            object_dir: "/tmp/test".into(),
            viewpoint: camera_transform,
            object_rotation: ObjectRotation::identity(),
            object_translation: Vec3::ZERO,
            object_scale: Vec3::ONE,
            render_config: RenderConfig::tbp_default(),
            target_point: Vec3::ZERO,
            targeting_policy: TargetingPolicy::Origin,
        };
        let output = RenderOutput {
            rgba: vec![0u8; 4],
            depth: vec![1.5],
            width: 1,
            height: 1,
            intrinsics: CameraIntrinsics {
                focal_length: [100.0, 100.0],
                principal_point: [0.0, 0.0],
                image_size: [1, 1],
            },
            camera_transform,
            object_rotation: ObjectRotation::identity(),
            object_translation: Vec3::ZERO,
            object_scale: Vec3::ONE,
            target_point: Vec3::ZERO,
            targeting_policy: TargetingPolicy::Origin,
        };

        let batch_output = BatchRenderOutput::from_render_output(request, output);
        let rows = batch_output.semantic_3d(7);

        assert_eq!(rows.len(), 1);
        assert!((rows[0][0]).abs() < 1e-6);
        assert!((rows[0][1]).abs() < 1e-6);
        assert!((rows[0][2] - 0.5).abs() < 1e-6);
        assert_eq!(rows[0][3], 7.0);
    }
}
