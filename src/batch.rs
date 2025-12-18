//! Batch rendering API for multiple viewpoints and objects.
//!
//! This module provides efficient batch rendering that eliminates subprocess spawning and
//! Bevy app initialization overhead. A single Bevy app instance is kept alive and reused
//! to render multiple viewpoints, achieving 10-100x speedup for typical batches.
//!
//! # Example
//!
//! ```ignore
//! use bevy_sensor::{
//!     create_batch_renderer, queue_render_request, render_next_in_batch,
//!     batch::BatchRenderRequest, BatchRenderConfig, RenderConfig, ObjectRotation,
//! };
//! use std::path::PathBuf;
//!
//! // Create a persistent renderer (initializes once)
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
//!             render_config: RenderConfig::tbp_default(),
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

use crate::{CameraIntrinsics, ObjectRotation, RenderConfig, RenderOutput};
use bevy::prelude::Transform;
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
    /// Render configuration (resolution, lighting, etc.)
    pub render_config: RenderConfig,
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

    /// Convert from RenderOutput, copying all fields
    pub fn from_render_output(request: BatchRenderRequest, output: RenderOutput) -> Self {
        Self {
            request,
            rgba: output.rgba,
            depth: output.depth,
            width: output.width,
            height: output.height,
            intrinsics: output.intrinsics,
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

/// Manages a persistent Bevy app for batch rendering.
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
            render_config: RenderConfig::tbp_default(),
        };
        assert!(renderer.queue_request(request).is_ok());
        assert_eq!(renderer.pending_count(), 1);
    }

    #[test]
    fn test_queue_full() {
        let mut config = BatchRenderConfig::default();
        config.max_batch_size = 1;
        let mut renderer = BatchRenderer::new(config);

        let request = BatchRenderRequest {
            object_dir: "/tmp/test".into(),
            viewpoint: Transform::default(),
            object_rotation: ObjectRotation::identity(),
            render_config: RenderConfig::tbp_default(),
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
            render_config: RenderConfig::tbp_default(),
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
            status: RenderStatus::Success,
            error_message: None,
        };

        let rgb = output.to_rgb_image();
        assert_eq!(rgb.len(), 2); // 2 rows
        assert_eq!(rgb[0].len(), 2); // 2 cols
        assert_eq!(rgb[0][0], [255, 0, 0]); // Red
    }
}
