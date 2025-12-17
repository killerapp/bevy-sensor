//! bevy-sensor: Multi-view rendering for YCB object dataset
//!
//! This library provides Bevy-based rendering of 3D objects from multiple viewpoints,
//! designed to match TBP (Thousand Brains Project) habitat sensor conventions for
//! use in neocortx sensorimotor learning experiments.
//!
//! # Headless Rendering (NEW)
//!
//! Render directly to memory buffers for use in sensorimotor learning:
//!
//! ```ignore
//! use bevy_sensor::{render_to_buffer, RenderConfig, ViewpointConfig, ObjectRotation};
//! use std::path::Path;
//!
//! let config = RenderConfig::tbp_default(); // 64x64, RGBD
//! let viewpoint = bevy_sensor::generate_viewpoints(&ViewpointConfig::default())[0];
//! let rotation = ObjectRotation::identity();
//!
//! let output = render_to_buffer(
//!     Path::new("/tmp/ycb/003_cracker_box"),
//!     &viewpoint,
//!     &rotation,
//!     &config,
//! )?;
//!
//! // output.rgba: Vec<u8> - RGBA pixels (64*64*4 bytes)
//! // output.depth: Vec<f32> - Depth values (64*64 floats)
//! ```
//!
//! # File-based Capture (Legacy)
//!
//! ```ignore
//! use bevy_sensor::{SensorConfig, ViewpointConfig, ObjectRotation};
//!
//! let config = SensorConfig {
//!     viewpoints: ViewpointConfig::default(),
//!     object_rotations: ObjectRotation::tbp_benchmark_rotations(),
//!     ..Default::default()
//! };
//! ```
//!
//! # YCB Dataset
//!
//! Download YCB models programmatically:
//!
//! ```ignore
//! use bevy_sensor::ycb::{download_models, Subset};
//!
//! // Download representative subset (3 objects)
//! download_models("/tmp/ycb", Subset::Representative).await?;
//! ```

use bevy::prelude::*;
use std::f32::consts::PI;
use std::path::Path;

// Headless rendering implementation
// Full GPU rendering requires a display - see render module for details
mod render;

// Test fixtures for pre-rendered images (CI/CD support)
pub mod fixtures;

// Re-export ycbust types for convenience
pub use ycbust::{self, DownloadOptions, Subset as YcbSubset, REPRESENTATIVE_OBJECTS, TEN_OBJECTS};

/// YCB dataset utilities
pub mod ycb {
    pub use ycbust::{download_ycb, DownloadOptions, Subset, REPRESENTATIVE_OBJECTS, TEN_OBJECTS};

    use std::path::Path;

    /// Download YCB models to the specified directory.
    ///
    /// # Arguments
    /// * `output_dir` - Directory to download models to
    /// * `subset` - Which subset of objects to download
    ///
    /// # Example
    /// ```ignore
    /// use bevy_sensor::ycb::{download_models, Subset};
    ///
    /// download_models("/tmp/ycb", Subset::Representative).await?;
    /// ```
    pub async fn download_models<P: AsRef<Path>>(
        output_dir: P,
        subset: Subset,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let options = DownloadOptions {
            overwrite: false,
            full: false,
            show_progress: true,
            delete_archives: true,
        };
        download_ycb(subset, output_dir.as_ref(), options).await?;
        Ok(())
    }

    /// Download YCB models with custom options.
    pub async fn download_models_with_options<P: AsRef<Path>>(
        output_dir: P,
        subset: Subset,
        options: DownloadOptions,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        download_ycb(subset, output_dir.as_ref(), options).await?;
        Ok(())
    }

    /// Check if YCB models exist at the given path
    pub fn models_exist<P: AsRef<Path>>(output_dir: P) -> bool {
        let path = output_dir.as_ref();
        // Check for at least one representative object
        path.join("003_cracker_box/google_16k/textured.obj")
            .exists()
    }

    /// Get the path to a specific YCB object's OBJ file
    pub fn object_mesh_path<P: AsRef<Path>>(output_dir: P, object_id: &str) -> std::path::PathBuf {
        output_dir
            .as_ref()
            .join(object_id)
            .join("google_16k")
            .join("textured.obj")
    }

    /// Get the path to a specific YCB object's texture file
    pub fn object_texture_path<P: AsRef<Path>>(
        output_dir: P,
        object_id: &str,
    ) -> std::path::PathBuf {
        output_dir
            .as_ref()
            .join(object_id)
            .join("google_16k")
            .join("texture_map.png")
    }
}

/// Object rotation in Euler angles (degrees), matching TBP benchmark format.
/// Format: [pitch, yaw, roll] or [x, y, z] rotation.
#[derive(Clone, Debug, PartialEq)]
pub struct ObjectRotation {
    /// Rotation around X-axis (pitch) in degrees
    pub pitch: f64,
    /// Rotation around Y-axis (yaw) in degrees
    pub yaw: f64,
    /// Rotation around Z-axis (roll) in degrees
    pub roll: f64,
}

impl ObjectRotation {
    /// Create a new rotation from Euler angles in degrees
    pub fn new(pitch: f64, yaw: f64, roll: f64) -> Self {
        Self { pitch, yaw, roll }
    }

    /// Create from TBP-style array [pitch, yaw, roll] in degrees
    pub fn from_array(arr: [f64; 3]) -> Self {
        Self {
            pitch: arr[0],
            yaw: arr[1],
            roll: arr[2],
        }
    }

    /// Identity rotation (no rotation)
    pub fn identity() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }

    /// TBP benchmark rotations: [0,0,0], [0,90,0], [0,180,0]
    /// Used in shorter YCB experiments to reduce computational load.
    pub fn tbp_benchmark_rotations() -> Vec<Self> {
        vec![
            Self::from_array([0.0, 0.0, 0.0]),
            Self::from_array([0.0, 90.0, 0.0]),
            Self::from_array([0.0, 180.0, 0.0]),
        ]
    }

    /// TBP 14 known orientations (cube faces and corners)
    /// These are the orientations objects are learned in during training.
    pub fn tbp_known_orientations() -> Vec<Self> {
        vec![
            // 6 cube faces (90° rotations around each axis)
            Self::from_array([0.0, 0.0, 0.0]),   // Front
            Self::from_array([0.0, 90.0, 0.0]),  // Right
            Self::from_array([0.0, 180.0, 0.0]), // Back
            Self::from_array([0.0, 270.0, 0.0]), // Left
            Self::from_array([90.0, 0.0, 0.0]),  // Top
            Self::from_array([-90.0, 0.0, 0.0]), // Bottom
            // 8 cube corners (45° rotations)
            Self::from_array([45.0, 45.0, 0.0]),
            Self::from_array([45.0, 135.0, 0.0]),
            Self::from_array([45.0, 225.0, 0.0]),
            Self::from_array([45.0, 315.0, 0.0]),
            Self::from_array([-45.0, 45.0, 0.0]),
            Self::from_array([-45.0, 135.0, 0.0]),
            Self::from_array([-45.0, 225.0, 0.0]),
            Self::from_array([-45.0, 315.0, 0.0]),
        ]
    }

    /// Convert to Bevy Quat (converts f64 to f32 for Bevy compatibility)
    pub fn to_quat(&self) -> Quat {
        Quat::from_euler(
            EulerRot::XYZ,
            (self.pitch as f32).to_radians(),
            (self.yaw as f32).to_radians(),
            (self.roll as f32).to_radians(),
        )
    }

    /// Convert to Bevy Transform (rotation only, no translation)
    pub fn to_transform(&self) -> Transform {
        Transform::from_rotation(self.to_quat())
    }
}

impl Default for ObjectRotation {
    fn default() -> Self {
        Self::identity()
    }
}

/// Configuration for viewpoint generation matching TBP habitat sensor behavior.
/// Uses spherical coordinates to capture objects from multiple elevations.
#[derive(Clone, Debug)]
pub struct ViewpointConfig {
    /// Distance from camera to object center (meters)
    pub radius: f32,
    /// Number of horizontal positions (yaw angles) around the object
    pub yaw_count: usize,
    /// Elevation angles in degrees (pitch). Positive = above, negative = below.
    pub pitch_angles_deg: Vec<f32>,
}

impl Default for ViewpointConfig {
    fn default() -> Self {
        Self {
            radius: 0.5,
            yaw_count: 8,
            // Three elevations: below (-30°), level (0°), above (+30°)
            // This matches TBP's look_up/look_down capability
            pitch_angles_deg: vec![-30.0, 0.0, 30.0],
        }
    }
}

impl ViewpointConfig {
    /// Total number of viewpoints this config will generate
    pub fn viewpoint_count(&self) -> usize {
        self.yaw_count * self.pitch_angles_deg.len()
    }
}

/// Full sensor configuration for capture sessions
#[derive(Clone, Debug, Resource)]
pub struct SensorConfig {
    /// Viewpoint configuration (camera positions)
    pub viewpoints: ViewpointConfig,
    /// Object rotations to capture (each rotation generates a full viewpoint set)
    pub object_rotations: Vec<ObjectRotation>,
    /// Output directory for captures
    pub output_dir: String,
    /// Filename pattern (use {view} for view index, {rot} for rotation index)
    pub filename_pattern: String,
}

impl Default for SensorConfig {
    fn default() -> Self {
        Self {
            viewpoints: ViewpointConfig::default(),
            object_rotations: vec![ObjectRotation::identity()],
            output_dir: ".".to_string(),
            filename_pattern: "capture_{rot}_{view}.png".to_string(),
        }
    }
}

impl SensorConfig {
    /// Create config for TBP benchmark comparison (3 rotations × 24 viewpoints = 72 captures)
    pub fn tbp_benchmark() -> Self {
        Self {
            viewpoints: ViewpointConfig::default(),
            object_rotations: ObjectRotation::tbp_benchmark_rotations(),
            output_dir: ".".to_string(),
            filename_pattern: "capture_{rot}_{view}.png".to_string(),
        }
    }

    /// Create config for full TBP training (14 rotations × 24 viewpoints = 336 captures)
    pub fn tbp_full_training() -> Self {
        Self {
            viewpoints: ViewpointConfig::default(),
            object_rotations: ObjectRotation::tbp_known_orientations(),
            output_dir: ".".to_string(),
            filename_pattern: "capture_{rot}_{view}.png".to_string(),
        }
    }

    /// Total number of captures this config will generate
    pub fn total_captures(&self) -> usize {
        self.viewpoints.viewpoint_count() * self.object_rotations.len()
    }
}

/// Generate camera viewpoints using spherical coordinates.
///
/// Spherical coordinate system (matching TBP habitat sensor conventions):
/// - Yaw: horizontal rotation around Y-axis (0° to 360°)
/// - Pitch: elevation angle from horizontal plane (-90° to +90°)
/// - Radius: distance from origin (object center)
pub fn generate_viewpoints(config: &ViewpointConfig) -> Vec<Transform> {
    let mut views = Vec::with_capacity(config.viewpoint_count());

    for pitch_deg in &config.pitch_angles_deg {
        let pitch = pitch_deg.to_radians();

        for i in 0..config.yaw_count {
            let yaw = (i as f32) * 2.0 * PI / (config.yaw_count as f32);

            // Spherical to Cartesian conversion (Y-up coordinate system)
            // x = r * cos(pitch) * sin(yaw)
            // y = r * sin(pitch)
            // z = r * cos(pitch) * cos(yaw)
            let x = config.radius * pitch.cos() * yaw.sin();
            let y = config.radius * pitch.sin();
            let z = config.radius * pitch.cos() * yaw.cos();

            let transform = Transform::from_xyz(x, y, z).looking_at(Vec3::ZERO, Vec3::Y);
            views.push(transform);
        }
    }
    views
}

/// Marker component for the target object being captured
#[derive(Component)]
pub struct CaptureTarget;

/// Marker component for the capture camera
#[derive(Component)]
pub struct CaptureCamera;

// ============================================================================
// Headless Rendering API (NEW)
// ============================================================================

/// Configuration for headless rendering.
///
/// Matches TBP habitat sensor defaults: 64x64 resolution with RGBD output.
#[derive(Clone, Debug)]
pub struct RenderConfig {
    /// Image width in pixels (default: 64)
    pub width: u32,
    /// Image height in pixels (default: 64)
    pub height: u32,
    /// Zoom factor affecting field of view (default: 1.0)
    /// Use >1 to zoom in (narrower FOV), <1 to zoom out (wider FOV)
    pub zoom: f32,
    /// Near clipping plane in meters (default: 0.01)
    pub near_plane: f32,
    /// Far clipping plane in meters (default: 10.0)
    pub far_plane: f32,
    /// Lighting configuration
    pub lighting: LightingConfig,
}

/// Lighting configuration for rendering.
///
/// Controls ambient light and point lights in the scene.
#[derive(Clone, Debug)]
pub struct LightingConfig {
    /// Ambient light brightness (0.0 - 1.0, default: 0.3)
    pub ambient_brightness: f32,
    /// Key light intensity in lumens (default: 1500.0)
    pub key_light_intensity: f32,
    /// Key light position [x, y, z] (default: [4.0, 8.0, 4.0])
    pub key_light_position: [f32; 3],
    /// Fill light intensity in lumens (default: 500.0)
    pub fill_light_intensity: f32,
    /// Fill light position [x, y, z] (default: [-4.0, 2.0, -4.0])
    pub fill_light_position: [f32; 3],
    /// Enable shadows (default: false for performance)
    pub shadows_enabled: bool,
}

impl Default for LightingConfig {
    fn default() -> Self {
        Self {
            ambient_brightness: 0.3,
            key_light_intensity: 1500.0,
            key_light_position: [4.0, 8.0, 4.0],
            fill_light_intensity: 500.0,
            fill_light_position: [-4.0, 2.0, -4.0],
            shadows_enabled: false,
        }
    }
}

impl LightingConfig {
    /// Bright lighting for clear visibility
    pub fn bright() -> Self {
        Self {
            ambient_brightness: 0.5,
            key_light_intensity: 2000.0,
            key_light_position: [4.0, 8.0, 4.0],
            fill_light_intensity: 800.0,
            fill_light_position: [-4.0, 2.0, -4.0],
            shadows_enabled: false,
        }
    }

    /// Soft lighting with minimal shadows
    pub fn soft() -> Self {
        Self {
            ambient_brightness: 0.4,
            key_light_intensity: 1000.0,
            key_light_position: [3.0, 6.0, 3.0],
            fill_light_intensity: 600.0,
            fill_light_position: [-3.0, 3.0, -3.0],
            shadows_enabled: false,
        }
    }

    /// Unlit mode - ambient only, no point lights
    pub fn unlit() -> Self {
        Self {
            ambient_brightness: 1.0,
            key_light_intensity: 0.0,
            key_light_position: [0.0, 0.0, 0.0],
            fill_light_intensity: 0.0,
            fill_light_position: [0.0, 0.0, 0.0],
            shadows_enabled: false,
        }
    }
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self::tbp_default()
    }
}

impl RenderConfig {
    /// TBP-compatible 64x64 RGBD sensor configuration.
    ///
    /// This matches the default resolution used in TBP's habitat sensor.
    pub fn tbp_default() -> Self {
        Self {
            width: 64,
            height: 64,
            zoom: 1.0,
            near_plane: 0.01,
            far_plane: 10.0,
            lighting: LightingConfig::default(),
        }
    }

    /// Higher resolution configuration for debugging and visualization.
    pub fn preview() -> Self {
        Self {
            width: 256,
            height: 256,
            zoom: 1.0,
            near_plane: 0.01,
            far_plane: 10.0,
            lighting: LightingConfig::default(),
        }
    }

    /// High resolution configuration for detailed captures.
    pub fn high_res() -> Self {
        Self {
            width: 512,
            height: 512,
            zoom: 1.0,
            near_plane: 0.01,
            far_plane: 10.0,
            lighting: LightingConfig::default(),
        }
    }

    /// Calculate vertical field of view in radians based on zoom.
    ///
    /// Base FOV is 60 degrees, adjusted by zoom factor.
    pub fn fov_radians(&self) -> f32 {
        let base_fov_deg = 60.0_f32;
        (base_fov_deg / self.zoom).to_radians()
    }

    /// Compute camera intrinsics for use with neocortx.
    ///
    /// Returns focal length and principal point based on resolution and FOV.
    /// Uses f64 for TBP numerical precision compatibility.
    pub fn intrinsics(&self) -> CameraIntrinsics {
        let fov = self.fov_radians() as f64;
        // focal_length = (height/2) / tan(fov/2)
        let fy = (self.height as f64 / 2.0) / (fov / 2.0).tan();
        let fx = fy; // Assuming square pixels

        CameraIntrinsics {
            focal_length: [fx, fy],
            principal_point: [self.width as f64 / 2.0, self.height as f64 / 2.0],
            image_size: [self.width, self.height],
        }
    }
}

/// Camera intrinsic parameters for 3D reconstruction.
///
/// Compatible with neocortx's VisionIntrinsics format.
/// Uses f64 for TBP numerical precision compatibility.
#[derive(Clone, Debug, PartialEq)]
pub struct CameraIntrinsics {
    /// Focal length in pixels (fx, fy)
    pub focal_length: [f64; 2],
    /// Principal point (cx, cy) - typically image center
    pub principal_point: [f64; 2],
    /// Image dimensions (width, height)
    pub image_size: [u32; 2],
}

impl CameraIntrinsics {
    /// Project a 3D point to 2D pixel coordinates.
    pub fn project(&self, point: Vec3) -> Option<[f64; 2]> {
        if point.z <= 0.0 {
            return None;
        }
        let x = (point.x as f64 / point.z as f64) * self.focal_length[0] + self.principal_point[0];
        let y = (point.y as f64 / point.z as f64) * self.focal_length[1] + self.principal_point[1];
        Some([x, y])
    }

    /// Unproject a 2D pixel to a 3D point at given depth.
    pub fn unproject(&self, pixel: [f64; 2], depth: f64) -> [f64; 3] {
        let x = (pixel[0] - self.principal_point[0]) / self.focal_length[0] * depth;
        let y = (pixel[1] - self.principal_point[1]) / self.focal_length[1] * depth;
        [x, y, depth]
    }
}

/// Output from headless rendering containing RGBA and depth data.
#[derive(Clone, Debug)]
pub struct RenderOutput {
    /// RGBA pixel data in row-major order (width * height * 4 bytes)
    pub rgba: Vec<u8>,
    /// Depth values in meters, row-major order (width * height f64s)
    /// Values are linear depth from camera, not normalized.
    /// Uses f64 for TBP numerical precision compatibility.
    pub depth: Vec<f64>,
    /// Image width in pixels
    pub width: u32,
    /// Image height in pixels
    pub height: u32,
    /// Camera intrinsics used for this render
    pub intrinsics: CameraIntrinsics,
    /// Camera transform (world position and orientation)
    pub camera_transform: Transform,
    /// Object rotation applied during render
    pub object_rotation: ObjectRotation,
}

impl RenderOutput {
    /// Get RGBA pixel at (x, y). Returns None if out of bounds.
    pub fn get_rgba(&self, x: u32, y: u32) -> Option<[u8; 4]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = ((y * self.width + x) * 4) as usize;
        Some([
            self.rgba[idx],
            self.rgba[idx + 1],
            self.rgba[idx + 2],
            self.rgba[idx + 3],
        ])
    }

    /// Get depth value at (x, y) in meters. Returns None if out of bounds.
    pub fn get_depth(&self, x: u32, y: u32) -> Option<f64> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = (y * self.width + x) as usize;
        Some(self.depth[idx])
    }

    /// Get RGB pixel (without alpha) at (x, y).
    pub fn get_rgb(&self, x: u32, y: u32) -> Option<[u8; 3]> {
        self.get_rgba(x, y).map(|rgba| [rgba[0], rgba[1], rgba[2]])
    }

    /// Convert to neocortx-compatible image format: Vec<Vec<[u8; 3]>>
    pub fn to_rgb_image(&self) -> Vec<Vec<[u8; 3]>> {
        let mut image = Vec::with_capacity(self.height as usize);
        for y in 0..self.height {
            let mut row = Vec::with_capacity(self.width as usize);
            for x in 0..self.width {
                row.push(self.get_rgb(x, y).unwrap_or([0, 0, 0]));
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
                row.push(self.get_depth(x, y).unwrap_or(0.0));
            }
            image.push(row);
        }
        image
    }
}

/// Errors that can occur during rendering.
#[derive(Debug, Clone)]
pub enum RenderError {
    /// Object mesh file not found
    MeshNotFound(String),
    /// Object texture file not found
    TextureNotFound(String),
    /// Bevy rendering failed
    RenderFailed(String),
    /// Invalid configuration
    InvalidConfig(String),
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RenderError::MeshNotFound(path) => write!(f, "Mesh not found: {}", path),
            RenderError::TextureNotFound(path) => write!(f, "Texture not found: {}", path),
            RenderError::RenderFailed(msg) => write!(f, "Render failed: {}", msg),
            RenderError::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
        }
    }
}

impl std::error::Error for RenderError {}

/// Render a YCB object to an in-memory buffer.
///
/// This is the primary API for headless rendering. It spawns a minimal Bevy app,
/// renders a single frame, extracts the RGBA and depth data, and shuts down.
///
/// # Arguments
/// * `object_dir` - Path to YCB object directory (e.g., "/tmp/ycb/003_cracker_box")
/// * `camera_transform` - Camera position and orientation (use `generate_viewpoints`)
/// * `object_rotation` - Rotation to apply to the object
/// * `config` - Render configuration (resolution, depth range, etc.)
///
/// # Example
/// ```ignore
/// use bevy_sensor::{render_to_buffer, RenderConfig, ViewpointConfig, ObjectRotation};
/// use std::path::Path;
///
/// let viewpoints = bevy_sensor::generate_viewpoints(&ViewpointConfig::default());
/// let output = render_to_buffer(
///     Path::new("/tmp/ycb/003_cracker_box"),
///     &viewpoints[0],
///     &ObjectRotation::identity(),
///     &RenderConfig::tbp_default(),
/// )?;
/// ```
pub fn render_to_buffer(
    object_dir: &Path,
    camera_transform: &Transform,
    object_rotation: &ObjectRotation,
    config: &RenderConfig,
) -> Result<RenderOutput, RenderError> {
    // Use the actual Bevy headless renderer
    render::render_headless(object_dir, camera_transform, object_rotation, config)
}

/// Render all viewpoints and rotations for a YCB object.
///
/// Convenience function that renders all combinations of viewpoints and rotations.
///
/// # Arguments
/// * `object_dir` - Path to YCB object directory
/// * `viewpoint_config` - Viewpoint configuration (camera positions)
/// * `rotations` - Object rotations to render
/// * `render_config` - Render configuration
///
/// # Returns
/// Vector of RenderOutput, one per viewpoint × rotation combination.
pub fn render_all_viewpoints(
    object_dir: &Path,
    viewpoint_config: &ViewpointConfig,
    rotations: &[ObjectRotation],
    render_config: &RenderConfig,
) -> Result<Vec<RenderOutput>, RenderError> {
    let viewpoints = generate_viewpoints(viewpoint_config);
    let mut outputs = Vec::with_capacity(viewpoints.len() * rotations.len());

    for rotation in rotations {
        for viewpoint in &viewpoints {
            let output = render_to_buffer(object_dir, viewpoint, rotation, render_config)?;
            outputs.push(output);
        }
    }

    Ok(outputs)
}

/// Render directly to files (for subprocess mode).
///
/// This function is designed for subprocess rendering where the process will exit
/// after rendering. It saves RGBA and depth data directly to the specified files
/// before the process terminates.
///
/// # Arguments
/// * `object_dir` - Path to YCB object directory
/// * `camera_transform` - Camera position and orientation
/// * `object_rotation` - Rotation to apply to the object
/// * `config` - Render configuration
/// * `rgba_path` - Output path for RGBA PNG
/// * `depth_path` - Output path for depth data (raw f32 bytes)
///
/// # Note
/// This function may call `std::process::exit(0)` and not return.
pub fn render_to_files(
    object_dir: &Path,
    camera_transform: &Transform,
    object_rotation: &ObjectRotation,
    config: &RenderConfig,
    rgba_path: &Path,
    depth_path: &Path,
) -> Result<(), RenderError> {
    render::render_to_files(
        object_dir,
        camera_transform,
        object_rotation,
        config,
        rgba_path,
        depth_path,
    )
}

// Re-export bevy types that consumers will need
pub use bevy::prelude::{Quat, Transform, Vec3};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_object_rotation_identity() {
        let rot = ObjectRotation::identity();
        assert_eq!(rot.pitch, 0.0);
        assert_eq!(rot.yaw, 0.0);
        assert_eq!(rot.roll, 0.0);
    }

    #[test]
    fn test_object_rotation_from_array() {
        let rot = ObjectRotation::from_array([10.0, 20.0, 30.0]);
        assert_eq!(rot.pitch, 10.0);
        assert_eq!(rot.yaw, 20.0);
        assert_eq!(rot.roll, 30.0);
    }

    #[test]
    fn test_tbp_benchmark_rotations() {
        let rotations = ObjectRotation::tbp_benchmark_rotations();
        assert_eq!(rotations.len(), 3);
        assert_eq!(rotations[0], ObjectRotation::from_array([0.0, 0.0, 0.0]));
        assert_eq!(rotations[1], ObjectRotation::from_array([0.0, 90.0, 0.0]));
        assert_eq!(rotations[2], ObjectRotation::from_array([0.0, 180.0, 0.0]));
    }

    #[test]
    fn test_tbp_known_orientations_count() {
        let orientations = ObjectRotation::tbp_known_orientations();
        assert_eq!(orientations.len(), 14);
    }

    #[test]
    fn test_rotation_to_quat() {
        let rot = ObjectRotation::identity();
        let quat = rot.to_quat();
        // Identity quaternion should be approximately (1, 0, 0, 0)
        assert!((quat.w - 1.0).abs() < 0.001);
        assert!(quat.x.abs() < 0.001);
        assert!(quat.y.abs() < 0.001);
        assert!(quat.z.abs() < 0.001);
    }

    #[test]
    fn test_rotation_90_yaw() {
        let rot = ObjectRotation::new(0.0, 90.0, 0.0);
        let quat = rot.to_quat();
        // 90° Y rotation: w ≈ 0.707, y ≈ 0.707
        assert!((quat.w - 0.707).abs() < 0.01);
        assert!((quat.y - 0.707).abs() < 0.01);
    }

    #[test]
    fn test_viewpoint_config_default() {
        let config = ViewpointConfig::default();
        assert_eq!(config.radius, 0.5);
        assert_eq!(config.yaw_count, 8);
        assert_eq!(config.pitch_angles_deg.len(), 3);
    }

    #[test]
    fn test_viewpoint_count() {
        let config = ViewpointConfig::default();
        assert_eq!(config.viewpoint_count(), 24); // 8 × 3
    }

    #[test]
    fn test_generate_viewpoints_count() {
        let config = ViewpointConfig::default();
        let viewpoints = generate_viewpoints(&config);
        assert_eq!(viewpoints.len(), 24);
    }

    #[test]
    fn test_viewpoints_spherical_radius() {
        let config = ViewpointConfig::default();
        let viewpoints = generate_viewpoints(&config);

        for (i, transform) in viewpoints.iter().enumerate() {
            let actual_radius = transform.translation.length();
            assert!(
                (actual_radius - config.radius).abs() < 0.001,
                "Viewpoint {} has incorrect radius: {} (expected {})",
                i,
                actual_radius,
                config.radius
            );
        }
    }

    #[test]
    fn test_viewpoints_looking_at_origin() {
        let config = ViewpointConfig::default();
        let viewpoints = generate_viewpoints(&config);

        for (i, transform) in viewpoints.iter().enumerate() {
            let forward = transform.forward();
            let to_origin = (Vec3::ZERO - transform.translation).normalize();
            let dot = forward.dot(to_origin);
            assert!(
                dot > 0.99,
                "Viewpoint {} not looking at origin, dot product: {}",
                i,
                dot
            );
        }
    }

    #[test]
    fn test_sensor_config_default() {
        let config = SensorConfig::default();
        assert_eq!(config.object_rotations.len(), 1);
        assert_eq!(config.total_captures(), 24);
    }

    #[test]
    fn test_sensor_config_tbp_benchmark() {
        let config = SensorConfig::tbp_benchmark();
        assert_eq!(config.object_rotations.len(), 3);
        assert_eq!(config.total_captures(), 72); // 3 rotations × 24 viewpoints
    }

    #[test]
    fn test_sensor_config_tbp_full() {
        let config = SensorConfig::tbp_full_training();
        assert_eq!(config.object_rotations.len(), 14);
        assert_eq!(config.total_captures(), 336); // 14 rotations × 24 viewpoints
    }

    #[test]
    fn test_ycb_representative_objects() {
        // Verify representative objects are defined
        assert_eq!(crate::ycb::REPRESENTATIVE_OBJECTS.len(), 3);
        assert!(crate::ycb::REPRESENTATIVE_OBJECTS.contains(&"003_cracker_box"));
    }

    #[test]
    fn test_ycb_ten_objects() {
        // Verify ten objects subset is defined
        assert_eq!(crate::ycb::TEN_OBJECTS.len(), 10);
    }

    #[test]
    fn test_ycb_object_mesh_path() {
        let path = crate::ycb::object_mesh_path("/tmp/ycb", "003_cracker_box");
        assert_eq!(
            path.to_string_lossy(),
            "/tmp/ycb/003_cracker_box/google_16k/textured.obj"
        );
    }

    #[test]
    fn test_ycb_object_texture_path() {
        let path = crate::ycb::object_texture_path("/tmp/ycb", "003_cracker_box");
        assert_eq!(
            path.to_string_lossy(),
            "/tmp/ycb/003_cracker_box/google_16k/texture_map.png"
        );
    }

    // =========================================================================
    // Headless Rendering API Tests
    // =========================================================================

    #[test]
    fn test_render_config_tbp_default() {
        let config = RenderConfig::tbp_default();
        assert_eq!(config.width, 64);
        assert_eq!(config.height, 64);
        assert_eq!(config.zoom, 1.0);
        assert_eq!(config.near_plane, 0.01);
        assert_eq!(config.far_plane, 10.0);
    }

    #[test]
    fn test_render_config_preview() {
        let config = RenderConfig::preview();
        assert_eq!(config.width, 256);
        assert_eq!(config.height, 256);
    }

    #[test]
    fn test_render_config_default_is_tbp() {
        let default = RenderConfig::default();
        let tbp = RenderConfig::tbp_default();
        assert_eq!(default.width, tbp.width);
        assert_eq!(default.height, tbp.height);
    }

    #[test]
    fn test_render_config_fov() {
        let config = RenderConfig::tbp_default();
        let fov = config.fov_radians();
        // Base FOV is 60 degrees = ~1.047 radians
        assert!((fov - 1.047).abs() < 0.01);

        // Zoom in should reduce FOV
        let zoomed = RenderConfig {
            zoom: 2.0,
            ..config
        };
        assert!(zoomed.fov_radians() < fov);
    }

    #[test]
    fn test_render_config_intrinsics() {
        let config = RenderConfig::tbp_default();
        let intrinsics = config.intrinsics();

        assert_eq!(intrinsics.image_size, [64, 64]);
        assert_eq!(intrinsics.principal_point, [32.0, 32.0]);
        // Focal length should be positive and reasonable
        assert!(intrinsics.focal_length[0] > 0.0);
        assert!(intrinsics.focal_length[1] > 0.0);
        // For 64x64 with 60° FOV, focal length ≈ 55.4 pixels
        assert!((intrinsics.focal_length[0] - 55.4).abs() < 1.0);
    }

    #[test]
    fn test_camera_intrinsics_project() {
        let intrinsics = CameraIntrinsics {
            focal_length: [100.0, 100.0],
            principal_point: [32.0, 32.0],
            image_size: [64, 64],
        };

        // Point at origin of camera frame projects to principal point
        let center = intrinsics.project(Vec3::new(0.0, 0.0, 1.0));
        assert!(center.is_some());
        let [x, y] = center.unwrap();
        assert!((x - 32.0).abs() < 0.001);
        assert!((y - 32.0).abs() < 0.001);

        // Point behind camera returns None
        let behind = intrinsics.project(Vec3::new(0.0, 0.0, -1.0));
        assert!(behind.is_none());
    }

    #[test]
    fn test_camera_intrinsics_unproject() {
        let intrinsics = CameraIntrinsics {
            focal_length: [100.0, 100.0],
            principal_point: [32.0, 32.0],
            image_size: [64, 64],
        };

        // Unproject principal point at depth 1.0
        let point = intrinsics.unproject([32.0, 32.0], 1.0);
        assert!((point[0]).abs() < 0.001); // x
        assert!((point[1]).abs() < 0.001); // y
        assert!((point[2] - 1.0).abs() < 0.001); // z
    }

    #[test]
    fn test_render_output_get_rgba() {
        let output = RenderOutput {
            rgba: vec![
                255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
            ],
            depth: vec![1.0, 2.0, 3.0, 4.0],
            width: 2,
            height: 2,
            intrinsics: RenderConfig::tbp_default().intrinsics(),
            camera_transform: Transform::IDENTITY,
            object_rotation: ObjectRotation::identity(),
        };

        // Top-left: red
        assert_eq!(output.get_rgba(0, 0), Some([255, 0, 0, 255]));
        // Top-right: green
        assert_eq!(output.get_rgba(1, 0), Some([0, 255, 0, 255]));
        // Bottom-left: blue
        assert_eq!(output.get_rgba(0, 1), Some([0, 0, 255, 255]));
        // Bottom-right: white
        assert_eq!(output.get_rgba(1, 1), Some([255, 255, 255, 255]));
        // Out of bounds
        assert_eq!(output.get_rgba(2, 0), None);
    }

    #[test]
    fn test_render_output_get_depth() {
        let output = RenderOutput {
            rgba: vec![0u8; 16],
            depth: vec![1.0, 2.0, 3.0, 4.0],
            width: 2,
            height: 2,
            intrinsics: RenderConfig::tbp_default().intrinsics(),
            camera_transform: Transform::IDENTITY,
            object_rotation: ObjectRotation::identity(),
        };

        assert_eq!(output.get_depth(0, 0), Some(1.0));
        assert_eq!(output.get_depth(1, 0), Some(2.0));
        assert_eq!(output.get_depth(0, 1), Some(3.0));
        assert_eq!(output.get_depth(1, 1), Some(4.0));
        assert_eq!(output.get_depth(2, 0), None);
    }

    #[test]
    fn test_render_output_to_rgb_image() {
        let output = RenderOutput {
            rgba: vec![
                255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
            ],
            depth: vec![1.0, 2.0, 3.0, 4.0],
            width: 2,
            height: 2,
            intrinsics: RenderConfig::tbp_default().intrinsics(),
            camera_transform: Transform::IDENTITY,
            object_rotation: ObjectRotation::identity(),
        };

        let image = output.to_rgb_image();
        assert_eq!(image.len(), 2); // 2 rows
        assert_eq!(image[0].len(), 2); // 2 columns
        assert_eq!(image[0][0], [255, 0, 0]); // Red
        assert_eq!(image[0][1], [0, 255, 0]); // Green
        assert_eq!(image[1][0], [0, 0, 255]); // Blue
        assert_eq!(image[1][1], [255, 255, 255]); // White
    }

    #[test]
    fn test_render_output_to_depth_image() {
        let output = RenderOutput {
            rgba: vec![0u8; 16],
            depth: vec![1.0, 2.0, 3.0, 4.0],
            width: 2,
            height: 2,
            intrinsics: RenderConfig::tbp_default().intrinsics(),
            camera_transform: Transform::IDENTITY,
            object_rotation: ObjectRotation::identity(),
        };

        let depth_image = output.to_depth_image();
        assert_eq!(depth_image.len(), 2);
        assert_eq!(depth_image[0], vec![1.0, 2.0]);
        assert_eq!(depth_image[1], vec![3.0, 4.0]);
    }

    #[test]
    fn test_render_error_display() {
        let err = RenderError::MeshNotFound("/path/to/mesh.obj".to_string());
        assert!(err.to_string().contains("Mesh not found"));
        assert!(err.to_string().contains("/path/to/mesh.obj"));
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_object_rotation_extreme_angles() {
        // Test angles beyond 360 degrees
        let rot = ObjectRotation::new(450.0, -720.0, 1080.0);
        let quat = rot.to_quat();
        // Quaternion should still be valid (normalized)
        assert!((quat.length() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_object_rotation_to_transform() {
        let rot = ObjectRotation::new(45.0, 90.0, 0.0);
        let transform = rot.to_transform();
        // Transform should have no translation
        assert_eq!(transform.translation, Vec3::ZERO);
        // Should have rotation
        assert!(transform.rotation != Quat::IDENTITY);
    }

    #[test]
    fn test_viewpoint_config_single_viewpoint() {
        let config = ViewpointConfig {
            radius: 1.0,
            yaw_count: 1,
            pitch_angles_deg: vec![0.0],
        };
        assert_eq!(config.viewpoint_count(), 1);
        let viewpoints = generate_viewpoints(&config);
        assert_eq!(viewpoints.len(), 1);
        // Single viewpoint at yaw=0, pitch=0 should be at (0, 0, radius)
        let pos = viewpoints[0].translation;
        assert!((pos.x).abs() < 0.001);
        assert!((pos.y).abs() < 0.001);
        assert!((pos.z - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_viewpoint_radius_scaling() {
        let config1 = ViewpointConfig {
            radius: 0.5,
            yaw_count: 4,
            pitch_angles_deg: vec![0.0],
        };
        let config2 = ViewpointConfig {
            radius: 2.0,
            yaw_count: 4,
            pitch_angles_deg: vec![0.0],
        };

        let v1 = generate_viewpoints(&config1);
        let v2 = generate_viewpoints(&config2);

        // Viewpoints should scale proportionally
        for (vp1, vp2) in v1.iter().zip(v2.iter()) {
            let ratio = vp2.translation.length() / vp1.translation.length();
            assert!((ratio - 4.0).abs() < 0.01); // 2.0 / 0.5 = 4.0
        }
    }

    #[test]
    fn test_camera_intrinsics_project_at_z_zero() {
        let intrinsics = CameraIntrinsics {
            focal_length: [100.0, 100.0],
            principal_point: [32.0, 32.0],
            image_size: [64, 64],
        };

        // Point at z=0 should return None (division by zero protection)
        let result = intrinsics.project(Vec3::new(1.0, 1.0, 0.0));
        assert!(result.is_none());
    }

    #[test]
    fn test_camera_intrinsics_roundtrip() {
        let intrinsics = CameraIntrinsics {
            focal_length: [100.0, 100.0],
            principal_point: [32.0, 32.0],
            image_size: [64, 64],
        };

        // Project a 3D point
        let original = Vec3::new(0.5, -0.3, 2.0);
        let projected = intrinsics.project(original).unwrap();

        // Unproject back with the same depth (convert f32 to f64)
        let unprojected = intrinsics.unproject(projected, original.z as f64);

        // Should get back approximately the same point
        assert!((unprojected[0] - original.x as f64).abs() < 0.001); // x
        assert!((unprojected[1] - original.y as f64).abs() < 0.001); // y
        assert!((unprojected[2] - original.z as f64).abs() < 0.001); // z
    }

    #[test]
    fn test_render_output_empty() {
        let output = RenderOutput {
            rgba: vec![],
            depth: vec![],
            width: 0,
            height: 0,
            intrinsics: RenderConfig::tbp_default().intrinsics(),
            camera_transform: Transform::IDENTITY,
            object_rotation: ObjectRotation::identity(),
        };

        // Should handle empty gracefully
        assert_eq!(output.get_rgba(0, 0), None);
        assert_eq!(output.get_depth(0, 0), None);
        assert!(output.to_rgb_image().is_empty());
        assert!(output.to_depth_image().is_empty());
    }

    #[test]
    fn test_render_output_1x1() {
        let output = RenderOutput {
            rgba: vec![128, 64, 32, 255],
            depth: vec![0.5],
            width: 1,
            height: 1,
            intrinsics: RenderConfig::tbp_default().intrinsics(),
            camera_transform: Transform::IDENTITY,
            object_rotation: ObjectRotation::identity(),
        };

        assert_eq!(output.get_rgba(0, 0), Some([128, 64, 32, 255]));
        assert_eq!(output.get_depth(0, 0), Some(0.5));
        assert_eq!(output.get_rgb(0, 0), Some([128, 64, 32]));

        let rgb_img = output.to_rgb_image();
        assert_eq!(rgb_img.len(), 1);
        assert_eq!(rgb_img[0].len(), 1);
        assert_eq!(rgb_img[0][0], [128, 64, 32]);
    }

    #[test]
    fn test_render_config_high_res() {
        let config = RenderConfig::high_res();
        assert_eq!(config.width, 512);
        assert_eq!(config.height, 512);

        let intrinsics = config.intrinsics();
        assert_eq!(intrinsics.image_size, [512, 512]);
        assert_eq!(intrinsics.principal_point, [256.0, 256.0]);
    }

    #[test]
    fn test_render_config_zoom_affects_fov() {
        let base = RenderConfig::tbp_default();
        let zoomed = RenderConfig {
            zoom: 2.0,
            ..base.clone()
        };

        // Higher zoom = lower FOV
        assert!(zoomed.fov_radians() < base.fov_radians());
        // Specifically, 2x zoom = half FOV
        assert!((zoomed.fov_radians() - base.fov_radians() / 2.0).abs() < 0.01);
    }

    #[test]
    fn test_render_config_zoom_affects_intrinsics() {
        let base = RenderConfig::tbp_default();
        let zoomed = RenderConfig {
            zoom: 2.0,
            ..base.clone()
        };

        // Higher zoom = higher focal length
        let base_intrinsics = base.intrinsics();
        let zoomed_intrinsics = zoomed.intrinsics();

        assert!(zoomed_intrinsics.focal_length[0] > base_intrinsics.focal_length[0]);
    }

    #[test]
    fn test_lighting_config_variants() {
        let default = LightingConfig::default();
        let bright = LightingConfig::bright();
        let soft = LightingConfig::soft();
        let unlit = LightingConfig::unlit();

        // Bright should have higher intensity than default
        assert!(bright.key_light_intensity > default.key_light_intensity);

        // Unlit should have no point lights
        assert_eq!(unlit.key_light_intensity, 0.0);
        assert_eq!(unlit.fill_light_intensity, 0.0);
        assert_eq!(unlit.ambient_brightness, 1.0);

        // Soft should have lower intensity
        assert!(soft.key_light_intensity < default.key_light_intensity);
    }

    #[test]
    fn test_all_render_error_variants() {
        let errors = vec![
            RenderError::MeshNotFound("mesh.obj".to_string()),
            RenderError::TextureNotFound("texture.png".to_string()),
            RenderError::RenderFailed("GPU error".to_string()),
            RenderError::InvalidConfig("bad config".to_string()),
        ];

        for err in errors {
            // All variants should have Display impl
            let msg = err.to_string();
            assert!(!msg.is_empty());
        }
    }

    #[test]
    fn test_tbp_known_orientations_unique() {
        let orientations = ObjectRotation::tbp_known_orientations();

        // All 14 orientations should produce unique quaternions
        let quats: Vec<Quat> = orientations.iter().map(|r| r.to_quat()).collect();

        for (i, q1) in quats.iter().enumerate() {
            for (j, q2) in quats.iter().enumerate() {
                if i != j {
                    // Quaternions should be different (accounting for q == -q equivalence)
                    let dot = q1.dot(*q2).abs();
                    assert!(
                        dot < 0.999,
                        "Orientations {} and {} produce same quaternion",
                        i,
                        j
                    );
                }
            }
        }
    }
}
