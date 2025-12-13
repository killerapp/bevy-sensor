//! bevy-sensor: Multi-view rendering for YCB object dataset
//!
//! This library provides Bevy-based rendering of 3D objects from multiple viewpoints,
//! designed to match TBP (Thousand Brains Project) habitat sensor conventions for
//! use in neocortx sensorimotor learning experiments.
//!
//! # Example
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
        path.join("003_cracker_box/google_16k/textured.obj").exists()
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
    pub fn object_texture_path<P: AsRef<Path>>(output_dir: P, object_id: &str) -> std::path::PathBuf {
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
    pub pitch: f32,
    /// Rotation around Y-axis (yaw) in degrees
    pub yaw: f32,
    /// Rotation around Z-axis (roll) in degrees
    pub roll: f32,
}

impl ObjectRotation {
    /// Create a new rotation from Euler angles in degrees
    pub fn new(pitch: f32, yaw: f32, roll: f32) -> Self {
        Self { pitch, yaw, roll }
    }

    /// Create from TBP-style array [pitch, yaw, roll] in degrees
    pub fn from_array(arr: [f32; 3]) -> Self {
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
            Self::from_array([0.0, 0.0, 0.0]),     // Front
            Self::from_array([0.0, 90.0, 0.0]),    // Right
            Self::from_array([0.0, 180.0, 0.0]),   // Back
            Self::from_array([0.0, 270.0, 0.0]),   // Left
            Self::from_array([90.0, 0.0, 0.0]),    // Top
            Self::from_array([-90.0, 0.0, 0.0]),   // Bottom
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

    /// Convert to Bevy Quat
    pub fn to_quat(&self) -> Quat {
        Quat::from_euler(
            EulerRot::XYZ,
            self.pitch.to_radians(),
            self.yaw.to_radians(),
            self.roll.to_radians(),
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

// Re-export bevy types that consumers will need
pub use bevy::prelude::{Transform, Vec3, Quat};

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
}
