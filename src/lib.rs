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
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use std::path::{Path, PathBuf};

// Headless rendering implementation
// Full GPU rendering requires a display - see render module for details
mod render;

// Batch rendering API for efficient multi-viewpoint rendering
pub mod batch;

// Benchmark helpers for renderer throughput artifacts
pub mod benchmark;

// WebGPU and cross-platform backend support
pub mod backend;

// Model caching system for efficient multi-viewpoint rendering
pub mod cache;

// Test fixtures for pre-rendered images (CI/CD support)
pub mod fixtures;

/// Stable renderer/targeting-policy version for cache manifests.
pub const RENDERER_POLICY_VERSION: &str = "tbp-targeting-v1";

// Re-export ycbust types for convenience
pub use ycbust::{
    self, DownloadOptions, Subset as YcbSubset, GOOGLE_16K_MESH_RELATIVE, REPRESENTATIVE_OBJECTS,
    TBP_SIMILAR_OBJECTS, TBP_STANDARD_OBJECTS,
};

/// YCB dataset utilities
pub mod ycb {
    pub use ycbust::{
        download_ycb, DownloadOptions, Subset, REPRESENTATIVE_OBJECTS, TBP_SIMILAR_OBJECTS,
        TBP_STANDARD_OBJECTS,
    };

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
        download_ycb(subset, output_dir.as_ref(), DownloadOptions::default()).await?;
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

    /// Download specific YCB objects by object ID using the standard `google_16k` meshes.
    ///
    /// Thin wrapper over [`ycbust::download_objects`] (added upstream in v0.3.3):
    /// preserves this crate's ergonomic `P: AsRef<Path>` surface while delegating
    /// skip / resume / integrity / parallelism to the upstream implementation.
    pub async fn download_objects<P: AsRef<Path>>(
        output_dir: P,
        object_ids: &[&str],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        ycbust::download_objects(object_ids, output_dir.as_ref(), DownloadOptions::default())
            .await?;
        Ok(())
    }

    /// Return object IDs whose standard `google_16k` mesh or texture is missing.
    pub fn missing_objects<P: AsRef<Path>>(output_dir: P, object_ids: &[&str]) -> Vec<String> {
        ycbust::validate_objects(output_dir.as_ref(), object_ids)
            .into_iter()
            .filter(|validation| !validation.is_complete())
            .map(|validation| validation.name)
            .collect()
    }

    /// Check if all requested YCB objects exist at the given path.
    pub fn objects_exist<P: AsRef<Path>>(output_dir: P, object_ids: &[&str]) -> bool {
        missing_objects(output_dir, object_ids).is_empty()
    }

    /// Check if the representative YCB models exist at the given path.
    pub fn models_exist<P: AsRef<Path>>(output_dir: P) -> bool {
        objects_exist(output_dir, REPRESENTATIVE_OBJECTS)
    }

    /// Get the path to a specific YCB object's OBJ file
    pub fn object_mesh_path<P: AsRef<Path>>(output_dir: P, object_id: &str) -> std::path::PathBuf {
        ycbust::object_mesh_path(output_dir.as_ref(), object_id)
    }

    /// Get the path to a specific YCB object's texture file
    pub fn object_texture_path<P: AsRef<Path>>(
        output_dir: P,
        object_id: &str,
    ) -> std::path::PathBuf {
        ycbust::object_texture_path(output_dir.as_ref(), object_id)
    }
}

/// Initialize bevy-sensor rendering backend configuration.
///
/// **IMPORTANT**: Call this function ONCE at the start of your application,
/// before any rendering operations, especially when using bevy-sensor as a library.
///
/// This ensures proper backend selection (WebGPU for WSL2, Vulkan for Linux, etc.)
/// and is critical for GPU rendering on WSL2 environments.
///
/// # Why This Matters
///
/// The WGPU rendering backend caches its backend selection early during initialization.
/// When bevy-sensor is used as a library, environment variables must be set BEFORE
/// any GPU rendering code runs. This function does that automatically.
///
/// # Example
///
/// ```ignore
/// use bevy_sensor;
///
/// fn main() {
///     // Initialize FIRST, before any rendering
///     bevy_sensor::initialize();
///
///     // Now use the rendering API
///     let output = bevy_sensor::render_to_buffer(
///         object_dir, &viewpoint, &rotation, &config
///     )?;
/// }
/// ```
///
/// # Calling Multiple Times
///
/// Safe to call multiple times - subsequent calls are no-ops after the first call.
pub fn initialize() {
    // Use a OnceCell equivalent to ensure this only runs once
    use std::sync::atomic::{AtomicBool, Ordering};
    static INITIALIZED: AtomicBool = AtomicBool::new(false);

    if !INITIALIZED.swap(true, Ordering::SeqCst) {
        // First call - initialize backend
        let config = backend::BackendConfig::new();
        config.apply_env();
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

/// Axis-aligned mesh bounds in object-local coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshBounds {
    /// Minimum object-local vertex coordinate.
    pub min: Vec3,
    /// Maximum object-local vertex coordinate.
    pub max: Vec3,
    /// Center of the axis-aligned bounding box.
    pub center: Vec3,
    /// Number of vertices inspected while computing the bounds.
    pub vertex_count: usize,
}

impl MeshBounds {
    /// Size of the axis-aligned bounding box on each axis.
    pub fn extents(&self) -> Vec3 {
        self.max - self.min
    }
}

/// Render-target selection policy for TBP/YCB camera orbits.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "policy", content = "target", rename_all = "snake_case")]
pub enum TargetingPolicy {
    /// Preserve historical behavior: camera viewpoints look at world origin.
    Origin,
    /// Load the YCB mesh AABB center and rotate it by the object rotation.
    MeshCenter,
    /// Use a caller-provided world target point.
    ExplicitTarget([f32; 3]),
}

impl TargetingPolicy {
    /// Stable label for manifests and logs.
    pub fn label(&self) -> &'static str {
        match self {
            TargetingPolicy::Origin => "origin",
            TargetingPolicy::MeshCenter => "mesh-center",
            TargetingPolicy::ExplicitTarget(_) => "explicit-target",
        }
    }
}

/// Generated viewpoints plus the target metadata used to create them.
#[derive(Clone, Debug, PartialEq)]
pub struct TargetedViewpoints {
    /// Targeting policy used for this viewpoint set.
    pub policy: TargetingPolicy,
    /// Point every viewpoint looks at in world coordinates.
    pub target_point: Vec3,
    /// Mesh bounds when the policy required loading object-local bounds.
    pub mesh_bounds: Option<MeshBounds>,
    /// Camera viewpoints for the selected policy.
    pub viewpoints: Vec<Transform>,
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
    generate_viewpoints_around_target(config, Vec3::ZERO)
}

/// Generate camera viewpoints around an explicit target point.
///
/// The generated camera offsets match [`generate_viewpoints`], but each camera
/// is translated by `target` and rotated to look at that target. This is the
/// caller-provided target form used by NeoCortx parity probes that should not
/// assume the object surface of interest is at the world origin.
pub fn generate_viewpoints_around_target(config: &ViewpointConfig, target: Vec3) -> Vec<Transform> {
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

            let translation = target + Vec3::new(x, y, z);
            let transform = Transform::from_translation(translation).looking_at(target, Vec3::Y);
            views.push(transform);
        }
    }
    views
}

/// Rotate an object-local mesh center into the rendered world frame.
///
/// This uses the same object-rotation convention as rendering itself
/// (`ObjectRotation::to_quat`). It intentionally applies yaw-only rotations as
/// well as pitch/roll rotations, so downstream parity code does not need a
/// temporary special case for centered YCB renders.
pub fn rotated_mesh_center(mesh_center: Vec3, object_rotation: &ObjectRotation) -> Vec3 {
    object_rotation.to_quat() * mesh_center
}

/// Generate TBP viewpoint transforms around a rotated object mesh center.
///
/// Use this when the YCB mesh's AABB center is a better render target than the
/// source origin. The camera orbit remains exactly the same shape as
/// [`generate_viewpoints`], but centered on `object_rotation * mesh_center`.
pub fn generate_object_centered_viewpoints(
    config: &ViewpointConfig,
    mesh_center: Vec3,
    object_rotation: &ObjectRotation,
) -> Vec<Transform> {
    generate_viewpoints_around_target(config, rotated_mesh_center(mesh_center, object_rotation))
}

/// Load axis-aligned bounds from an OBJ mesh.
///
/// This is a small public wrapper around the same YCB `google_16k/textured.obj`
/// layout used by the renderer. It lets downstream callers avoid carrying their
/// own OBJ parsing just to target an object's visual center.
pub fn load_mesh_bounds(mesh_path: &Path) -> Result<MeshBounds, RenderError> {
    if !mesh_path.exists() {
        return Err(RenderError::MeshNotFound(mesh_path.display().to_string()));
    }

    let (models, _) = tobj::load_obj(
        mesh_path,
        &tobj::LoadOptions {
            triangulate: false,
            single_index: true,
            ..Default::default()
        },
    )
    .map_err(|err| {
        RenderError::DataParsingError(format!(
            "Failed to parse OBJ mesh {}: {}",
            mesh_path.display(),
            err
        ))
    })?;

    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    let mut vertex_count = 0usize;

    for model in models {
        for vertex in model.mesh.positions.chunks_exact(3) {
            let point = Vec3::new(vertex[0], vertex[1], vertex[2]);
            min = min.min(point);
            max = max.max(point);
            vertex_count += 1;
        }
    }

    if vertex_count == 0 {
        return Err(RenderError::DataParsingError(format!(
            "OBJ mesh {} contains no vertices",
            mesh_path.display()
        )));
    }

    Ok(MeshBounds {
        min,
        max,
        center: (min + max) * 0.5,
        vertex_count,
    })
}

/// Load bounds for a YCB object directory using the standard google_16k mesh.
pub fn load_ycb_mesh_bounds(object_dir: &Path) -> Result<MeshBounds, RenderError> {
    load_mesh_bounds(&object_dir.join(GOOGLE_16K_MESH_RELATIVE))
}

/// Generate object-centered TBP viewpoints for a YCB object directory.
pub fn generate_ycb_object_centered_viewpoints(
    object_dir: &Path,
    config: &ViewpointConfig,
    object_rotation: &ObjectRotation,
) -> Result<Vec<Transform>, RenderError> {
    let bounds = load_ycb_mesh_bounds(object_dir)?;
    Ok(generate_object_centered_viewpoints(
        config,
        bounds.center,
        object_rotation,
    ))
}

/// Generate viewpoints for a requested targeting policy.
pub fn generate_targeted_viewpoints(
    object_dir: &Path,
    config: &ViewpointConfig,
    object_rotation: &ObjectRotation,
    policy: &TargetingPolicy,
) -> Result<TargetedViewpoints, RenderError> {
    match policy {
        TargetingPolicy::Origin => Ok(TargetedViewpoints {
            policy: policy.clone(),
            target_point: Vec3::ZERO,
            mesh_bounds: None,
            viewpoints: generate_viewpoints(config),
        }),
        TargetingPolicy::MeshCenter => {
            let bounds = load_ycb_mesh_bounds(object_dir)?;
            let target_point = rotated_mesh_center(bounds.center, object_rotation);
            Ok(TargetedViewpoints {
                policy: policy.clone(),
                target_point,
                mesh_bounds: Some(bounds),
                viewpoints: generate_viewpoints_around_target(config, target_point),
            })
        }
        TargetingPolicy::ExplicitTarget(target) => {
            let target_point = Vec3::from_array(*target);
            Ok(TargetedViewpoints {
                policy: policy.clone(),
                target_point,
                mesh_bounds: None,
                viewpoints: generate_viewpoints_around_target(config, target_point),
            })
        }
    }
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
#[derive(Clone, Debug, PartialEq)]
pub struct RenderConfig {
    /// Image width in pixels (default: 64)
    pub width: u32,
    /// Image height in pixels (default: 64)
    pub height: u32,
    /// Zoom factor affecting field of view (`tbp_default`: 4.0)
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
#[derive(Clone, Debug, PartialEq)]
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
    /// TBP-compatible 64x64 RGBD patch sensor configuration.
    ///
    /// Uses TBP's 90° base-HFOV zoom formula with a 64x64 patch render. TBP's
    /// Habitat patch sensor uses zoom=10 with a separate viewfinder; the current
    /// single-sensor YCB benchmark keeps zoom=4 for centering stability.
    ///
    /// TBP ref: `missing_depthto3d_sensor2_semantic0.yaml` (zoom=10 upstream)
    pub fn tbp_default() -> Self {
        Self {
            width: 64,
            height: 64,
            zoom: 4.0,
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
    /// TBP zooms by dividing the focal length, not the angle:
    ///   `fx_norm = tan(hfov/2) / zoom`
    /// This is equivalent to `fov = 2 * atan(tan(hfov/2) / zoom)`.
    /// With hfov=90° and zoom=10, effective FOV ≈ 11.4° (not 9°).
    pub fn fov_radians(&self) -> f32 {
        let base_hfov_rad = 90.0_f32.to_radians();
        let half_tan = (base_hfov_rad / 2.0).tan() / self.zoom;
        2.0 * half_tan.atan()
    }

    /// Compute camera intrinsics for use with neocortx.
    ///
    /// Returns focal length and principal point based on resolution and FOV.
    /// Matches TBP Python: `fx = tan(hfov/2) / zoom` in normalized [-1,1] space,
    /// converted to pixel space: `fx_pixel = (width/2) / fx_normalized`.
    ///
    /// TBP ref: `transforms.py:440` `fx = np.tan(hfov[i] / 2.0) / zoom`
    pub fn intrinsics(&self) -> CameraIntrinsics {
        self.intrinsics_for_size(self.width, self.height)
    }

    /// Compute camera intrinsics for a concrete render target size.
    ///
    /// This keeps readback metadata aligned with the actual image dimensions
    /// while preserving TBP's focal-length-space zoom formula.
    pub fn intrinsics_for_size(&self, width: u32, height: u32) -> CameraIntrinsics {
        let base_hfov_rad = 90.0_f64.to_radians();
        // TBP normalized focal length: fx_norm = tan(hfov/2) / zoom
        let fx_norm = (base_hfov_rad / 2.0).tan() / self.zoom as f64;
        // Convert to pixel focal length: fx_pixel = (width/2) / fx_norm
        let fx = (width as f64 / 2.0) / fx_norm;
        let fy = fx; // Square pixels (TBP adjusts fy for aspect ratio, but we use 64x64)

        CameraIntrinsics {
            focal_length: [fx, fy],
            principal_point: [width as f64 / 2.0, height as f64 / 2.0],
            image_size: [width, height],
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

/// Cheap diagnostics derived from a rendered depth buffer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RenderHealth {
    /// Center pixel selected from camera intrinsics, clamped to image bounds.
    pub center_pixel: Option<[u32; 2]>,
    /// Raw depth at the center pixel, including far-plane/background values.
    pub center_depth: Option<f64>,
    /// Whether the center pixel has a finite positive depth before the far plane.
    pub center_foreground: bool,
    /// Number of foreground pixels in the full depth buffer.
    pub foreground_pixel_count: usize,
    /// Foreground fraction in `[0, 1]` over the declared image size.
    pub foreground_coverage: f64,
    /// Number of foreground pixels in the 5x5 window centered on `center_pixel`.
    pub center_5x5_foreground_count: usize,
    /// Foreground pixel nearest to `center_pixel`, if any foreground exists.
    pub nearest_foreground_pixel: Option<[u32; 2]>,
    /// Depth at `nearest_foreground_pixel`.
    pub nearest_foreground_depth: Option<f64>,
    /// Euclidean pixel distance from `center_pixel` to `nearest_foreground_pixel`.
    pub nearest_foreground_distance_px: Option<f64>,
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
    /// Point the camera was intended to target for this render.
    pub target_point: Vec3,
    /// Policy used to derive `target_point`.
    pub targeting_policy: TargetingPolicy,
}

pub(crate) fn semantic_3d_from_depth(
    depth: &[f64],
    width: u32,
    height: u32,
    intrinsics: &CameraIntrinsics,
    camera_transform: Transform,
    object_semantic_id: u32,
    far_plane: f64,
) -> Vec<[f64; 4]> {
    let total_pixels = (width as usize).saturating_mul(height as usize);
    let mut rows = Vec::with_capacity(total_pixels);
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) as usize;
            let Some(&pixel_depth) = depth.get(idx) else {
                rows.push([0.0, 0.0, 0.0, 0.0]);
                continue;
            };
            let Some(world) = pixel_surface_point_world_from_parts(
                pixel_depth,
                [x, y],
                intrinsics,
                camera_transform,
                far_plane,
            ) else {
                rows.push([0.0, 0.0, 0.0, 0.0]);
                continue;
            };
            rows.push([world[0], world[1], world[2], object_semantic_id as f64]);
        }
    }
    rows
}

fn pixel_surface_point_world_from_parts(
    depth: f64,
    pixel: [u32; 2],
    intrinsics: &CameraIntrinsics,
    camera_transform: Transform,
    far_plane: f64,
) -> Option<[f64; 3]> {
    if !RenderOutput::is_foreground_depth(depth, far_plane) {
        return None;
    }

    let fx = intrinsics.focal_length[0];
    let fy = intrinsics.focal_length[1];
    if !fx.is_finite() || !fy.is_finite() || fx.abs() <= f64::EPSILON || fy.abs() <= f64::EPSILON {
        return None;
    }

    let [x, y] = pixel;
    let camera_x = (x as f64 - intrinsics.principal_point[0]) / fx * depth;
    let camera_y = -((y as f64 - intrinsics.principal_point[1]) / fy * depth);
    let point = Vec3::new(camera_x as f32, camera_y as f32, -depth as f32);
    let world = camera_transform.translation + camera_transform.rotation * point;
    Some([world.x as f64, world.y as f64, world.z as f64])
}

impl RenderOutput {
    /// Default far plane used by TBP render helpers.
    pub const TBP_FAR_PLANE_METERS: f64 = 10.0;

    /// Attach the render target metadata used to generate this camera transform.
    pub fn with_targeting(mut self, target_point: Vec3, targeting_policy: TargetingPolicy) -> Self {
        self.target_point = target_point;
        self.targeting_policy = targeting_policy;
        self
    }

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

    /// Pixel nearest the camera principal point, clamped to image bounds.
    pub fn center_pixel(&self) -> Option<[u32; 2]> {
        if self.width == 0 || self.height == 0 {
            return None;
        }

        let x = self.intrinsics.principal_point[0]
            .round()
            .clamp(0.0, (self.width - 1) as f64) as u32;
        let y = self.intrinsics.principal_point[1]
            .round()
            .clamp(0.0, (self.height - 1) as f64) as u32;
        Some([x, y])
    }

    /// Raw center-pixel depth, including far-plane/background values.
    pub fn center_pixel_raw_depth(&self) -> Option<f64> {
        let [x, y] = self.center_pixel()?;
        self.get_depth(x, y)
    }

    /// Center-pixel object depth using the TBP default far plane.
    pub fn center_pixel_depth(&self) -> Option<f64> {
        self.center_pixel_depth_with_far_plane(Self::TBP_FAR_PLANE_METERS)
    }

    /// Center-pixel object depth using a caller-provided far plane.
    pub fn center_pixel_depth_with_far_plane(&self, far_plane: f64) -> Option<f64> {
        self.center_pixel_raw_depth()
            .filter(|depth| Self::is_foreground_depth(*depth, far_plane))
    }

    /// Whether a depth value should be treated as foreground/object surface.
    pub fn is_foreground_depth(depth: f64, far_plane: f64) -> bool {
        depth.is_finite() && depth > 0.0 && far_plane.is_finite() && depth < far_plane * 0.999
    }

    /// Compute render-health diagnostics using the TBP default far plane.
    pub fn health(&self) -> RenderHealth {
        self.health_with_far_plane(Self::TBP_FAR_PLANE_METERS)
    }

    /// Compute render-health diagnostics using a caller-provided far plane.
    pub fn health_with_far_plane(&self, far_plane: f64) -> RenderHealth {
        let center_pixel = self.center_pixel();
        let center_depth = self.center_pixel_raw_depth();
        let center_foreground = center_depth
            .map(|depth| Self::is_foreground_depth(depth, far_plane))
            .unwrap_or(false);

        let total_pixels = (self.width as usize).saturating_mul(self.height as usize);
        let mut foreground_pixel_count = 0usize;
        let mut center_5x5_foreground_count = 0usize;
        let mut nearest_foreground_pixel = None;
        let mut nearest_foreground_depth = None;
        let mut nearest_foreground_distance_px = None;

        for y in 0..self.height {
            for x in 0..self.width {
                let Some(depth) = self.get_depth(x, y) else {
                    continue;
                };
                if !Self::is_foreground_depth(depth, far_plane) {
                    continue;
                }

                foreground_pixel_count += 1;

                if let Some([cx, cy]) = center_pixel {
                    let dx = x as i64 - cx as i64;
                    let dy = y as i64 - cy as i64;

                    if dx.abs() <= 2 && dy.abs() <= 2 {
                        center_5x5_foreground_count += 1;
                    }

                    let distance = ((dx * dx + dy * dy) as f64).sqrt();
                    if nearest_foreground_distance_px
                        .map(|current| distance < current)
                        .unwrap_or(true)
                    {
                        nearest_foreground_pixel = Some([x, y]);
                        nearest_foreground_depth = Some(depth);
                        nearest_foreground_distance_px = Some(distance);
                    }
                }
            }
        }

        RenderHealth {
            center_pixel,
            center_depth,
            center_foreground,
            foreground_pixel_count,
            foreground_coverage: if total_pixels > 0 {
                foreground_pixel_count as f64 / total_pixels as f64
            } else {
                0.0
            },
            center_5x5_foreground_count,
            nearest_foreground_pixel,
            nearest_foreground_depth,
            nearest_foreground_distance_px,
        }
    }

    /// Transform a point from Bevy camera-local coordinates into world space.
    pub fn camera_to_world_point(&self, camera_point: [f64; 3]) -> [f64; 3] {
        let point = Vec3::new(
            camera_point[0] as f32,
            camera_point[1] as f32,
            camera_point[2] as f32,
        );
        let rotated = self.camera_transform.rotation * point;
        let translated = self.camera_transform.translation + rotated;
        [
            translated.x as f64,
            translated.y as f64,
            translated.z as f64,
        ]
    }

    /// Transform a point from world space into Bevy camera-local coordinates.
    pub fn world_to_camera_point(&self, world_point: [f64; 3]) -> [f64; 3] {
        let point = Vec3::new(
            world_point[0] as f32,
            world_point[1] as f32,
            world_point[2] as f32,
        );
        let relative = point - self.camera_transform.translation;
        let camera_point = self.camera_transform.rotation.inverse() * relative;
        [
            camera_point.x as f64,
            camera_point.y as f64,
            camera_point.z as f64,
        ]
    }

    /// Surface point at the center pixel using the TBP default far plane.
    pub fn center_surface_point_world(&self) -> Option<[f64; 3]> {
        self.center_surface_point_world_with_far_plane(Self::TBP_FAR_PLANE_METERS)
    }

    /// Surface point at the center pixel using a caller-provided far plane.
    pub fn center_surface_point_world_with_far_plane(&self, far_plane: f64) -> Option<[f64; 3]> {
        let [x, y] = self.center_pixel()?;
        self.pixel_surface_point_world_with_far_plane([x, y], far_plane)
    }

    /// Surface point at `pixel` using the TBP default far plane.
    pub fn pixel_surface_point_world(&self, pixel: [u32; 2]) -> Option<[f64; 3]> {
        self.pixel_surface_point_world_with_far_plane(pixel, Self::TBP_FAR_PLANE_METERS)
    }

    /// Surface point at `pixel` using a caller-provided far plane.
    ///
    /// Pixel coordinates follow image convention (`x` right, `y` down). The
    /// returned point is in world space. Internally this maps to Bevy's camera
    /// frame (`+X` right, `+Y` up, `-Z` forward).
    pub fn pixel_surface_point_world_with_far_plane(
        &self,
        pixel: [u32; 2],
        far_plane: f64,
    ) -> Option<[f64; 3]> {
        let [x, y] = pixel;
        let depth = self.get_depth(x, y)?;
        pixel_surface_point_world_from_parts(
            depth,
            pixel,
            &self.intrinsics,
            self.camera_transform,
            far_plane,
        )
    }

    /// Build TBP-style `semantic_3d` rows using the TBP default far plane.
    ///
    /// The returned vector is row-major with one `[x, y, z, semantic_id]` row
    /// per pixel. Foreground pixels are unprojected into world space and use
    /// `object_semantic_id`; background/far pixels are `[0, 0, 0, 0]`.
    pub fn semantic_3d(&self, object_semantic_id: u32) -> Vec<[f64; 4]> {
        self.semantic_3d_with_far_plane(object_semantic_id, Self::TBP_FAR_PLANE_METERS)
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

/// Errors that can occur during rendering and file operations.
#[derive(Debug, Clone)]
pub enum RenderError {
    /// Object mesh file not found
    MeshNotFound(String),
    /// Object texture file not found
    TextureNotFound(String),
    /// Generic file not found error
    FileNotFound { path: String, reason: String },
    /// File write failed
    FileWriteFailed { path: String, reason: String },
    /// Directory creation failed
    DirectoryCreationFailed { path: String, reason: String },
    /// Bevy rendering failed
    RenderFailed(String),
    /// Invalid configuration
    InvalidConfig(String),
    /// Invalid input parameters
    InvalidInput(String),
    /// JSON serialization/deserialization error
    SerializationError(String),
    /// Binary data parsing error
    DataParsingError(String),
    /// Render timeout
    RenderTimeout { duration_secs: u64 },
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RenderError::MeshNotFound(path) => write!(f, "Mesh not found: {}", path),
            RenderError::TextureNotFound(path) => write!(f, "Texture not found: {}", path),
            RenderError::FileNotFound { path, reason } => {
                write!(f, "File not found at {}: {}", path, reason)
            }
            RenderError::FileWriteFailed { path, reason } => {
                write!(f, "Failed to write file {}: {}", path, reason)
            }
            RenderError::DirectoryCreationFailed { path, reason } => {
                write!(f, "Failed to create directory {}: {}", path, reason)
            }
            RenderError::RenderFailed(msg) => write!(f, "Render failed: {}", msg),
            RenderError::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
            RenderError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            RenderError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            RenderError::DataParsingError(msg) => write!(f, "Data parsing error: {}", msg),
            RenderError::RenderTimeout { duration_secs } => {
                write!(f, "Render timeout after {} seconds", duration_secs)
            }
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

/// Render a YCB object and attach the target metadata used for the camera pose.
///
/// This is useful when callers generate camera transforms with
/// [`generate_targeted_viewpoints`] and need the live render output to carry the
/// exact per-render pivot point for downstream pose compensation.
pub fn render_to_buffer_with_target(
    object_dir: &Path,
    camera_transform: &Transform,
    object_rotation: &ObjectRotation,
    config: &RenderConfig,
    target_point: Vec3,
    targeting_policy: TargetingPolicy,
) -> Result<RenderOutput, RenderError> {
    render_to_buffer(object_dir, camera_transform, object_rotation, config)
        .map(|output| output.with_targeting(target_point, targeting_policy))
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

/// Structured center-hit validation report for one object.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CenterHitValidationReport {
    /// Object identifier used in logs/manifests.
    pub object_id: String,
    /// Object directory rendered.
    pub object_dir: String,
    /// Targeting policy used for all rotations.
    pub target_policy: TargetingPolicy,
    /// Per-rotation center-hit results.
    pub rotations: Vec<CenterHitRotationReport>,
}

impl CenterHitValidationReport {
    /// True when every rotation has at least one center-foreground hit.
    pub fn is_valid(&self) -> bool {
        self.rotations
            .iter()
            .all(|rotation| rotation.center_hits > 0)
    }

    /// Rotation indices with zero center-foreground hits.
    pub fn zero_hit_rotations(&self) -> Vec<usize> {
        self.rotations
            .iter()
            .filter(|rotation| rotation.center_hits == 0)
            .map(|rotation| rotation.rotation_index)
            .collect()
    }
}

/// Center-hit validation result for a single object rotation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CenterHitRotationReport {
    pub rotation_index: usize,
    pub rotation_euler: [f64; 3],
    pub target_point: [f32; 3],
    pub mesh_bounds: Option<MeshBoundsMetadata>,
    pub total_viewpoints: usize,
    pub center_hits: usize,
    pub center_misses: usize,
    pub misses: Vec<CenterHitMiss>,
}

/// Serializable mesh-bounds metadata for reports and manifests.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeshBoundsMetadata {
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub center: [f32; 3],
    pub vertex_count: usize,
}

impl From<MeshBounds> for MeshBoundsMetadata {
    fn from(bounds: MeshBounds) -> Self {
        Self {
            min: bounds.min.to_array(),
            max: bounds.max.to_array(),
            center: bounds.center.to_array(),
            vertex_count: bounds.vertex_count,
        }
    }
}

/// Center-hit miss with enough metadata to reproduce the bad viewpoint.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CenterHitMiss {
    pub viewpoint_index: usize,
    pub camera_position: [f32; 3],
    pub camera_rotation_xyzw: [f32; 4],
    pub health: RenderHealth,
}

/// Validate that each rotation has at least one viewpoint whose center pixel
/// lands on foreground before the render far plane.
pub fn validate_center_hits(
    object_id: impl Into<String>,
    object_dir: &Path,
    viewpoint_config: &ViewpointConfig,
    rotations: &[ObjectRotation],
    render_config: &RenderConfig,
    target_policy: &TargetingPolicy,
) -> Result<CenterHitValidationReport, RenderError> {
    let object_id = object_id.into();
    let mut rotation_reports = Vec::with_capacity(rotations.len());

    for (rotation_index, rotation) in rotations.iter().enumerate() {
        let targeted =
            generate_targeted_viewpoints(object_dir, viewpoint_config, rotation, target_policy)?;
        let requests: Vec<batch::BatchRenderRequest> = targeted
            .viewpoints
            .iter()
            .map(|viewpoint| batch::BatchRenderRequest {
                object_dir: PathBuf::from(object_dir),
                viewpoint: *viewpoint,
                object_rotation: rotation.clone(),
                render_config: render_config.clone(),
                target_point: targeted.target_point,
                targeting_policy: target_policy.clone(),
            })
            .collect();

        let outputs = render_batch(requests, &batch::BatchRenderConfig::default())
            .map_err(|error| RenderError::RenderFailed(error.to_string()))?;

        let mut center_hits = 0usize;
        let mut misses = Vec::new();
        for (viewpoint_index, output) in outputs.iter().enumerate() {
            if output.status != batch::RenderStatus::Success {
                return Err(RenderError::RenderFailed(format!(
                    "Render failed for {} rotation {} viewpoint {}: {:?}",
                    object_id, rotation_index, viewpoint_index, output.error_message
                )));
            }

            if output.health.center_foreground {
                center_hits += 1;
            } else {
                let t = output.request.viewpoint.translation;
                let q = output.request.viewpoint.rotation;
                misses.push(CenterHitMiss {
                    viewpoint_index,
                    camera_position: [t.x, t.y, t.z],
                    camera_rotation_xyzw: [q.x, q.y, q.z, q.w],
                    health: output.health.clone(),
                });
            }
        }

        rotation_reports.push(CenterHitRotationReport {
            rotation_index,
            rotation_euler: [rotation.pitch, rotation.yaw, rotation.roll],
            target_point: targeted.target_point.to_array(),
            mesh_bounds: targeted.mesh_bounds.map(MeshBoundsMetadata::from),
            total_viewpoints: outputs.len(),
            center_hits,
            center_misses: outputs.len().saturating_sub(center_hits),
            misses,
        });
    }

    Ok(CenterHitValidationReport {
        object_id,
        object_dir: object_dir.display().to_string(),
        target_policy: target_policy.clone(),
        rotations: rotation_reports,
    })
}

/// Render with model caching support for efficient multi-viewpoint rendering.
///
/// This function tracks which models have been loaded and provides performance
/// insights. It still spins up a fresh headless `App` per call. For workloads
/// that render many frames against the same object/config, prefer
/// `RenderSession` (homogeneous batches per episode) or `PersistentRenderer`
/// (one frame per call, scene held loaded across calls — built for surface-
/// policy feedback loops).
///
/// # Arguments
/// * `object_dir` - Path to YCB object directory
/// * `camera_transform` - Camera position and orientation
/// * `object_rotation` - Rotation to apply to the object
/// * `config` - Render configuration
/// * `cache` - Model cache to track loaded assets
///
/// # Returns
/// RenderOutput with rendered RGBA and depth data
///
/// # Example
/// ```ignore
/// use bevy_sensor::{render_to_buffer_cached, cache::ModelCache, RenderConfig, ObjectRotation};
/// use std::path::PathBuf;
///
/// let mut cache = ModelCache::new();
/// let object_dir = PathBuf::from("/tmp/ycb/003_cracker_box");
/// let config = RenderConfig::tbp_default();
/// let viewpoints = bevy_sensor::generate_viewpoints(&ViewpointConfig::default());
///
/// // First render: loads from disk and caches
/// let output1 = render_to_buffer_cached(
///     &object_dir,
///     &viewpoints[0],
///     &ObjectRotation::identity(),
///     &config,
///     &mut cache,
/// )?;
///
/// // Subsequent renders: tracks in cache
/// for viewpoint in &viewpoints[1..] {
///     let output = render_to_buffer_cached(
///         &object_dir,
///         viewpoint,
///         &ObjectRotation::identity(),
///         &config,
///         &mut cache,
///     )?;
/// }
/// ```
///
/// # Note
/// This function uses the same rendering engine as `render_to_buffer()`. The current
/// batch API preserves ordering and output structure but does not yet reuse a live
/// Bevy renderer across calls.
///
/// ```ignore
/// use bevy_sensor::{
///     render_batch, batch::BatchRenderRequest, BatchRenderConfig, RenderConfig,
///     ObjectRotation, TargetingPolicy, Vec3,
/// };
///
/// let requests: Vec<_> = viewpoints.iter().map(|vp| {
///     BatchRenderRequest {
///         object_dir: object_dir.clone(),
///         viewpoint: *vp,
///         object_rotation: ObjectRotation::identity(),
///         render_config: RenderConfig::tbp_default(),
///         target_point: Vec3::ZERO,
///         targeting_policy: TargetingPolicy::Origin,
///     }
/// }).collect();
///
/// let outputs = render_batch(requests, &BatchRenderConfig::default())?;
/// ```
pub fn render_to_buffer_cached(
    object_dir: &Path,
    camera_transform: &Transform,
    object_rotation: &ObjectRotation,
    config: &RenderConfig,
    cache: &mut cache::ModelCache,
) -> Result<RenderOutput, RenderError> {
    let mesh_path = object_dir.join("google_16k/textured.obj");
    let texture_path = object_dir.join("google_16k/texture_map.png");

    // Track in cache
    cache.cache_scene(mesh_path.clone());
    cache.cache_texture(texture_path.clone());

    // Render using standard pipeline
    render::render_headless(object_dir, camera_transform, object_rotation, config)
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

// Re-export batch types for convenient API access
pub use batch::{
    BatchRenderConfig, BatchRenderError, BatchRenderOutput, BatchRenderRequest, BatchRenderer,
    BatchState, RenderStatus,
};

/// Persistent batch render session. See the module docs in `render::RenderSession`
/// for lifetime, thread-affinity, and config-invariance guarantees.
pub use render::RenderSession;

/// Per-step persistent renderer for feedback loops. See the module docs in
/// `render::PersistentRenderer` for lifetime, thread-affinity, and
/// object/config-invariance guarantees. Built for the surface-policy use case
/// in neocortx where a fixed object is rendered from a moving camera many
/// times per episode (issue #65).
pub use render::PersistentRenderer;

/// Create a new batch renderer helper for multi-viewpoint workflows.
///
/// The current implementation stores queued requests and executes them sequentially via
/// `render_to_buffer()`. It does not yet keep a persistent Bevy app alive across renders.
///
/// # Arguments
/// * `config` - Batch rendering configuration
///
/// # Returns
/// A BatchRenderer instance ready to queue render requests
///
/// # Example
/// ```ignore
/// use bevy_sensor::{create_batch_renderer, queue_render_request, render_next_in_batch, BatchRenderConfig};
///
/// let mut renderer = create_batch_renderer(&BatchRenderConfig::default())?;
/// ```
pub fn create_batch_renderer(config: &BatchRenderConfig) -> Result<BatchRenderer, RenderError> {
    Ok(BatchRenderer::new(config.clone()))
}

/// Queue a render request for batch processing.
///
/// Adds a render request to the batch queue. Requests are processed in order
/// when you call render_next_in_batch().
///
/// # Arguments
/// * `renderer` - The batch renderer instance
/// * `request` - The render request
///
/// # Returns
/// Ok if queued successfully, Err if queue is full
///
/// # Example
/// ```ignore
/// use bevy_sensor::{batch::BatchRenderRequest, RenderConfig, ObjectRotation, TargetingPolicy, Vec3};
/// use std::path::PathBuf;
///
/// queue_render_request(&mut renderer, BatchRenderRequest {
///     object_dir: PathBuf::from("/tmp/ycb/003_cracker_box"),
///     viewpoint: camera_transform,
///     object_rotation: ObjectRotation::identity(),
///     render_config: RenderConfig::tbp_default(),
///     target_point: Vec3::ZERO,
///     targeting_policy: TargetingPolicy::Origin,
/// })?;
/// ```
pub fn queue_render_request(
    renderer: &mut BatchRenderer,
    request: BatchRenderRequest,
) -> Result<(), RenderError> {
    renderer
        .queue_request(request)
        .map_err(|e| RenderError::RenderFailed(e.to_string()))
}

/// Process and execute the next render in the batch queue.
///
/// Executes a single queued request via `render_to_buffer()`. Returns None when the queue
/// is empty. Use this in a loop to process all queued renders in a stable order.
///
/// # Arguments
/// * `renderer` - The batch renderer instance
/// * `timeout_ms` - Timeout in milliseconds for this render
///
/// # Returns
/// Some(output) if a render completed, None if queue is empty
///
/// # Example
/// ```ignore
/// loop {
///     match render_next_in_batch(&mut renderer, 500)? {
///         Some(output) => println!("Render complete: {:?}", output.status),
///         None => break, // All renders done
///     }
/// }
/// ```
pub fn render_next_in_batch(
    renderer: &mut BatchRenderer,
    _timeout_ms: u32,
) -> Result<Option<BatchRenderOutput>, RenderError> {
    if let Some(request) = renderer.pending_requests.pop_front() {
        let output = render_to_buffer(
            &request.object_dir,
            &request.viewpoint,
            &request.object_rotation,
            &request.render_config,
        )?;
        let batch_output = BatchRenderOutput::from_render_output(request, output);
        renderer.completed_results.push(batch_output.clone());
        renderer.renders_processed += 1;
        Ok(Some(batch_output))
    } else {
        Ok(None)
    }
}

/// Render multiple requests in batch (convenience function).
///
/// Queues all requests and executes them in batch, returning all results.
/// Simpler than manage queue + loop for one-off batches.
///
/// # Arguments
/// * `requests` - Vector of render requests
/// * `config` - Batch rendering configuration
///
/// # Returns
/// Vector of BatchRenderOutput results in same order as input
///
/// # Example
/// ```ignore
/// use bevy_sensor::{render_batch, batch::BatchRenderRequest, BatchRenderConfig};
///
/// let results = render_batch(requests, &BatchRenderConfig::default())?;
/// ```
pub fn render_batch(
    requests: Vec<BatchRenderRequest>,
    config: &BatchRenderConfig,
) -> Result<Vec<BatchRenderOutput>, RenderError> {
    if requests.is_empty() {
        return Ok(Vec::new());
    }

    if requests.len() > 1 && requests_share_batch_context(&requests) {
        let first_request = requests[0].clone();
        let viewpoints: Vec<Transform> = requests.iter().map(|request| request.viewpoint).collect();
        let outputs = render::render_headless_sequence(
            &first_request.object_dir,
            &viewpoints,
            &first_request.object_rotation,
            &first_request.render_config,
        )?;

        return Ok(requests
            .into_iter()
            .zip(outputs)
            .map(|(request, output)| BatchRenderOutput::from_render_output(request, output))
            .collect());
    }

    let mut renderer = create_batch_renderer(config)?;

    // Queue all requests
    for request in requests {
        queue_render_request(&mut renderer, request)?;
    }

    // Execute all and collect results
    let mut results = Vec::new();
    while let Some(output) = render_next_in_batch(&mut renderer, config.frame_timeout_ms)? {
        results.push(output);
    }

    Ok(results)
}

fn requests_share_batch_context(requests: &[BatchRenderRequest]) -> bool {
    let Some(first) = requests.first() else {
        return true;
    };

    requests.iter().all(|request| {
        request.object_dir == first.object_dir
            && request.object_rotation == first.object_rotation
            && request.render_config == first.render_config
    })
}

// Re-export bevy types that consumers will need
pub use bevy::prelude::{Quat, Transform, Vec3};

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_vec3_close(actual: Vec3, expected: Vec3) {
        assert!(
            (actual - expected).length() < 1e-5,
            "expected {:?}, got {:?}",
            expected,
            actual
        );
    }

    fn assert_point_close(actual: [f64; 3], expected: [f64; 3]) {
        for axis in 0..3 {
            assert!(
                (actual[axis] - expected[axis]).abs() < 1e-5,
                "axis {} expected {:?}, got {:?}",
                axis,
                expected,
                actual
            );
        }
    }

    fn render_output_for_depth(
        width: u32,
        height: u32,
        depth: Vec<f64>,
        intrinsics: CameraIntrinsics,
        camera_transform: Transform,
    ) -> RenderOutput {
        RenderOutput {
            rgba: vec![0u8; (width * height * 4) as usize],
            depth,
            width,
            height,
            intrinsics,
            camera_transform,
            object_rotation: ObjectRotation::identity(),
            target_point: Vec3::ZERO,
            targeting_policy: TargetingPolicy::Origin,
        }
    }

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
    fn test_requests_share_batch_context_for_homogeneous_batch() {
        let config = RenderConfig::tbp_default();
        let request = BatchRenderRequest {
            object_dir: "/tmp/ycb/003_cracker_box".into(),
            viewpoint: Transform::IDENTITY,
            object_rotation: ObjectRotation::identity(),
            render_config: config.clone(),
            target_point: Vec3::ZERO,
            targeting_policy: TargetingPolicy::Origin,
        };

        assert!(requests_share_batch_context(&[
            request.clone(),
            BatchRenderRequest {
                viewpoint: Transform::from_xyz(1.0, 0.0, 0.0),
                ..request
            },
        ]));
    }

    #[test]
    fn test_requests_share_batch_context_rejects_mixed_objects() {
        let config = RenderConfig::tbp_default();
        let request = BatchRenderRequest {
            object_dir: "/tmp/ycb/003_cracker_box".into(),
            viewpoint: Transform::IDENTITY,
            object_rotation: ObjectRotation::identity(),
            render_config: config.clone(),
            target_point: Vec3::ZERO,
            targeting_policy: TargetingPolicy::Origin,
        };

        assert!(!requests_share_batch_context(&[
            request.clone(),
            BatchRenderRequest {
                object_dir: "/tmp/ycb/005_tomato_soup_can".into(),
                ..request
            },
        ]));
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
    fn test_generate_viewpoints_around_target_preserves_orbit() {
        let config = ViewpointConfig {
            radius: 2.0,
            yaw_count: 4,
            pitch_angles_deg: vec![0.0],
        };
        let target = Vec3::new(1.0, -0.5, 0.25);
        let viewpoints = generate_viewpoints_around_target(&config, target);

        assert_eq!(viewpoints.len(), 4);
        for (i, transform) in viewpoints.iter().enumerate() {
            let offset = transform.translation - target;
            assert!(
                (offset.length() - config.radius).abs() < 1e-5,
                "viewpoint {} has radius {}, expected {}",
                i,
                offset.length(),
                config.radius
            );

            let forward = transform.forward();
            let to_target = (target - transform.translation).normalize();
            assert!(
                forward.dot(to_target) > 0.99,
                "viewpoint {} is not looking at target",
                i
            );
        }
    }

    #[test]
    fn test_generate_viewpoints_keeps_origin_targeting() {
        let config = ViewpointConfig {
            radius: 1.0,
            yaw_count: 1,
            pitch_angles_deg: vec![0.0],
        };

        let origin_view = generate_viewpoints(&config)[0];
        let explicit_origin_view = generate_viewpoints_around_target(&config, Vec3::ZERO)[0];

        assert_vec3_close(origin_view.translation, explicit_origin_view.translation);
        let forward = origin_view.forward();
        let to_origin = (Vec3::ZERO - origin_view.translation).normalize();
        assert!(forward.dot(to_origin) > 0.99);
    }

    #[test]
    fn test_object_centered_viewpoints_apply_yaw_rotation_to_target() {
        let config = ViewpointConfig {
            radius: 1.0,
            yaw_count: 1,
            pitch_angles_deg: vec![0.0],
        };
        let mesh_center = Vec3::new(0.25, 0.0, 0.0);
        let rotation = ObjectRotation::new(0.0, 90.0, 0.0);

        let target = rotated_mesh_center(mesh_center, &rotation);
        assert!(target.distance(mesh_center) > 0.1);

        let origin_view = generate_viewpoints(&config)[0];
        let centered_view = generate_object_centered_viewpoints(&config, mesh_center, &rotation)[0];

        assert_vec3_close(centered_view.translation, origin_view.translation + target);
        let forward = centered_view.forward();
        let to_target = (target - centered_view.translation).normalize();
        assert!(forward.dot(to_target) > 0.99);
    }

    #[test]
    fn test_load_ycb_mesh_bounds_from_standard_obj_path() {
        let dir = tempfile::tempdir().unwrap();
        let mesh_dir = dir.path().join("google_16k");
        std::fs::create_dir_all(&mesh_dir).unwrap();
        std::fs::write(
            mesh_dir.join("textured.obj"),
            "v -1.0 -2.0 -3.0\nv 3.0 4.0 5.0\nv 1.0 0.0 2.0\nf 1 2 3\n",
        )
        .unwrap();

        let bounds = load_ycb_mesh_bounds(dir.path()).unwrap();

        assert_eq!(bounds.vertex_count, 3);
        assert_vec3_close(bounds.min, Vec3::new(-1.0, -2.0, -3.0));
        assert_vec3_close(bounds.max, Vec3::new(3.0, 4.0, 5.0));
        assert_vec3_close(bounds.center, Vec3::new(1.0, 1.0, 1.0));
        assert_vec3_close(bounds.extents(), Vec3::new(4.0, 6.0, 8.0));
    }

    #[test]
    fn test_targeting_policy_serializes_stable_label() {
        assert_eq!(TargetingPolicy::Origin.label(), "origin");
        assert_eq!(TargetingPolicy::MeshCenter.label(), "mesh-center");

        let json = serde_json::to_string(&TargetingPolicy::MeshCenter).unwrap();
        assert!(json.contains("mesh_center"));
        let loaded: TargetingPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded, TargetingPolicy::MeshCenter);
    }

    #[test]
    fn test_render_output_with_targeting_overrides_origin_default() {
        let target_point = Vec3::new(0.1, 0.2, -0.3);
        let output = render_output_for_depth(
            1,
            1,
            vec![1.0],
            RenderConfig::tbp_default().intrinsics(),
            Transform::IDENTITY,
        )
        .with_targeting(target_point, TargetingPolicy::MeshCenter);

        assert_eq!(output.target_point, target_point);
        assert_eq!(output.targeting_policy, TargetingPolicy::MeshCenter);
    }

    #[test]
    fn test_center_hit_validation_report_detects_zero_hit_rotation() {
        let report = CenterHitValidationReport {
            object_id: "test_object".to_string(),
            object_dir: "/tmp/ycb/test_object".to_string(),
            target_policy: TargetingPolicy::MeshCenter,
            rotations: vec![
                CenterHitRotationReport {
                    rotation_index: 0,
                    rotation_euler: [0.0, 0.0, 0.0],
                    target_point: [0.0, 0.0, 0.0],
                    mesh_bounds: None,
                    total_viewpoints: 24,
                    center_hits: 1,
                    center_misses: 23,
                    misses: Vec::new(),
                },
                CenterHitRotationReport {
                    rotation_index: 1,
                    rotation_euler: [0.0, 90.0, 0.0],
                    target_point: [0.1, 0.0, 0.0],
                    mesh_bounds: None,
                    total_viewpoints: 24,
                    center_hits: 0,
                    center_misses: 24,
                    misses: Vec::new(),
                },
            ],
        };

        assert!(!report.is_valid());
        assert_eq!(report.zero_hit_rotations(), vec![1]);
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
    fn test_ycb_tbp_standard_objects() {
        assert_eq!(crate::ycb::TBP_STANDARD_OBJECTS.len(), 10);
        assert!(crate::ycb::TBP_STANDARD_OBJECTS.contains(&"025_mug"));
    }

    #[test]
    fn test_ycb_tbp_similar_objects() {
        assert_eq!(crate::ycb::TBP_SIMILAR_OBJECTS.len(), 10);
        assert!(crate::ycb::TBP_SIMILAR_OBJECTS.contains(&"003_cracker_box"));
    }

    #[test]
    fn test_ycb_object_mesh_path() {
        let path = crate::ycb::object_mesh_path("/tmp/ycb", "003_cracker_box");
        assert_eq!(
            path,
            std::path::Path::new("/tmp/ycb")
                .join("003_cracker_box")
                .join("google_16k")
                .join("textured.obj")
        );
    }

    #[test]
    fn test_ycb_object_texture_path() {
        let path = crate::ycb::object_texture_path("/tmp/ycb", "003_cracker_box");
        assert_eq!(
            path,
            std::path::Path::new("/tmp/ycb")
                .join("003_cracker_box")
                .join("google_16k")
                .join("texture_map.png")
        );
    }

    // =========================================================================
    // Headless Rendering API Tests
    // =========================================================================

    #[test]
    fn test_render_config_tbp_default() {
        let config = RenderConfig::tbp_default();
        // TBP spec: 64x64 patch sensor resolution
        assert_eq!(config.width, 64);
        assert_eq!(config.height, 64);
        // Zoom is a divisor in the FOV formula — must be positive
        assert!(config.zoom > 0.0);
        // Clipping planes must form a valid, positive range
        assert!(config.near_plane > 0.0);
        assert!(config.far_plane > config.near_plane);
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
        // FOV must be a valid positive angle strictly less than π for any
        // positive zoom — no cameras with ≥180° FOV.
        assert!(fov > 0.0);
        assert!(fov < PI);

        // Zoom in should reduce FOV (tighter view).
        let zoomed = RenderConfig {
            zoom: config.zoom * 2.0,
            ..config
        };
        assert!(zoomed.fov_radians() < fov);
    }

    #[test]
    fn test_render_config_intrinsics() {
        let config = RenderConfig::tbp_default();
        let intrinsics = config.intrinsics();

        // Image size matches config; principal point at image center.
        assert_eq!(intrinsics.image_size, [config.width, config.height]);
        assert_eq!(
            intrinsics.principal_point,
            [config.width as f64 / 2.0, config.height as f64 / 2.0]
        );
        // Square pixels: fx == fy.
        assert_eq!(intrinsics.focal_length[0], intrinsics.focal_length[1]);
        assert!(intrinsics.focal_length[0] > 0.0);
    }

    #[test]
    fn test_render_config_intrinsics_for_size_uses_tbp_zoom_formula() {
        let config = RenderConfig {
            width: 64,
            height: 64,
            zoom: 4.0,
            ..RenderConfig::tbp_default()
        };

        let intrinsics = config.intrinsics_for_size(64, 64);

        // TBP formula for 90° base HFOV:
        // fx = (width / 2) / (tan(45°) / zoom) = (width / 2) * zoom.
        assert!((intrinsics.focal_length[0] - 128.0).abs() < 1e-9);
        assert!((intrinsics.focal_length[1] - 128.0).abs() < 1e-9);
        assert_ne!(intrinsics.focal_length[0], 64.0 * config.zoom as f64);
        assert_eq!(intrinsics.principal_point, [32.0, 32.0]);
        assert_eq!(intrinsics.image_size, [64, 64]);
    }

    #[test]
    fn test_render_config_intrinsics_for_size_tracks_actual_readback_size() {
        let config = RenderConfig {
            width: 64,
            height: 64,
            zoom: 4.0,
            ..RenderConfig::tbp_default()
        };

        let intrinsics = config.intrinsics_for_size(128, 96);

        assert!((intrinsics.focal_length[0] - 256.0).abs() < 1e-9);
        assert!((intrinsics.focal_length[1] - 256.0).abs() < 1e-9);
        assert_eq!(intrinsics.principal_point, [64.0, 48.0]);
        assert_eq!(intrinsics.image_size, [128, 96]);
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
            target_point: Vec3::ZERO,
            targeting_policy: TargetingPolicy::Origin,
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
            target_point: Vec3::ZERO,
            targeting_policy: TargetingPolicy::Origin,
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
            target_point: Vec3::ZERO,
            targeting_policy: TargetingPolicy::Origin,
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
            target_point: Vec3::ZERO,
            targeting_policy: TargetingPolicy::Origin,
        };

        let depth_image = output.to_depth_image();
        assert_eq!(depth_image.len(), 2);
        assert_eq!(depth_image[0], vec![1.0, 2.0]);
        assert_eq!(depth_image[1], vec![3.0, 4.0]);
    }

    #[test]
    fn test_render_output_semantic_3d_marks_foreground_and_background() {
        let output = render_output_for_depth(
            2,
            2,
            vec![0.25, 10.0, 0.5, f64::INFINITY],
            CameraIntrinsics {
                focal_length: [1.0, 1.0],
                principal_point: [0.0, 0.0],
                image_size: [2, 2],
            },
            Transform::IDENTITY,
        );

        let semantic = output.semantic_3d(42);

        assert_eq!(semantic.len(), 4);
        assert_eq!(semantic[0][3], 42.0);
        assert_eq!(semantic[1], [0.0, 0.0, 0.0, 0.0]);
        assert_eq!(semantic[2][3], 42.0);
        assert_eq!(semantic[3], [0.0, 0.0, 0.0, 0.0]);
        assert_point_close(
            [semantic[0][0], semantic[0][1], semantic[0][2]],
            [0.0, 0.0, -0.25],
        );
        assert_point_close(
            [semantic[2][0], semantic[2][1], semantic[2][2]],
            [0.0, -0.5, -0.5],
        );
    }

    #[test]
    fn test_render_output_semantic_3d_matches_pixel_surface_points() {
        let output = render_output_for_depth(
            3,
            3,
            vec![10.0, 10.0, 2.0, 10.0, 0.25, 10.0, 10.0, 10.0, 10.0],
            CameraIntrinsics {
                focal_length: [1.0, 1.0],
                principal_point: [1.0, 1.0],
                image_size: [3, 3],
            },
            Transform::IDENTITY,
        );

        let semantic = output.semantic_3d(3);
        let top_right = output
            .pixel_surface_point_world([2, 0])
            .expect("foreground point");
        let center = output
            .pixel_surface_point_world([1, 1])
            .expect("foreground point");

        assert_point_close([semantic[2][0], semantic[2][1], semantic[2][2]], top_right);
        assert_eq!(semantic[2][3], 3.0);
        assert_point_close([semantic[4][0], semantic[4][1], semantic[4][2]], center);
        assert_eq!(semantic[4][3], 3.0);
    }

    #[test]
    fn test_render_health_center_hit() {
        let mut depth = vec![10.0; 7 * 7];
        depth[3 * 7 + 3] = 0.25;
        depth[6 * 7 + 6] = 0.5;
        let output = render_output_for_depth(
            7,
            7,
            depth,
            CameraIntrinsics {
                focal_length: [10.0, 10.0],
                principal_point: [3.0, 3.0],
                image_size: [7, 7],
            },
            Transform::IDENTITY,
        );

        let health = output.health();

        assert_eq!(health.center_pixel, Some([3, 3]));
        assert_eq!(health.center_depth, Some(0.25));
        assert!(health.center_foreground);
        assert_eq!(health.foreground_pixel_count, 2);
        assert!((health.foreground_coverage - 2.0 / 49.0).abs() < 1e-12);
        assert_eq!(health.center_5x5_foreground_count, 1);
        assert_eq!(health.nearest_foreground_pixel, Some([3, 3]));
        assert_eq!(health.nearest_foreground_depth, Some(0.25));
        assert_eq!(health.nearest_foreground_distance_px, Some(0.0));
    }

    #[test]
    fn test_render_health_far_center_uses_nearest_foreground() {
        let mut depth = vec![10.0; 7 * 7];
        depth[3 * 7 + 1] = 0.5;
        let output = render_output_for_depth(
            7,
            7,
            depth,
            CameraIntrinsics {
                focal_length: [10.0, 10.0],
                principal_point: [3.0, 3.0],
                image_size: [7, 7],
            },
            Transform::IDENTITY,
        );

        let health = output.health();

        assert_eq!(health.center_pixel, Some([3, 3]));
        assert_eq!(health.center_depth, Some(10.0));
        assert!(!health.center_foreground);
        assert_eq!(health.foreground_pixel_count, 1);
        assert_eq!(health.center_5x5_foreground_count, 1);
        assert_eq!(health.nearest_foreground_pixel, Some([1, 3]));
        assert_eq!(health.nearest_foreground_depth, Some(0.5));
        assert_eq!(health.nearest_foreground_distance_px, Some(2.0));
    }

    #[test]
    fn test_center_surface_point_world_uses_bevy_camera_forward() {
        let mut depth = vec![10.0; 3 * 3];
        depth[3 + 1] = 0.25;
        let output = render_output_for_depth(
            3,
            3,
            depth,
            CameraIntrinsics {
                focal_length: [1.0, 1.0],
                principal_point: [1.0, 1.0],
                image_size: [3, 3],
            },
            Transform::IDENTITY,
        );

        assert_eq!(output.center_pixel_depth(), Some(0.25));
        assert_point_close(
            output.center_surface_point_world().expect("surface point"),
            [0.0, 0.0, -0.25],
        );
    }

    #[test]
    fn test_pixel_surface_point_world_maps_image_y_down_to_camera_y_up() {
        let mut depth = vec![10.0; 3 * 3];
        depth[2] = 2.0;
        let output = render_output_for_depth(
            3,
            3,
            depth,
            CameraIntrinsics {
                focal_length: [1.0, 1.0],
                principal_point: [1.0, 1.0],
                image_size: [3, 3],
            },
            Transform::IDENTITY,
        );

        assert_point_close(
            output
                .pixel_surface_point_world([2, 0])
                .expect("surface point"),
            [2.0, 2.0, -2.0],
        );
    }

    #[test]
    fn test_camera_world_point_helpers_roundtrip() {
        let output = render_output_for_depth(
            1,
            1,
            vec![0.25],
            CameraIntrinsics {
                focal_length: [1.0, 1.0],
                principal_point: [0.0, 0.0],
                image_size: [1, 1],
            },
            Transform::from_xyz(0.0, 0.0, 1.0).looking_at(Vec3::ZERO, Vec3::Y),
        );

        assert_point_close(
            output.center_surface_point_world().expect("surface point"),
            [0.0, 0.0, 0.75],
        );

        let world_point = [0.1, -0.2, 0.7];
        let camera_point = output.world_to_camera_point(world_point);
        assert_point_close(output.camera_to_world_point(camera_point), world_point);
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
            target_point: Vec3::ZERO,
            targeting_policy: TargetingPolicy::Origin,
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
            target_point: Vec3::ZERO,
            targeting_policy: TargetingPolicy::Origin,
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
        // The formula fov = 2·atan(tan(base_hfov/2)/zoom) has an exact
        // invariant: tan(fov/2) * zoom is constant. So doubling zoom
        // halves tan(fov/2). (This is NOT the same as halving fov itself,
        // which only holds as a small-angle approximation.)
        let base = RenderConfig {
            zoom: 2.0,
            ..RenderConfig::tbp_default()
        };
        let doubled = RenderConfig {
            zoom: 4.0,
            ..RenderConfig::tbp_default()
        };

        // Higher zoom → tighter FOV (monotonicity).
        assert!(doubled.fov_radians() < base.fov_radians());

        // Exact invariant: tan(fov/2) scales as 1/zoom.
        let base_half_tan = (base.fov_radians() / 2.0).tan();
        let doubled_half_tan = (doubled.fov_radians() / 2.0).tan();
        assert!((base_half_tan / doubled_half_tan - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_render_config_zoom_affects_intrinsics() {
        // The formula fx = (width/2)·zoom/tan(base_hfov/2) is linear in
        // zoom for fixed width/base_hfov, so fx/zoom is constant.
        let a = RenderConfig {
            zoom: 2.0,
            ..RenderConfig::tbp_default()
        };
        let b = RenderConfig {
            zoom: 4.0,
            ..RenderConfig::tbp_default()
        };

        let fx_a = a.intrinsics().focal_length[0];
        let fx_b = b.intrinsics().focal_length[0];

        // Monotonic: higher zoom → larger focal length.
        assert!(fx_b > fx_a);

        // Exact linearity: fx/zoom is constant across configs.
        assert!((fx_a / a.zoom as f64 - fx_b / b.zoom as f64).abs() < 1e-9);
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
