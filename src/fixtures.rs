//! Test fixtures for pre-rendered YCB images
//!
//! This module provides utilities for loading pre-rendered images from disk,
//! enabling testing without GPU access.
//!
//! # Usage
//!
//! ```ignore
//! use bevy_sensor::fixtures::TestFixtures;
//!
//! let fixtures = TestFixtures::load("test_fixtures/renders")?;
//!
//! // Get a specific render
//! let render = fixtures.get_render("003_cracker_box", 0, 5)?;
//! let rgb_image = render.to_rgb_image();
//! let depth_image = render.to_depth_image();
//! ```

use crate::{CameraIntrinsics, MeshBoundsMetadata, RenderHealth, RenderOutput, TargetingPolicy};
use bevy::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Error type for fixture loading
#[derive(Debug)]
pub enum FixtureError {
    /// Directory not found
    NotFound(String),
    /// Metadata file missing or invalid
    InvalidMetadata(String),
    /// Render file missing
    RenderNotFound {
        object_id: String,
        rotation: usize,
        viewpoint: usize,
    },
    /// IO error
    IoError(std::io::Error),
    /// JSON parsing error
    JsonError(serde_json::Error),
}

impl std::fmt::Display for FixtureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FixtureError::NotFound(path) => write!(f, "Fixture directory not found: {}", path),
            FixtureError::InvalidMetadata(msg) => write!(f, "Invalid metadata: {}", msg),
            FixtureError::RenderNotFound {
                object_id,
                rotation,
                viewpoint,
            } => write!(
                f,
                "Render not found: {} r{} v{}",
                object_id, rotation, viewpoint
            ),
            FixtureError::IoError(e) => write!(f, "IO error: {}", e),
            FixtureError::JsonError(e) => write!(f, "JSON error: {}", e),
        }
    }
}

impl std::error::Error for FixtureError {}

impl From<std::io::Error> for FixtureError {
    fn from(e: std::io::Error) -> Self {
        FixtureError::IoError(e)
    }
}

impl From<serde_json::Error> for FixtureError {
    fn from(e: serde_json::Error) -> Self {
        FixtureError::JsonError(e)
    }
}

/// Dataset metadata from pre-rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub version: String,
    #[serde(default)]
    pub crate_version: Option<String>,
    #[serde(default)]
    pub renderer_policy_version: Option<String>,
    pub objects: Vec<String>,
    pub viewpoints_per_rotation: usize,
    pub rotations_per_object: usize,
    pub renders_per_object: usize,
    pub resolution: [u32; 2],
    #[serde(default)]
    pub resolution_width: Option<u32>,
    #[serde(default)]
    pub resolution_height: Option<u32>,
    #[serde(default)]
    pub targeting_policy: Option<TargetingPolicy>,
    pub intrinsics: IntrinsicsMetadata,
    pub viewpoint_config: ViewpointConfigMetadata,
    pub rotations: Vec<[f32; 3]>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntrinsicsMetadata {
    pub focal_length: [f32; 2],
    pub principal_point: [f32; 2],
    pub image_size: [u32; 2],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewpointConfigMetadata {
    pub radius: f32,
    pub yaw_count: usize,
    pub pitch_angles_deg: Vec<f32>,
}

/// Metadata for a single render
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderMetadata {
    pub object_id: String,
    pub rotation_index: usize,
    pub viewpoint_index: usize,
    pub rotation_euler: [f32; 3],
    pub camera_position: [f32; 3],
    #[serde(default)]
    pub camera_rotation_xyzw: Option<[f32; 4]>,
    #[serde(default)]
    pub object_translation: Option<[f32; 3]>,
    #[serde(default)]
    pub object_scale: Option<[f32; 3]>,
    #[serde(default)]
    pub target_point: Option<[f32; 3]>,
    #[serde(default)]
    pub targeting_policy: Option<TargetingPolicy>,
    #[serde(default)]
    pub mesh_bounds: Option<MeshBoundsMetadata>,
    #[serde(default)]
    pub health: Option<RenderHealth>,
    pub rgba_file: String,
    pub depth_file: String,
}

/// Pre-rendered test fixtures loaded from disk
pub struct TestFixtures {
    /// Root directory containing fixtures
    root: PathBuf,
    /// Dataset metadata
    pub metadata: DatasetMetadata,
    /// Per-object render indices
    indices: HashMap<String, Vec<RenderMetadata>>,
}

impl TestFixtures {
    /// Load test fixtures from a directory
    ///
    /// # Arguments
    /// * `path` - Path to the fixtures directory (e.g., "test_fixtures/renders")
    ///
    /// # Returns
    /// * `Ok(TestFixtures)` if loaded successfully
    /// * `Err(FixtureError)` if loading fails
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, FixtureError> {
        let root = path.as_ref().to_path_buf();

        if !root.exists() {
            return Err(FixtureError::NotFound(root.display().to_string()));
        }

        // Load metadata
        let metadata_path = root.join("metadata.json");
        if !metadata_path.exists() {
            return Err(FixtureError::InvalidMetadata(
                "metadata.json not found".to_string(),
            ));
        }

        let metadata_json = fs::read_to_string(&metadata_path)?;
        let metadata: DatasetMetadata = serde_json::from_str(&metadata_json)?;

        // Load per-object indices
        let mut indices = HashMap::new();
        for object_id in &metadata.objects {
            let index_path = root.join(object_id).join("index.json");
            if index_path.exists() {
                let index_json = fs::read_to_string(&index_path)?;
                let renders: Vec<RenderMetadata> = serde_json::from_str(&index_json)?;
                indices.insert(object_id.clone(), renders);
            }
        }

        Ok(Self {
            root,
            metadata,
            indices,
        })
    }

    /// Check if fixtures exist at the given path
    pub fn exists<P: AsRef<Path>>(path: P) -> bool {
        let root = path.as_ref();
        root.exists() && root.join("metadata.json").exists()
    }

    /// Get list of available objects
    pub fn objects(&self) -> &[String] {
        &self.metadata.objects
    }

    /// Get number of viewpoints per rotation
    pub fn viewpoints_per_rotation(&self) -> usize {
        self.metadata.viewpoints_per_rotation
    }

    /// Get number of rotations per object
    pub fn rotations_per_object(&self) -> usize {
        self.metadata.rotations_per_object
    }

    /// Get total renders available for an object
    pub fn renders_for_object(&self, object_id: &str) -> usize {
        self.indices.get(object_id).map(|v| v.len()).unwrap_or(0)
    }

    /// Get camera intrinsics (converts from f32 metadata to f64 for TBP precision)
    pub fn intrinsics(&self) -> CameraIntrinsics {
        CameraIntrinsics {
            focal_length: [
                self.metadata.intrinsics.focal_length[0] as f64,
                self.metadata.intrinsics.focal_length[1] as f64,
            ],
            principal_point: [
                self.metadata.intrinsics.principal_point[0] as f64,
                self.metadata.intrinsics.principal_point[1] as f64,
            ],
            image_size: self.metadata.intrinsics.image_size,
        }
    }

    /// Load a specific render by object, rotation index, and viewpoint index
    ///
    /// # Arguments
    /// * `object_id` - YCB object ID (e.g., "003_cracker_box")
    /// * `rotation_idx` - Rotation index (0-2 for benchmark rotations)
    /// * `viewpoint_idx` - Viewpoint index (0-23 for default config)
    pub fn get_render(
        &self,
        object_id: &str,
        rotation_idx: usize,
        viewpoint_idx: usize,
    ) -> Result<RenderOutput, FixtureError> {
        // Find the render metadata
        let renders = self
            .indices
            .get(object_id)
            .ok_or_else(|| FixtureError::RenderNotFound {
                object_id: object_id.to_string(),
                rotation: rotation_idx,
                viewpoint: viewpoint_idx,
            })?;

        let render_meta = renders
            .iter()
            .find(|r| r.rotation_index == rotation_idx && r.viewpoint_index == viewpoint_idx)
            .ok_or_else(|| FixtureError::RenderNotFound {
                object_id: object_id.to_string(),
                rotation: rotation_idx,
                viewpoint: viewpoint_idx,
            })?;

        // Load RGBA from PNG
        let rgba_path = self.root.join(object_id).join(&render_meta.rgba_file);
        let rgba = load_rgba_png(&rgba_path)?;

        // Load depth from binary
        let depth_path = self.root.join(object_id).join(&render_meta.depth_file);
        let expected_depth_values =
            (self.metadata.resolution[0] as usize) * (self.metadata.resolution[1] as usize);
        let depth = load_depth_binary(&depth_path, expected_depth_values)?;

        // Build camera transform from position. New manifests carry the exact
        // render rotation; older manifests only recorded origin-targeted
        // camera positions, so reconstruct with a target fallback.
        let pos = render_meta.camera_position;
        let translation = Vec3::new(pos[0], pos[1], pos[2]);
        let camera_transform = if let Some(q) = render_meta.camera_rotation_xyzw {
            Transform {
                translation,
                rotation: Quat::from_xyzw(q[0], q[1], q[2], q[3]),
                ..Default::default()
            }
        } else {
            let target = render_meta.target_point.unwrap_or([0.0, 0.0, 0.0]);
            Transform::from_translation(translation)
                .looking_at(Vec3::new(target[0], target[1], target[2]), Vec3::Y)
        };
        let target_point = Vec3::from_array(render_meta.target_point.unwrap_or([0.0, 0.0, 0.0]));
        let targeting_policy = render_meta
            .targeting_policy
            .clone()
            .or_else(|| self.metadata.targeting_policy.clone())
            .unwrap_or(TargetingPolicy::Origin);

        // Build object rotation (convert from f32 metadata to f64)
        let rot = render_meta.rotation_euler;
        let object_rotation =
            crate::ObjectRotation::new(rot[0] as f64, rot[1] as f64, rot[2] as f64);
        let object_translation =
            Vec3::from_array(render_meta.object_translation.unwrap_or([0.0, 0.0, 0.0]));
        let object_scale = Vec3::from_array(render_meta.object_scale.unwrap_or([1.0, 1.0, 1.0]));

        Ok(RenderOutput {
            rgba,
            depth,
            width: self.metadata.resolution[0],
            height: self.metadata.resolution[1],
            intrinsics: self.intrinsics(),
            camera_transform,
            object_rotation,
            object_translation,
            object_scale,
            target_point,
            targeting_policy,
        })
    }

    /// Load all renders for an object
    pub fn get_all_renders(&self, object_id: &str) -> Result<Vec<RenderOutput>, FixtureError> {
        let renders = self
            .indices
            .get(object_id)
            .ok_or_else(|| FixtureError::RenderNotFound {
                object_id: object_id.to_string(),
                rotation: 0,
                viewpoint: 0,
            })?;

        let mut outputs = Vec::with_capacity(renders.len());
        for meta in renders {
            let output = self.get_render(object_id, meta.rotation_index, meta.viewpoint_index)?;
            outputs.push(output);
        }

        Ok(outputs)
    }

    /// Iterate over all renders for an object
    pub fn iter_renders<'a>(
        &'a self,
        object_id: &'a str,
    ) -> impl Iterator<Item = Result<(usize, usize, RenderOutput), FixtureError>> + 'a {
        let renders = self.indices.get(object_id);

        renders.into_iter().flat_map(|v| v.iter()).map(move |meta| {
            let output = self.get_render(object_id, meta.rotation_index, meta.viewpoint_index)?;
            Ok((meta.rotation_index, meta.viewpoint_index, output))
        })
    }
}

/// Load RGBA data from a PNG file
fn load_rgba_png(path: &Path) -> Result<Vec<u8>, FixtureError> {
    let img = image::open(path).map_err(|e| FixtureError::IoError(std::io::Error::other(e)))?;

    let rgba = img.to_rgba8();
    Ok(rgba.into_raw())
}

/// Load depth data from binary f32 or f64 file and normalize to f64 for TBP precision.
fn load_depth_binary(path: &Path, expected_values: usize) -> Result<Vec<f64>, FixtureError> {
    let bytes = fs::read(path)?;

    if bytes.len() == expected_values * std::mem::size_of::<f64>() {
        return Ok(bytes
            .chunks_exact(8)
            .map(|chunk| {
                let arr: [u8; 8] = chunk.try_into().unwrap();
                f64::from_le_bytes(arr)
            })
            .collect());
    }

    if bytes.len() == expected_values * std::mem::size_of::<f32>() {
        return Ok(bytes
            .chunks_exact(4)
            .map(|chunk| {
                let arr: [u8; 4] = chunk.try_into().unwrap();
                f32::from_le_bytes(arr) as f64
            })
            .collect());
    }

    Err(FixtureError::InvalidMetadata(format!(
        "Depth file {} has {} bytes, expected {} f32 values or {} f64 values",
        path.display(),
        bytes.len(),
        expected_values,
        expected_values
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_fixture_not_found() {
        let result = TestFixtures::load("/nonexistent/path");
        assert!(matches!(result, Err(FixtureError::NotFound(_))));
    }

    #[test]
    fn test_fixtures_exists() {
        assert!(!TestFixtures::exists("/nonexistent/path"));
    }

    #[test]
    fn test_fixture_error_display() {
        let errors = vec![
            FixtureError::NotFound("/path".to_string()),
            FixtureError::InvalidMetadata("bad json".to_string()),
            FixtureError::RenderNotFound {
                object_id: "obj".to_string(),
                rotation: 0,
                viewpoint: 5,
            },
            FixtureError::IoError(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "file not found",
            )),
            FixtureError::JsonError(serde_json::from_str::<String>("invalid").unwrap_err()),
        ];

        for err in errors {
            let msg = err.to_string();
            assert!(!msg.is_empty());
        }
    }

    #[test]
    fn test_fixture_missing_metadata() {
        let temp_dir = TempDir::new().unwrap();
        let result = TestFixtures::load(temp_dir.path());
        assert!(matches!(result, Err(FixtureError::InvalidMetadata(_))));
    }

    #[test]
    fn test_fixture_load_metadata() {
        let temp_dir = TempDir::new().unwrap();

        // Create minimal metadata
        let metadata = DatasetMetadata {
            version: "1.0".to_string(),
            crate_version: None,
            renderer_policy_version: None,
            objects: vec!["test_object".to_string()],
            viewpoints_per_rotation: 24,
            rotations_per_object: 3,
            renders_per_object: 72,
            resolution: [64, 64],
            resolution_width: None,
            resolution_height: None,
            targeting_policy: None,
            intrinsics: IntrinsicsMetadata {
                focal_length: [55.4, 55.4],
                principal_point: [32.0, 32.0],
                image_size: [64, 64],
            },
            viewpoint_config: ViewpointConfigMetadata {
                radius: 0.5,
                yaw_count: 8,
                pitch_angles_deg: vec![-30.0, 0.0, 30.0],
            },
            rotations: vec![[0.0, 0.0, 0.0], [0.0, 90.0, 0.0], [0.0, 180.0, 0.0]],
        };

        let metadata_json = serde_json::to_string_pretty(&metadata).unwrap();
        let metadata_path = temp_dir.path().join("metadata.json");
        fs::write(&metadata_path, &metadata_json).unwrap();

        // Create object directory with empty index
        let obj_dir = temp_dir.path().join("test_object");
        fs::create_dir_all(&obj_dir).unwrap();
        fs::write(obj_dir.join("index.json"), "[]").unwrap();

        // Load fixtures
        let fixtures = TestFixtures::load(temp_dir.path()).unwrap();

        assert_eq!(fixtures.objects(), &["test_object"]);
        assert_eq!(fixtures.viewpoints_per_rotation(), 24);
        assert_eq!(fixtures.rotations_per_object(), 3);
        assert_eq!(fixtures.renders_for_object("test_object"), 0);
        assert_eq!(fixtures.renders_for_object("nonexistent"), 0);

        let intrinsics = fixtures.intrinsics();
        assert_eq!(intrinsics.image_size, [64, 64]);
    }

    #[test]
    fn test_load_depth_binary_f32() {
        let temp_dir = TempDir::new().unwrap();
        let depth_path = temp_dir.path().join("test.depth");

        // Write test depth values
        let depths: Vec<f32> = vec![0.5, 1.0, 2.0, 10.0];
        let bytes: Vec<u8> = depths.iter().flat_map(|f| f.to_le_bytes()).collect();
        fs::write(&depth_path, &bytes).unwrap();

        // Load and verify
        let loaded = load_depth_binary(&depth_path, depths.len()).unwrap();
        assert_eq!(loaded.len(), 4);
        assert!((loaded[0] - 0.5).abs() < 0.001);
        assert!((loaded[1] - 1.0).abs() < 0.001);
        assert!((loaded[2] - 2.0).abs() < 0.001);
        assert!((loaded[3] - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_load_depth_binary_f64() {
        let temp_dir = TempDir::new().unwrap();
        let depth_path = temp_dir.path().join("test.depth");

        let depths: Vec<f64> = vec![0.5, 1.0, 2.0, 10.0];
        let bytes: Vec<u8> = depths.iter().flat_map(|f| f.to_le_bytes()).collect();
        fs::write(&depth_path, &bytes).unwrap();

        let loaded = load_depth_binary(&depth_path, depths.len()).unwrap();
        assert_eq!(loaded, depths);
    }

    #[test]
    fn test_metadata_serialization_roundtrip() {
        let metadata = DatasetMetadata {
            version: "1.0".to_string(),
            crate_version: Some("0.5.5".to_string()),
            renderer_policy_version: Some(crate::RENDERER_POLICY_VERSION.to_string()),
            objects: vec!["obj1".to_string(), "obj2".to_string()],
            viewpoints_per_rotation: 24,
            rotations_per_object: 3,
            renders_per_object: 72,
            resolution: [64, 64],
            resolution_width: Some(64),
            resolution_height: Some(64),
            targeting_policy: Some(TargetingPolicy::MeshCenter),
            intrinsics: IntrinsicsMetadata {
                focal_length: [55.4, 55.4],
                principal_point: [32.0, 32.0],
                image_size: [64, 64],
            },
            viewpoint_config: ViewpointConfigMetadata {
                radius: 0.5,
                yaw_count: 8,
                pitch_angles_deg: vec![-30.0, 0.0, 30.0],
            },
            rotations: vec![[0.0, 0.0, 0.0]],
        };

        let json = serde_json::to_string(&metadata).unwrap();
        let loaded: DatasetMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.version, metadata.version);
        assert_eq!(loaded.objects, metadata.objects);
        assert_eq!(loaded.resolution, metadata.resolution);
    }

    #[test]
    fn test_render_metadata_serialization() {
        let meta = RenderMetadata {
            object_id: "003_cracker_box".to_string(),
            rotation_index: 1,
            viewpoint_index: 5,
            rotation_euler: [0.0, 90.0, 0.0],
            camera_position: [0.5, 0.0, 0.0],
            camera_rotation_xyzw: Some([0.0, 0.0, 0.0, 1.0]),
            object_translation: Some([0.1, 0.2, 0.3]),
            object_scale: Some([1.0, 1.25, 0.75]),
            target_point: Some([0.0, 0.0, 0.0]),
            targeting_policy: Some(TargetingPolicy::Origin),
            mesh_bounds: None,
            health: Some(RenderHealth {
                center_pixel: Some([32, 32]),
                center_depth: Some(0.25),
                center_foreground: true,
                foreground_pixel_count: 1,
                foreground_coverage: 1.0 / 4096.0,
                center_5x5_foreground_count: 1,
                nearest_foreground_pixel: Some([32, 32]),
                nearest_foreground_depth: Some(0.25),
                nearest_foreground_distance_px: Some(0.0),
            }),
            rgba_file: "r1_v05.png".to_string(),
            depth_file: "r1_v05.depth".to_string(),
        };

        let json = serde_json::to_string(&meta).unwrap();
        let loaded: RenderMetadata = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.object_id, meta.object_id);
        assert_eq!(loaded.rotation_index, meta.rotation_index);
        assert_eq!(loaded.viewpoint_index, meta.viewpoint_index);
    }
}
