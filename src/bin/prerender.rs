//! Pre-render YCB objects for CI/CD testing
//!
//! This binary renders YCB objects to disk for use in environments without GPU access.
//! The output can be committed to the repository for CI/CD testing.
//!
//! Usage:
//!   cargo run --bin prerender -- [--output-dir <path>] [--objects <obj1,obj2>]
//!
//! Default output: test_fixtures/renders/
//! Default objects: 003_cracker_box, 005_tomato_soup_can

use bevy_sensor::ycb;
use bevy_sensor::{
    generate_viewpoints, ObjectRotation, RenderConfig, RenderOutput, ViewpointConfig,
};
use image::{ImageBuffer, Rgba};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Metadata for a pre-rendered dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// Version of the dataset format
    pub version: String,
    /// Objects included in this dataset
    pub objects: Vec<String>,
    /// Number of viewpoints per rotation
    pub viewpoints_per_rotation: usize,
    /// Number of rotations per object
    pub rotations_per_object: usize,
    /// Total renders per object
    pub renders_per_object: usize,
    /// Image resolution
    pub resolution: [u32; 2],
    /// Camera intrinsics
    pub intrinsics: IntrinsicsMetadata,
    /// Viewpoint configuration
    pub viewpoint_config: ViewpointConfigMetadata,
    /// Object rotations used
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
    pub rgba_file: String,
    pub depth_file: String,
}

/// Default objects for CI/CD testing
const CI_TEST_OBJECTS: &[&str] = &["003_cracker_box", "005_tomato_soup_can"];

fn main() {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    let output_dir =
        parse_arg(&args, "--output-dir").unwrap_or_else(|| "test_fixtures/renders".to_string());
    let objects_arg = parse_arg(&args, "--objects");

    let objects: Vec<String> = if let Some(objs) = objects_arg {
        objs.split(',').map(|s| s.trim().to_string()).collect()
    } else {
        CI_TEST_OBJECTS.iter().map(|s| s.to_string()).collect()
    };

    println!("=== bevy-sensor Pre-renderer ===");
    println!("Output directory: {}", output_dir);
    println!("Objects: {:?}", objects);

    // Check if YCB models exist
    let ycb_dir = PathBuf::from("/tmp/ycb");
    if !ycb::models_exist(&ycb_dir) {
        println!("\nYCB models not found at {:?}", ycb_dir);
        println!("Please download first:");
        println!("  cargo run --example test_render");
        println!("Or use ycbust directly.");
        std::process::exit(1);
    }
    println!("YCB models found at {:?}", ycb_dir);

    // Create output directory
    let output_path = PathBuf::from(&output_dir);
    fs::create_dir_all(&output_path).expect("Failed to create output directory");

    // Configuration
    let render_config = RenderConfig::tbp_default();
    let viewpoint_config = ViewpointConfig::default();
    let rotations = ObjectRotation::tbp_benchmark_rotations();
    let viewpoints = generate_viewpoints(&viewpoint_config);

    println!("\nConfiguration:");
    println!(
        "  Resolution: {}x{}",
        render_config.width, render_config.height
    );
    println!("  Viewpoints: {}", viewpoints.len());
    println!("  Rotations: {}", rotations.len());
    println!(
        "  Total renders per object: {}",
        viewpoints.len() * rotations.len()
    );

    // Create dataset metadata
    let intrinsics = render_config.intrinsics();
    let metadata = DatasetMetadata {
        version: "1.0".to_string(),
        objects: objects.clone(),
        viewpoints_per_rotation: viewpoints.len(),
        rotations_per_object: rotations.len(),
        renders_per_object: viewpoints.len() * rotations.len(),
        resolution: [render_config.width, render_config.height],
        intrinsics: IntrinsicsMetadata {
            focal_length: intrinsics.focal_length,
            principal_point: intrinsics.principal_point,
            image_size: intrinsics.image_size,
        },
        viewpoint_config: ViewpointConfigMetadata {
            radius: viewpoint_config.radius,
            yaw_count: viewpoint_config.yaw_count,
            pitch_angles_deg: viewpoint_config.pitch_angles_deg.clone(),
        },
        rotations: rotations.iter().map(|r| [r.pitch, r.yaw, r.roll]).collect(),
    };

    // Save dataset metadata
    let metadata_path = output_path.join("metadata.json");
    let metadata_json = serde_json::to_string_pretty(&metadata).unwrap();
    fs::write(&metadata_path, &metadata_json).expect("Failed to write metadata");
    println!("\nSaved dataset metadata to {:?}", metadata_path);

    // Render each object
    let mut all_render_metadata: HashMap<String, Vec<RenderMetadata>> = HashMap::new();

    for object_id in &objects {
        println!("\n--- Rendering {} ---", object_id);

        let object_dir = ycb_dir.join(object_id);
        if !object_dir.exists() {
            println!("  WARNING: Object directory not found: {:?}", object_dir);
            println!("  Skipping...");
            continue;
        }

        // Create object output directory
        let object_output = output_path.join(object_id);
        fs::create_dir_all(&object_output).expect("Failed to create object directory");

        let mut object_renders: Vec<RenderMetadata> = Vec::new();
        let mut render_count = 0;

        for (rot_idx, rotation) in rotations.iter().enumerate() {
            for (view_idx, viewpoint) in viewpoints.iter().enumerate() {
                // Render
                let result =
                    bevy_sensor::render_to_buffer(&object_dir, viewpoint, rotation, &render_config);

                match result {
                    Ok(output) => {
                        // Save RGBA as PNG
                        let rgba_filename = format!("r{}_v{:02}.png", rot_idx, view_idx);
                        let rgba_path = object_output.join(&rgba_filename);
                        save_rgba_png(&output, &rgba_path);

                        // Save depth as binary f32
                        let depth_filename = format!("r{}_v{:02}.depth", rot_idx, view_idx);
                        let depth_path = object_output.join(&depth_filename);
                        save_depth_binary(&output, &depth_path);

                        // Record metadata
                        let camera_pos = viewpoint.translation;
                        object_renders.push(RenderMetadata {
                            object_id: object_id.clone(),
                            rotation_index: rot_idx,
                            viewpoint_index: view_idx,
                            rotation_euler: [rotation.pitch, rotation.yaw, rotation.roll],
                            camera_position: [camera_pos.x, camera_pos.y, camera_pos.z],
                            rgba_file: rgba_filename,
                            depth_file: depth_filename,
                        });

                        render_count += 1;
                        print!(
                            "\r  Rendered {}/{}",
                            render_count,
                            viewpoints.len() * rotations.len()
                        );
                    }
                    Err(e) => {
                        println!("\n  ERROR rendering r{}_v{}: {:?}", rot_idx, view_idx, e);
                    }
                }
            }
        }
        println!();

        // Save object render index
        let index_path = object_output.join("index.json");
        let index_json = serde_json::to_string_pretty(&object_renders).unwrap();
        fs::write(&index_path, &index_json).expect("Failed to write index");
        println!(
            "  Saved {} renders to {:?}",
            object_renders.len(),
            object_output
        );

        all_render_metadata.insert(object_id.clone(), object_renders);
    }

    // Summary
    let total_renders: usize = all_render_metadata.values().map(|v| v.len()).sum();
    println!("\n=== Pre-rendering Complete ===");
    println!("Total renders: {}", total_renders);
    println!("Output directory: {:?}", output_path);
    println!("\nTo use in tests:");
    println!("  let fixtures = TestFixtures::load(\"test_fixtures/renders\");");
}

fn parse_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

fn save_rgba_png(output: &RenderOutput, path: &Path) {
    let img: ImageBuffer<Rgba<u8>, Vec<u8>> =
        ImageBuffer::from_raw(output.width, output.height, output.rgba.clone())
            .expect("Failed to create image buffer");

    img.save(path).expect("Failed to save PNG");
}

fn save_depth_binary(output: &RenderOutput, path: &Path) {
    // Save as raw f32 bytes (little-endian)
    let bytes: Vec<u8> = output.depth.iter().flat_map(|f| f.to_le_bytes()).collect();

    fs::write(path, &bytes).expect("Failed to save depth");
}
