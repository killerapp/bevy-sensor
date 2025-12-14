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
//!
//! For single-render mode (used internally for subprocess rendering):
//!   cargo run --bin prerender -- --single-render --object <name> --rotation <idx> --viewpoint <idx> --output <dir>

use bevy_sensor::ycb;
use bevy_sensor::{generate_viewpoints, ObjectRotation, RenderConfig, ViewpointConfig};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

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

    // Check for single-render mode (used by batch subprocess)
    if args.iter().any(|a| a == "--single-render") {
        run_single_render(&args);
        return;
    }

    // Batch mode
    run_batch_render(&args);
}

/// Run a single render (subprocess mode)
fn run_single_render(args: &[String]) {
    let object_id = parse_arg(args, "--object").expect("--object required for single-render");
    let rotation_idx: usize = parse_arg(args, "--rotation")
        .expect("--rotation required")
        .parse()
        .expect("--rotation must be a number");
    let viewpoint_idx: usize = parse_arg(args, "--viewpoint")
        .expect("--viewpoint required")
        .parse()
        .expect("--viewpoint must be a number");
    let output_dir =
        parse_arg(args, "--output").unwrap_or_else(|| "test_fixtures/renders".to_string());

    let ycb_dir = PathBuf::from("/tmp/ycb");
    let object_dir = ycb_dir.join(&object_id);

    let render_config = RenderConfig::tbp_default();
    let viewpoint_config = ViewpointConfig::default();
    let rotations = ObjectRotation::tbp_benchmark_rotations();
    let viewpoints = generate_viewpoints(&viewpoint_config);

    let rotation = &rotations[rotation_idx];
    let viewpoint = &viewpoints[viewpoint_idx];

    // Create output directory
    let object_output = PathBuf::from(&output_dir).join(&object_id);
    fs::create_dir_all(&object_output).expect("Failed to create output directory");

    // Output paths
    let rgba_path = object_output.join(format!("r{}_v{:02}.png", rotation_idx, viewpoint_idx));
    let depth_path = object_output.join(format!("r{}_v{:02}.depth", rotation_idx, viewpoint_idx));

    // Render directly to files - this function will call process::exit() when done
    let result = bevy_sensor::render_to_files(
        &object_dir,
        viewpoint,
        rotation,
        &render_config,
        &rgba_path,
        &depth_path,
    );

    // Only reached on error (render_to_files calls process::exit on success)
    if let Err(e) = result {
        eprintln!("RENDER_ERROR: {:?}", e);
        std::process::exit(1);
    }
}

/// Run batch rendering (main mode)
fn run_batch_render(args: &[String]) {
    let output_dir =
        parse_arg(args, "--output-dir").unwrap_or_else(|| "test_fixtures/renders".to_string());
    let objects_arg = parse_arg(args, "--objects");

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

    // Get the current executable path for subprocess spawning
    let exe_path = std::env::current_exe().expect("Failed to get current executable path");

    // Render each object using subprocesses
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
        let total_renders = viewpoints.len() * rotations.len();

        for (rot_idx, rotation) in rotations.iter().enumerate() {
            for (view_idx, viewpoint) in viewpoints.iter().enumerate() {
                // Spawn subprocess to render this viewpoint
                let status = Command::new(&exe_path)
                    .arg("--single-render")
                    .arg("--object")
                    .arg(object_id)
                    .arg("--rotation")
                    .arg(rot_idx.to_string())
                    .arg("--viewpoint")
                    .arg(view_idx.to_string())
                    .arg("--output")
                    .arg(&output_dir)
                    .env("WGPU_BACKEND", "vulkan")
                    .status();

                match status {
                    Ok(exit_status) if exit_status.success() => {
                        // Record metadata
                        let camera_pos = viewpoint.translation;
                        object_renders.push(RenderMetadata {
                            object_id: object_id.clone(),
                            rotation_index: rot_idx,
                            viewpoint_index: view_idx,
                            rotation_euler: [rotation.pitch, rotation.yaw, rotation.roll],
                            camera_position: [camera_pos.x, camera_pos.y, camera_pos.z],
                            rgba_file: format!("r{}_v{:02}.png", rot_idx, view_idx),
                            depth_file: format!("r{}_v{:02}.depth", rot_idx, view_idx),
                        });

                        render_count += 1;
                        print!("\r  Rendered {}/{}", render_count, total_renders);
                        use std::io::Write;
                        std::io::stdout().flush().ok();
                    }
                    Ok(exit_status) => {
                        println!(
                            "\n  ERROR rendering r{}_v{}: subprocess exited with {:?}",
                            rot_idx, view_idx, exit_status
                        );
                    }
                    Err(e) => {
                        println!(
                            "\n  ERROR rendering r{}_v{}: failed to spawn subprocess: {:?}",
                            rot_idx, view_idx, e
                        );
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
