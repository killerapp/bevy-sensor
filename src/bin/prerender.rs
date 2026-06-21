//! Pre-render YCB objects for CI/CD testing
//!
//! This binary renders YCB objects to disk for use in environments without GPU access.
//! The output can be committed to the repository for CI/CD testing.
//!
//! Usage:
//!   cargo run --bin prerender -- [--output-dir <path>] [--objects <obj1,obj2>] [--target origin|mesh-center]
//!   cargo run --bin prerender -- --validate-center-hit --objects <obj1,obj2> --target mesh-center
//!
//! Default output: test_fixtures/renders/
//! Default objects: 003_cracker_box, 005_tomato_soup_can
//! Default batch target: origin
//! Default validation target: mesh-center
//!
//! For single-render mode:
//!   cargo run --bin prerender -- --single-render --object <name> --rotation <idx> --viewpoint <idx> --output <dir>

use bevy_sensor::ycb;
use bevy_sensor::ycbust;
use bevy_sensor::{
    generate_targeted_viewpoints, render_batch, validate_center_hits, BatchRenderConfig,
    BatchRenderOutput, BatchRenderRequest, MeshBoundsMetadata, ObjectRotation, RenderConfig,
    RenderHealth, RenderStatus, TargetingPolicy, Vec3, ViewpointConfig, RENDERER_POLICY_VERSION,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Metadata for a pre-rendered dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    /// Version of the dataset format
    pub version: String,
    /// bevy-sensor crate version that generated the dataset
    pub crate_version: String,
    /// Renderer/targeting-policy version
    pub renderer_policy_version: String,
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
    /// Named image width for manifest consumers
    pub resolution_width: u32,
    /// Named image height for manifest consumers
    pub resolution_height: u32,
    /// Targeting policy used for viewpoint generation
    pub targeting_policy: TargetingPolicy,
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
    pub camera_rotation_xyzw: [f32; 4],
    pub object_translation: [f32; 3],
    pub object_scale: [f32; 3],
    pub target_point: [f32; 3],
    pub targeting_policy: TargetingPolicy,
    pub mesh_bounds: Option<MeshBoundsMetadata>,
    pub health: RenderHealth,
    pub rgba_file: String,
    pub depth_file: String,
}

/// Default objects for CI/CD testing
const CI_TEST_OBJECTS: &[&str] = &["003_cracker_box", "005_tomato_soup_can"];

fn main() {
    // Initialize backend configuration FIRST
    // This ensures proper backend selection (WebGPU for WSL2, etc.)
    bevy_sensor::initialize();

    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();

    // Keep single-render mode available for scripts that render one capture.
    if args.iter().any(|a| a == "--single-render") {
        run_single_render(&args);
        return;
    }

    if args.iter().any(|a| a == "--validate-center-hit") {
        run_center_hit_validation(&args);
        return;
    }

    // Batch mode
    run_batch_render(&args);
}

/// Run a single render (subprocess mode)
fn run_single_render(args: &[String]) {
    if let Err(e) = run_single_render_impl(args) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run_single_render_impl(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let object_id =
        parse_arg(args, "--object").ok_or("Error: --object required for single-render")?;
    let rotation_idx: usize = parse_arg(args, "--rotation")
        .ok_or("Error: --rotation required")?
        .parse()
        .map_err(|_| "Error: --rotation must be a valid number")?;
    let viewpoint_idx: usize = parse_arg(args, "--viewpoint")
        .ok_or("Error: --viewpoint required")?
        .parse()
        .map_err(|_| "Error: --viewpoint must be a valid number")?;
    let output_dir =
        parse_arg(args, "--output").unwrap_or_else(|| "test_fixtures/renders".to_string());
    let data_dir_str = parse_arg(args, "--data-dir").unwrap_or_else(|| "/tmp/ycb".to_string());
    let target_policy = parse_target_policy(args, "origin")?;

    let ycb_dir = PathBuf::from(&data_dir_str);

    ensure_ycb_objects(&ycb_dir, &[object_id.as_str()])?;

    // Canonicalize to ensure absolute path for Bevy asset loading
    let ycb_dir = fs::canonicalize(&ycb_dir).unwrap_or(ycb_dir);
    let object_dir = ycb_dir.join(&object_id);

    let render_config = RenderConfig::tbp_default();
    let viewpoint_config = ViewpointConfig::default();
    let rotations = rotations_from_args(args)?;

    let rotation = &rotations[rotation_idx];
    let targeted =
        generate_targeted_viewpoints(&object_dir, &viewpoint_config, rotation, &target_policy)?;
    let viewpoint = targeted.viewpoints.get(viewpoint_idx).ok_or_else(|| {
        format!(
            "Error: --viewpoint {} out of range for {} generated viewpoints",
            viewpoint_idx,
            targeted.viewpoints.len()
        )
    })?;

    // Create output directory
    let object_output = PathBuf::from(&output_dir).join(&object_id);
    fs::create_dir_all(&object_output).map_err(|e| {
        format!(
            "Error: Failed to create output directory {}: {}",
            object_output.display(),
            e
        )
    })?;

    // Output paths
    let rgba_path = object_output.join(format!("r{}_v{:02}.png", rotation_idx, viewpoint_idx));
    let depth_path = object_output.join(format!("r{}_v{:02}.depth", rotation_idx, viewpoint_idx));

    // Render directly to files - this function will call process::exit() when done
    bevy_sensor::render_to_files(
        &object_dir,
        viewpoint,
        rotation,
        &render_config,
        &rgba_path,
        &depth_path,
    )
    .map_err(|e| format!("Render failed: {}", e))?;

    // If render succeeded, process::exit() was already called by render_to_files
    // This line should not be reached
    Ok(())
}

/// Run center-hit validation without saving render outputs.
fn run_center_hit_validation(args: &[String]) {
    if let Err(e) = run_center_hit_validation_impl(args) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run_center_hit_validation_impl(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let data_dir_str = parse_arg(args, "--data-dir").unwrap_or_else(|| "/tmp/ycb".to_string());
    let objects_arg = parse_arg(args, "--objects");
    let target_policy = parse_target_policy(args, "mesh-center")?;
    let rotations = rotations_from_args(args)?;

    let objects: Vec<String> = if let Some(objs) = objects_arg {
        objs.split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect()
    } else {
        CI_TEST_OBJECTS.iter().map(|s| s.to_string()).collect()
    };

    println!("=== bevy-sensor Center-Hit Validation ===");
    println!("Data directory: {}", data_dir_str);
    println!("Objects: {:?}", objects);
    println!("Targeting: {}", target_policy.label());
    println!("Rotations: {}", rotations.len());

    let ycb_dir = PathBuf::from(&data_dir_str);
    let object_refs: Vec<&str> = objects.iter().map(String::as_str).collect();
    ensure_ycb_objects(&ycb_dir, &object_refs)?;
    let ycb_dir = fs::canonicalize(&ycb_dir).unwrap_or(ycb_dir);

    let render_config = RenderConfig::tbp_default();
    let viewpoint_config = ViewpointConfig::default();
    let mut reports = Vec::with_capacity(objects.len());

    for object_id in &objects {
        let object_dir = ycb_dir.join(object_id);
        println!("\n--- Validating {} ---", object_id);
        let report = validate_center_hits(
            object_id.clone(),
            &object_dir,
            &viewpoint_config,
            &rotations,
            &render_config,
            &target_policy,
        )?;

        for rotation in &report.rotations {
            println!(
                "  rotation {} {:?}: {}/{} center hits",
                rotation.rotation_index,
                rotation.rotation_euler,
                rotation.center_hits,
                rotation.total_viewpoints
            );
        }

        let report_json = serde_json::to_string_pretty(&report)?;
        println!("{}", report_json);

        if !report.is_valid() {
            return Err(format!(
                "{} failed center-hit validation for rotations {:?}",
                object_id,
                report.zero_hit_rotations()
            )
            .into());
        }

        reports.push(report);
    }

    if let Some(report_path) = parse_arg(args, "--validation-report") {
        let report_path = PathBuf::from(report_path);
        if let Some(parent) = report_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&report_path, serde_json::to_string_pretty(&reports)?)?;
        println!("\nSaved validation report to {}", report_path.display());
    }

    println!(
        "\nCenter-hit validation passed for {} objects.",
        reports.len()
    );
    Ok(())
}

/// Run batch rendering (main mode)
fn run_batch_render(args: &[String]) {
    if let Err(e) = run_batch_render_impl(args) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run_batch_render_impl(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    let output_dir =
        parse_arg(args, "--output-dir").unwrap_or_else(|| "test_fixtures/renders".to_string());
    let data_dir_str = parse_arg(args, "--data-dir").unwrap_or_else(|| "/tmp/ycb".to_string());
    let objects_arg = parse_arg(args, "--objects");
    let target_policy = parse_target_policy(args, "origin")?;
    let rotations = rotations_from_args(args)?;

    let objects: Vec<String> = if let Some(objs) = objects_arg {
        objs.split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect()
    } else {
        CI_TEST_OBJECTS.iter().map(|s| s.to_string()).collect()
    };

    println!("=== bevy-sensor Pre-renderer ===");
    println!("Output directory: {}", output_dir);
    println!("Data directory: {}", data_dir_str);
    println!("Objects: {:?}", objects);
    println!("Targeting: {}", target_policy.label());

    let ycb_dir = PathBuf::from(&data_dir_str);
    let object_refs: Vec<&str> = objects.iter().map(String::as_str).collect();
    ensure_ycb_objects(&ycb_dir, &object_refs)?;

    // Canonicalize to ensure absolute path for Bevy asset loading
    let ycb_dir = fs::canonicalize(&ycb_dir).unwrap_or(ycb_dir);
    println!("YCB models found at {:?}", ycb_dir);

    // Create output directory
    let output_path = PathBuf::from(&output_dir);
    fs::create_dir_all(&output_path).map_err(|e| {
        format!(
            "Failed to create output directory {}: {}",
            output_path.display(),
            e
        )
    })?;

    // Configuration
    let render_config = RenderConfig::tbp_default();
    let viewpoint_config = ViewpointConfig::default();
    let viewpoints_per_rotation = viewpoint_config.viewpoint_count();

    println!("\nConfiguration:");
    println!(
        "  Resolution: {}x{}",
        render_config.width, render_config.height
    );
    println!("  Viewpoints: {}", viewpoints_per_rotation);
    println!("  Rotations: {}", rotations.len());
    println!(
        "  Total renders per object: {}",
        viewpoints_per_rotation * rotations.len()
    );

    // Create dataset metadata
    let intrinsics = render_config.intrinsics();
    let metadata = DatasetMetadata {
        version: "1.1".to_string(),
        crate_version: env!("CARGO_PKG_VERSION").to_string(),
        renderer_policy_version: RENDERER_POLICY_VERSION.to_string(),
        objects: objects.clone(),
        viewpoints_per_rotation,
        rotations_per_object: rotations.len(),
        renders_per_object: viewpoints_per_rotation * rotations.len(),
        resolution: [render_config.width, render_config.height],
        resolution_width: render_config.width,
        resolution_height: render_config.height,
        targeting_policy: target_policy.clone(),
        intrinsics: IntrinsicsMetadata {
            // Convert f64 intrinsics to f32 for JSON serialization (backward compatible)
            focal_length: [
                intrinsics.focal_length[0] as f32,
                intrinsics.focal_length[1] as f32,
            ],
            principal_point: [
                intrinsics.principal_point[0] as f32,
                intrinsics.principal_point[1] as f32,
            ],
            image_size: intrinsics.image_size,
        },
        viewpoint_config: ViewpointConfigMetadata {
            radius: viewpoint_config.radius,
            yaw_count: viewpoint_config.yaw_count,
            pitch_angles_deg: viewpoint_config.pitch_angles_deg.clone(),
        },
        // Convert f64 rotations to f32 for JSON serialization (backward compatible)
        rotations: rotations
            .iter()
            .map(|r| [r.pitch as f32, r.yaw as f32, r.roll as f32])
            .collect(),
    };

    // Save dataset metadata
    let metadata_path = output_path.join("metadata.json");
    let metadata_json = serde_json::to_string_pretty(&metadata)
        .map_err(|e| format!("Failed to serialize metadata JSON: {}", e))?;
    fs::write(&metadata_path, &metadata_json).map_err(|e| {
        format!(
            "Failed to write metadata to {}: {}",
            metadata_path.display(),
            e
        )
    })?;
    println!("\nSaved dataset metadata to {:?}", metadata_path);

    // Render each object in-process. Grouping by object and rotation lets the library reuse
    // scene setup across the 24 viewpoints while keeping output filenames unchanged.
    let mut all_render_metadata: HashMap<String, Vec<RenderMetadata>> = HashMap::new();

    for object_id in &objects {
        println!("\n--- Rendering {} ---", object_id);

        let object_dir = ycb_dir.join(object_id);
        if !object_dir.exists() {
            return Err(format!(
                "Object directory not found after validation: {:?}",
                object_dir
            )
            .into());
        }

        // Create object output directory
        let object_output = output_path.join(object_id);
        fs::create_dir_all(&object_output).map_err(|e| {
            format!(
                "Failed to create object directory {}: {}",
                object_output.display(),
                e
            )
        })?;

        let mut object_renders: Vec<RenderMetadata> = Vec::new();
        let mut render_count = 0;
        let total_renders = viewpoints_per_rotation * rotations.len();

        for (rot_idx, rotation) in rotations.iter().enumerate() {
            let targeted = generate_targeted_viewpoints(
                &object_dir,
                &viewpoint_config,
                rotation,
                &target_policy,
            )
            .map_err(|e| {
                format!(
                    "Failed to generate targeted viewpoints for {} rotation {}: {}",
                    object_id, rot_idx, e
                )
            })?;

            let requests: Vec<BatchRenderRequest> = targeted
                .viewpoints
                .iter()
                .map(|viewpoint| BatchRenderRequest {
                    object_dir: object_dir.clone(),
                    viewpoint: *viewpoint,
                    object_rotation: rotation.clone(),
                    object_translation: Vec3::ZERO,
                    object_scale: Vec3::ONE,
                    render_config: render_config.clone(),
                    target_point: targeted.target_point,
                    targeting_policy: target_policy.clone(),
                })
                .collect();

            let outputs = render_batch(requests, &BatchRenderConfig::default()).map_err(|e| {
                format!("Failed to render {} rotation {}: {}", object_id, rot_idx, e)
            })?;

            for (view_idx, output) in outputs.iter().enumerate() {
                if output.status != RenderStatus::Success {
                    return Err(format!(
                        "Render failed for {} r{}_v{:02}: {:?}",
                        object_id, rot_idx, view_idx, output.error_message
                    )
                    .into());
                }

                let rgba_file = format!("r{}_v{:02}.png", rot_idx, view_idx);
                let depth_file = format!("r{}_v{:02}.depth", rot_idx, view_idx);
                save_batch_render_output(
                    output,
                    &object_output.join(&rgba_file),
                    &object_output.join(&depth_file),
                )?;

                let camera_pos = output.request.viewpoint.translation;
                let camera_rot = output.request.viewpoint.rotation;
                let object_translation = output.object_translation;
                let object_scale = output.object_scale;
                object_renders.push(RenderMetadata {
                    object_id: object_id.clone(),
                    rotation_index: rot_idx,
                    viewpoint_index: view_idx,
                    // Convert f64 rotation to f32 for JSON serialization
                    rotation_euler: [
                        rotation.pitch as f32,
                        rotation.yaw as f32,
                        rotation.roll as f32,
                    ],
                    camera_position: [camera_pos.x, camera_pos.y, camera_pos.z],
                    camera_rotation_xyzw: [camera_rot.x, camera_rot.y, camera_rot.z, camera_rot.w],
                    object_translation: object_translation.to_array(),
                    object_scale: object_scale.to_array(),
                    target_point: output.target_point.to_array(),
                    targeting_policy: output.targeting_policy.clone(),
                    mesh_bounds: targeted.mesh_bounds.map(MeshBoundsMetadata::from),
                    health: output.health.clone(),
                    rgba_file,
                    depth_file,
                });

                render_count += 1;
                print!("\r  Rendered {}/{}", render_count, total_renders);
                use std::io::Write;
                std::io::stdout().flush().ok();
            }
        }
        println!();

        // Save object render index
        let index_path = object_output.join("index.json");
        let index_json = serde_json::to_string_pretty(&object_renders)
            .map_err(|e| format!("Failed to serialize object index JSON: {}", e))?;
        fs::write(&index_path, &index_json)
            .map_err(|e| format!("Failed to write index to {}: {}", index_path.display(), e))?;
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

    Ok(())
}

fn save_batch_render_output(
    output: &BatchRenderOutput,
    rgba_path: &Path,
    depth_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = rgba_path.parent() {
        fs::create_dir_all(parent)?;
    }
    image::save_buffer(
        rgba_path,
        &output.rgba,
        output.width,
        output.height,
        image::ColorType::Rgba8,
    )?;

    if let Some(parent) = depth_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let depth_bytes: Vec<u8> = output
        .depth
        .iter()
        .flat_map(|depth| depth.to_le_bytes())
        .collect();
    fs::write(depth_path, depth_bytes)?;

    Ok(())
}

fn parse_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1).cloned())
}

fn parse_target_policy(
    args: &[String],
    default: &str,
) -> Result<TargetingPolicy, Box<dyn std::error::Error>> {
    let value = parse_arg(args, "--target").unwrap_or_else(|| default.to_string());
    match value.as_str() {
        "origin" => Ok(TargetingPolicy::Origin),
        "mesh-center" | "mesh_center" => Ok(TargetingPolicy::MeshCenter),
        other => Err(format!(
            "Invalid --target '{}'. Supported values: origin, mesh-center",
            other
        )
        .into()),
    }
}

fn rotations_from_args(args: &[String]) -> Result<Vec<ObjectRotation>, Box<dyn std::error::Error>> {
    let schedule =
        parse_arg(args, "--rotation-schedule").unwrap_or_else(|| "tbp-parity".to_string());
    match schedule.as_str() {
        "tbp-parity" | "tbp-benchmark" | "benchmark" => {
            Ok(ObjectRotation::tbp_benchmark_rotations())
        }
        "tbp-known" | "tbp-known-orientations" | "known" | "full" => {
            Ok(ObjectRotation::tbp_known_orientations())
        }
        other => Err(format!(
            "Invalid --rotation-schedule '{}'. Supported values: tbp-parity, tbp-known",
            other
        )
        .into()),
    }
}

fn ensure_ycb_objects(
    ycb_dir: &Path,
    object_ids: &[&str],
) -> Result<(), Box<dyn std::error::Error>> {
    let missing = ycb::missing_objects(ycb_dir, object_ids);
    if missing.is_empty() {
        return Ok(());
    }

    println!("\nMissing YCB objects at {:?}: {:?}", ycb_dir, missing);
    println!("Downloading missing objects...");

    let missing_refs: Vec<&str> = missing.iter().map(String::as_str).collect();
    ycbust::blocking::download_objects_blocking(
        &missing_refs,
        ycb_dir,
        ycbust::DownloadOptions::default(),
    )?;

    let still_missing = ycb::missing_objects(ycb_dir, object_ids);
    if !still_missing.is_empty() {
        return Err(format!(
            "YCB download completed but objects are still incomplete: {:?}",
            still_missing
        )
        .into());
    }

    println!("Download complete.");
    Ok(())
}
