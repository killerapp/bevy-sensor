//! Replay a single logged camera pose through one-shot and persistent render paths.
//!
//! This is a narrow diagnostic utility for downstream parity failures where the
//! caller already has an exact `camera_t` + `camera_q_xyzw` from bevy-sensor logs.

use bevy_sensor::{
    load_ycb_mesh_bounds, render_to_buffer, rotated_mesh_center, ObjectRotation,
    PersistentRenderer, Quat, RenderConfig, RenderHealth, Transform, Vec3,
};
use serde::Serialize;
use std::env;
use std::error::Error;
use std::path::PathBuf;

#[derive(Debug)]
struct Options {
    data_dir: PathBuf,
    object: String,
    camera_t: Vec3,
    camera_q_xyzw: Quat,
    object_rotation: ObjectRotation,
    repeat: usize,
    path: ReplayPath,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReplayPath {
    Both,
    OneShot,
    Persistent,
}

#[derive(Debug, Serialize)]
struct PoseReplayReport {
    object: String,
    camera_t: [f32; 3],
    camera_q_xyzw: [f32; 4],
    object_rot_deg: [f64; 3],
    tbp_zoom: f32,
    near_plane: f32,
    far_plane: f32,
    target_diagnostics: Vec<TargetDiagnostic>,
    one_shot: Option<PathReport>,
    persistent: Option<PathReport>,
}

#[derive(Debug, Serialize)]
struct TargetDiagnostic {
    target: String,
    world_point: [f32; 3],
    camera_forward_dot: f32,
    camera_forward_angle_deg: f32,
    camera_local: [f32; 3],
    projected_pixel: Option<[f64; 2]>,
    projected_in_bounds: bool,
}

#[derive(Debug, Serialize)]
struct PathReport {
    path: String,
    frames: Vec<FrameReport>,
}

#[derive(Debug, Serialize)]
struct FrameReport {
    iteration: usize,
    width: u32,
    height: u32,
    center_depth: Option<f64>,
    center_foreground: bool,
    foreground_pixel_count: usize,
    foreground_coverage: f64,
    center_5x5_foreground_count: usize,
    nearest_foreground_pixel: Option<[u32; 2]>,
    nearest_foreground_depth: Option<f64>,
    nearest_foreground_distance_px: Option<f64>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::parse(env::args().collect())?;
    let object_dir = options.data_dir.join(&options.object);
    let config = RenderConfig::tbp_default();
    let camera_transform = Transform {
        translation: options.camera_t,
        rotation: options.camera_q_xyzw,
        ..Default::default()
    };

    let mut target_diagnostics = vec![target_diagnostic(
        "origin",
        Vec3::ZERO,
        &camera_transform,
        &config,
    )];
    if let Ok(bounds) = load_ycb_mesh_bounds(&object_dir) {
        let mesh_center = rotated_mesh_center(bounds.center, &options.object_rotation);
        target_diagnostics.push(target_diagnostic(
            "rotated_mesh_center",
            mesh_center,
            &camera_transform,
            &config,
        ));
    }

    let one_shot = if matches!(options.path, ReplayPath::Both | ReplayPath::OneShot) {
        let mut frames = Vec::with_capacity(options.repeat);
        for iteration in 0..options.repeat {
            let output = render_to_buffer(
                &object_dir,
                &camera_transform,
                &options.object_rotation,
                &config,
            )?;
            frames.push(frame_report(
                iteration,
                &output.health_with_far_plane(config.far_plane as f64),
                output.width,
                output.height,
            ));
        }
        Some(PathReport {
            path: "one-shot".to_string(),
            frames,
        })
    } else {
        None
    };

    let persistent = if matches!(options.path, ReplayPath::Both | ReplayPath::Persistent) {
        let mut renderer = PersistentRenderer::new(&object_dir, &config)?;
        let mut frames = Vec::with_capacity(options.repeat);
        for iteration in 0..options.repeat {
            let output = renderer.render(&camera_transform, &options.object_rotation)?;
            frames.push(frame_report(
                iteration,
                &output.health_with_far_plane(config.far_plane as f64),
                output.width,
                output.height,
            ));
        }
        Some(PathReport {
            path: "persistent".to_string(),
            frames,
        })
    } else {
        None
    };

    let report = PoseReplayReport {
        object: options.object,
        camera_t: options.camera_t.to_array(),
        camera_q_xyzw: [
            options.camera_q_xyzw.x,
            options.camera_q_xyzw.y,
            options.camera_q_xyzw.z,
            options.camera_q_xyzw.w,
        ],
        object_rot_deg: [
            options.object_rotation.pitch,
            options.object_rotation.yaw,
            options.object_rotation.roll,
        ],
        tbp_zoom: config.zoom,
        near_plane: config.near_plane,
        far_plane: config.far_plane,
        target_diagnostics,
        one_shot,
        persistent,
    };

    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn frame_report(iteration: usize, health: &RenderHealth, width: u32, height: u32) -> FrameReport {
    FrameReport {
        iteration,
        width,
        height,
        center_depth: health.center_depth,
        center_foreground: health.center_foreground,
        foreground_pixel_count: health.foreground_pixel_count,
        foreground_coverage: health.foreground_coverage,
        center_5x5_foreground_count: health.center_5x5_foreground_count,
        nearest_foreground_pixel: health.nearest_foreground_pixel,
        nearest_foreground_depth: health.nearest_foreground_depth,
        nearest_foreground_distance_px: health.nearest_foreground_distance_px,
    }
}

fn target_diagnostic(
    target: &str,
    world_point: Vec3,
    camera_transform: &Transform,
    config: &RenderConfig,
) -> TargetDiagnostic {
    let forward = camera_transform.rotation * Vec3::NEG_Z;
    let to_target = (world_point - camera_transform.translation).normalize_or_zero();
    let dot = forward.dot(to_target).clamp(-1.0, 1.0);
    let angle = dot.acos().to_degrees();
    let local = camera_transform.rotation.inverse() * (world_point - camera_transform.translation);
    let projected_pixel = project_camera_local(local, config);
    let projected_in_bounds = projected_pixel
        .map(|[x, y]| x >= 0.0 && x < config.width as f64 && y >= 0.0 && y < config.height as f64)
        .unwrap_or(false);

    TargetDiagnostic {
        target: target.to_string(),
        world_point: world_point.to_array(),
        camera_forward_dot: dot,
        camera_forward_angle_deg: angle,
        camera_local: local.to_array(),
        projected_pixel,
        projected_in_bounds,
    }
}

fn project_camera_local(local: Vec3, config: &RenderConfig) -> Option<[f64; 2]> {
    if local.z >= 0.0 {
        return None;
    }
    let depth = -local.z as f64;
    let intrinsics = config.intrinsics();
    let x = (local.x as f64 / depth) * intrinsics.focal_length[0] + intrinsics.principal_point[0];
    let y = (-local.y as f64 / depth) * intrinsics.focal_length[1] + intrinsics.principal_point[1];
    Some([x, y])
}

impl Options {
    fn parse(args: Vec<String>) -> Result<Self, Box<dyn Error>> {
        let mut data_dir = None;
        let mut object = None;
        let mut camera_t = None;
        let mut camera_q_xyzw = None;
        let mut object_rotation = ObjectRotation::identity();
        let mut repeat = 1usize;
        let mut path = ReplayPath::Both;

        let mut index = 1usize;
        while index < args.len() {
            match args[index].as_str() {
                "--data-dir" => {
                    data_dir = Some(PathBuf::from(next_arg(&args, &mut index, "--data-dir")?));
                }
                "--object" => {
                    object = Some(next_arg(&args, &mut index, "--object")?.to_string());
                }
                "--camera-t" => {
                    camera_t = Some(parse_vec3(next_arg(&args, &mut index, "--camera-t")?)?);
                }
                "--camera-q-xyzw" => {
                    camera_q_xyzw =
                        Some(parse_quat(next_arg(&args, &mut index, "--camera-q-xyzw")?)?);
                }
                "--object-rot" => {
                    object_rotation =
                        parse_object_rotation(next_arg(&args, &mut index, "--object-rot")?)?;
                }
                "--repeat" => {
                    repeat = next_arg(&args, &mut index, "--repeat")?.parse()?;
                }
                "--path" => {
                    path = match next_arg(&args, &mut index, "--path")? {
                        "both" => ReplayPath::Both,
                        "one-shot" => ReplayPath::OneShot,
                        "persistent" => ReplayPath::Persistent,
                        value => {
                            return Err(format!(
                                "invalid --path `{value}`; expected both, one-shot, or persistent"
                            )
                            .into())
                        }
                    };
                }
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                unknown => return Err(format!("unknown argument `{unknown}`").into()),
            }
            index += 1;
        }

        Ok(Self {
            data_dir: data_dir.ok_or("--data-dir is required")?,
            object: object.ok_or("--object is required")?,
            camera_t: camera_t.ok_or("--camera-t is required")?,
            camera_q_xyzw: camera_q_xyzw.ok_or("--camera-q-xyzw is required")?,
            object_rotation,
            repeat,
            path,
        })
    }
}

fn next_arg<'a>(args: &'a [String], index: &mut usize, flag: &str) -> Result<&'a str, String> {
    *index += 1;
    args.get(*index)
        .map(String::as_str)
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_vec3(value: &str) -> Result<Vec3, Box<dyn Error>> {
    let parts = parse_csv_f32(value, 3)?;
    Ok(Vec3::new(parts[0], parts[1], parts[2]))
}

fn parse_quat(value: &str) -> Result<Quat, Box<dyn Error>> {
    let parts = parse_csv_f32(value, 4)?;
    Ok(Quat::from_xyzw(parts[0], parts[1], parts[2], parts[3]).normalize())
}

fn parse_object_rotation(value: &str) -> Result<ObjectRotation, Box<dyn Error>> {
    let parts = parse_csv_f64(value, 3)?;
    Ok(ObjectRotation::new(parts[0], parts[1], parts[2]))
}

fn parse_csv_f32(value: &str, expected: usize) -> Result<Vec<f32>, Box<dyn Error>> {
    let parts = value
        .split(',')
        .map(|part| part.trim().parse::<f32>())
        .collect::<Result<Vec<_>, _>>()?;
    if parts.len() != expected {
        return Err(format!("expected {expected} comma-separated values in `{value}`").into());
    }
    Ok(parts)
}

fn parse_csv_f64(value: &str, expected: usize) -> Result<Vec<f64>, Box<dyn Error>> {
    let parts = value
        .split(',')
        .map(|part| part.trim().parse::<f64>())
        .collect::<Result<Vec<_>, _>>()?;
    if parts.len() != expected {
        return Err(format!("expected {expected} comma-separated values in `{value}`").into());
    }
    Ok(parts)
}

fn print_usage() {
    eprintln!(
        "Usage: replay_pose --data-dir <YCB_DIR> --object <OBJECT_ID> \\
         --camera-t x,y,z --camera-q-xyzw x,y,z,w [--object-rot pitch,yaw,roll] \\
         [--repeat n] [--path both|one-shot|persistent]"
    );
}
