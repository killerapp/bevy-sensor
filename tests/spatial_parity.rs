//! Spatial parity tests: pixel -> world-point unprojection across the Bevy
//! migration.
//!
//! # Why this file exists
//!
//! The neocortx ycb parity-gate cratered on the Bevy 0.18 RC (ycb-77
//! 78.63% -> 2.99%) even though render buffers populate and features extract.
//! See neocortx#390 / bevy-sensor#92. The failure is *spatial*: something in the
//! 0.18 render path (camera projection / depth linearization / image
//! Y-orientation / row order) moved where a pixel unprojects to in world space.
//!
//! The existing GPU tests assert RGBA byte-equality and f32 depth-parity
//! *across render paths within one Bevy version*. They compare raw buffers, so
//! they cannot catch a spatial regression: a vertically-flipped or depth-rescaled
//! buffer is still "valid" by those tests while every unprojected world point is
//! wrong.
//!
//! This file closes that gap. It checks the actual quantity neocortx consumes:
//! `RenderOutput::pixel_surface_point_world` (pixel + depth -> world point).
//!
//! ## Layer 1 — `unproject_*` (runs in CI, no GPU)
//!
//! Pins the pixel->world *convention* on a synthetic, hand-built `RenderOutput`.
//! Version-independent: documents and guards the image-y-down -> world-up mapping
//! that 0.5.6 satisfies and that neocortx depends on.
//!
//! ## Layer 2 — `gpu_spatial_parity` (`#[ignore]`, needs GPU + YCB models)
//!
//! Renders a known object/viewpoint on the *current* Bevy build and asserts
//! geometric invariants that any correct render (0.5.6 or a fixed 0.18) must
//! satisfy, then compares the center-pixel world point against an optional 0.5.6
//! golden. Run with:
//!
//! ```sh
//! cargo test --test spatial_parity -- --ignored --nocapture
//! ```

use bevy_sensor::{
    render_to_buffer, CameraIntrinsics, ObjectRotation, RenderConfig, RenderOutput,
    TargetingPolicy, Transform, Vec3, ViewpointConfig,
};
use std::path::PathBuf;

// ===========================================================================
// Layer 1: pixel -> world convention (synthetic, runs in CI)
// ===========================================================================

/// Build a synthetic `RenderOutput` with a uniform foreground depth so the
/// pixel->world contract can be checked without a GPU.
fn synthetic_output(camera_transform: Transform, depth_m: f64) -> RenderOutput {
    let width = 64u32;
    let height = 64u32;
    let intrinsics = CameraIntrinsics {
        focal_length: [100.0, 100.0],
        principal_point: [width as f64 / 2.0, height as f64 / 2.0],
        image_size: [width, height],
    };
    RenderOutput {
        rgba: vec![255u8; (width * height * 4) as usize],
        depth: vec![depth_m; (width * height) as usize],
        width,
        height,
        intrinsics,
        camera_transform,
        object_rotation: ObjectRotation::identity(),
        target_point: Vec3::ZERO,
        targeting_policy: TargetingPolicy::Origin,
    }
}

fn v(p: [f64; 3]) -> Vec3 {
    Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32)
}

/// Center pixel unprojects onto the camera->target ray, `depth` meters in front
/// of the camera. For the principal-point pixel the result is independent of
/// focal length, so this isolates the camera transform + depth path.
#[test]
fn unproject_center_pixel_lands_on_forward_ray() {
    // Camera 2 m back on +Z, looking at the origin. forward = -Z, up = +Y.
    let camera = Transform::from_xyz(0.0, 0.0, 2.0).looking_at(Vec3::ZERO, Vec3::Y);
    let depth = 1.5_f64;
    let output = synthetic_output(camera, depth);

    let center = output.center_pixel().expect("center pixel");
    assert_eq!(center, [32, 32]);

    let world = v(output
        .pixel_surface_point_world(center)
        .expect("center pixel is foreground"));

    // cam_pos + forward * depth = (0,0,2) + (0,0,-1)*1.5 = (0,0,0.5)
    let expected = Vec3::new(0.0, 0.0, 0.5);
    assert!(
        (world - expected).length() < 1e-4,
        "center world point {world:?} != expected {expected:?}"
    );
}

/// A pixel to the *right* of center unprojects to a point on the camera's +right
/// axis; a pixel *above* center unprojects to a point on the camera's +up axis.
///
/// This is the convention a vertical flip (image row order) or a horizontal
/// mirror would violate: it pins image-y-down -> world-up and image-x-right ->
/// world-right for an identity-oriented camera.
#[test]
fn unproject_offcenter_pixels_respect_image_axes() {
    let camera = Transform::from_xyz(0.0, 0.0, 2.0).looking_at(Vec3::ZERO, Vec3::Y);
    let output = synthetic_output(camera, 1.5);

    let center = v(output.pixel_surface_point_world([32, 32]).unwrap());
    let right = v(output.pixel_surface_point_world([42, 32]).unwrap());
    let above = v(output.pixel_surface_point_world([32, 22]).unwrap());

    // Pixel +x (to the right in the image) -> larger world X (camera right = +X).
    assert!(
        right.x > center.x + 0.05,
        "pixel right of center must map to world +right; center.x={}, right.x={}",
        center.x,
        right.x
    );
    assert!(
        (right.y - center.y).abs() < 1e-4 && (right.z - center.z).abs() < 1e-4,
        "horizontal probe drifted off the row: {right:?} vs {center:?}"
    );

    // Pixel -y (above center in the image) -> larger world Y (camera up = +Y).
    // A vertically flipped buffer would invert this inequality.
    assert!(
        above.y > center.y + 0.05,
        "pixel above center must map to world +up; center.y={}, above.y={}",
        center.y,
        above.y
    );
    assert!(
        (above.x - center.x).abs() < 1e-4 && (above.z - center.z).abs() < 1e-4,
        "vertical probe drifted off the column: {above:?} vs {center:?}"
    );
}

// ===========================================================================
// Layer 2: GPU spatial parity against the live render path (#[ignore])
// ===========================================================================

const OBJECT_DIR: &str = "/tmp/ycb/003_cracker_box";

/// Render the default first viewpoint of 003_cracker_box at TBP 64x64 and assert
/// spatial invariants on the unprojected surface points, then compare the
/// center-pixel world point against a 0.5.6 golden if one is provided via
/// `BEVY_SENSOR_SPATIAL_GOLDEN` (path to a JSON written by `print_*` below).
#[test]
#[ignore] // GPU + YCB models required; run with --ignored
fn gpu_spatial_parity_center_pixel_unprojects_near_object_center() {
    let object_dir = PathBuf::from(OBJECT_DIR);
    if !object_dir.join("google_16k/textured.obj").exists() {
        eprintln!("skip: YCB model not found at {}", object_dir.display());
        return;
    }

    let viewpoints = bevy_sensor::generate_viewpoints(&ViewpointConfig::default());
    assert!(!viewpoints.is_empty());
    let viewpoint = viewpoints[0];
    let config = RenderConfig::tbp_default();

    let output = render_to_buffer(&object_dir, &viewpoint, &ObjectRotation::identity(), &config)
        .expect("GPU render failed");

    // The default (non-targeted) viewpoints orbit and look at the world origin.
    let target = Vec3::ZERO;
    let cam_pos = output.camera_transform.translation;
    let to_target = (target - cam_pos).normalize();
    let cam_to_target_dist = (target - cam_pos).length();

    // --- diagnostics: always emit, so a failing run hands neocortx real numbers
    let health = output.health();
    println!("== gpu_spatial_parity (003_cracker_box, viewpoint 0) ==");
    println!("camera_pos       = {cam_pos:?}");
    println!("camera_to_target = {cam_to_target_dist:.6} m");
    println!(
        "foreground       = {}/{} px ({:.1}%)",
        health.foreground_pixel_count,
        output.width * output.height,
        health.foreground_coverage * 100.0
    );
    println!("center_pixel     = {:?}", output.center_pixel());
    println!("center_raw_depth = {:?}", output.center_pixel_raw_depth());
    print_depth_orientation(&output);
    let stats = depth_stats(&output);
    println!(
        "depth stats      = min {:.5} max {:.5} spread {:.5} distinct {} bg(far) {}",
        stats.min, stats.max, stats.max - stats.min, stats.distinct, stats.far_count
    );
    println!("rgba distinct colors = {}", distinct_colors(&output));

    // --- Invariant 1: center pixel sees the object surface.
    let center_depth = output
        .center_pixel_depth()
        .expect("center pixel must hit the object surface (foreground depth)");
    assert!(
        center_depth > 0.05 && center_depth < cam_to_target_dist as f64 + 0.05,
        "center depth {center_depth} m implausible for camera at {cam_to_target_dist} m"
    );

    // --- Invariant 1b: the depth buffer must encode real 3D geometry, not a
    // flat plane. A uniform depth (every pixel at the pivot distance) extracts
    // features and passes buffer-equality tests, but unprojects to a flat sheet
    // -> neocortx spatial matching collapses (neocortx#390). A real object face
    // spans several mm of depth across the frame.
    assert!(
        stats.max - stats.min > 5e-3,
        "degenerate depth buffer: spread {:.6} m across the object (min {:.5}, max {:.5}, \
         distinct {}). A flat depth plane unprojects to a sheet and breaks spatial matching.",
        stats.max - stats.min,
        stats.min,
        stats.max,
        stats.distinct
    );

    // --- Invariant 2: center surface point lies on the camera->target ray, in
    // front of the camera, and STRICTLY between the camera and the target.
    //
    // We deliberately do NOT assert the point is near the world origin: YCB mesh
    // origins are offset from their geometric center, so the visible surface
    // along the center ray legitimately sits some distance ahead of the target.
    // The strict "in front of the target" bound is what catches the camera-
    // distance fallback (which sets every depth = |camera|, placing the point
    // exactly AT the target -> along-ray depth == camera-to-target distance).
    let center_world = v(output
        .center_surface_point_world()
        .expect("center surface point"));
    println!(
        "center_world     = {center_world:?}  ({:.6} m from target)",
        (center_world - target).length()
    );

    let to_point = (center_world - cam_pos).normalize();
    let cos = to_point.dot(to_target);
    assert!(
        cos > 0.999,
        "center surface point off the camera->target ray: cos(angle)={cos:.6}"
    );
    let depth_along = (center_world - cam_pos).dot(to_target);
    assert!(
        depth_along > 0.05 && depth_along < cam_to_target_dist - 0.02,
        "center surface point not strictly between camera and target: along-ray depth \
         {depth_along:.4} m (camera->target {cam_to_target_dist:.4} m). A depth == camera-distance \
         fallback would land exactly at the target ({cam_to_target_dist:.4})."
    );

    // --- Optional: strict numeric comparison against a captured 0.5.6 golden.
    match load_golden() {
        Some(golden) => {
            let gw = Vec3::new(golden.center_world[0], golden.center_world[1], golden.center_world[2]);
            let delta = (center_world - gw).length();
            println!("golden_world     = {gw:?}  (delta = {delta:.6} m)");
            assert!(
                delta < 1e-3,
                "center world point diverged from 0.5.6 golden by {delta:.6} m \
                 (0.18 {center_world:?} vs 0.5.6 {gw:?})"
            );
            assert!(
                (center_depth - golden.center_depth).abs() < 1e-3,
                "center depth diverged from 0.5.6 golden: 0.18 {center_depth} vs 0.5.6 {}",
                golden.center_depth
            );
        }
        None => {
            println!(
                "no 0.5.6 golden loaded (set BEVY_SENSOR_SPATIAL_GOLDEN=<path>). \
                 Geometric invariants checked; strict numeric parity skipped."
            );
            write_current_for_capture(&output, center_depth, center_world);
        }
    }
}

/// Print mean foreground depth for the top/middle/bottom and left/middle/right
/// thirds of the image. A vertical flip swaps top<->bottom means; this is the
/// signature to diff against a 0.5.6 capture.
fn print_depth_orientation(output: &RenderOutput) {
    let far = RenderOutput::TBP_FAR_PLANE_METERS;
    let (w, h) = (output.width, output.height);
    let mut rows = [(0.0f64, 0usize); 3];
    let mut cols = [(0.0f64, 0usize); 3];
    for y in 0..h {
        for x in 0..w {
            let Some(d) = output.get_depth(x, y) else { continue };
            if !RenderOutput::is_foreground_depth(d, far) {
                continue;
            }
            let rb = (y * 3 / h).min(2) as usize;
            let cb = (x * 3 / w).min(2) as usize;
            rows[rb].0 += d;
            rows[rb].1 += 1;
            cols[cb].0 += d;
            cols[cb].1 += 1;
        }
    }
    let mean = |(s, n): (f64, usize)| if n > 0 { s / n as f64 } else { f64::NAN };
    println!(
        "depth thirds rows[top,mid,bot] = [{:.4}, {:.4}, {:.4}]",
        mean(rows[0]),
        mean(rows[1]),
        mean(rows[2])
    );
    println!(
        "depth thirds cols[lft,mid,rgt] = [{:.4}, {:.4}, {:.4}]",
        mean(cols[0]),
        mean(cols[1]),
        mean(cols[2])
    );
}

struct DepthStats {
    min: f64,
    max: f64,
    distinct: usize,
    far_count: usize,
}

/// Min/max/distinct over foreground depth + count of far-plane (background)
/// pixels. Distinguishes a real render from a flat/constant depth buffer.
fn depth_stats(output: &RenderOutput) -> DepthStats {
    let far = RenderOutput::TBP_FAR_PLANE_METERS;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut far_count = 0usize;
    let mut seen = std::collections::BTreeSet::new();
    for &d in &output.depth {
        if RenderOutput::is_foreground_depth(d, far) {
            min = min.min(d);
            max = max.max(d);
            // Quantize to 0.1 mm so f32 noise doesn't inflate the distinct count.
            seen.insert((d * 10_000.0).round() as i64);
        } else {
            far_count += 1;
        }
    }
    if !min.is_finite() {
        min = 0.0;
        max = 0.0;
    }
    DepthStats {
        min,
        max,
        distinct: seen.len(),
        far_count,
    }
}

/// Number of distinct RGB triples in the color buffer. A flat/blank render has
/// only a handful; a textured object has many.
fn distinct_colors(output: &RenderOutput) -> usize {
    let mut seen = std::collections::BTreeSet::new();
    for px in output.rgba.chunks_exact(4) {
        seen.insert((px[0], px[1], px[2]));
    }
    seen.len()
}

struct Golden {
    center_world: [f32; 3],
    center_depth: f64,
}

/// Load a 0.5.6 golden from `BEVY_SENSOR_SPATIAL_GOLDEN` (a JSON file written by
/// `write_current_for_capture` on a 0.5.6 build). Hand-parsed to avoid a
/// serde_json dev-dependency just for three numbers.
fn load_golden() -> Option<Golden> {
    let path = std::env::var("BEVY_SENSOR_SPATIAL_GOLDEN").ok()?;
    let text = std::fs::read_to_string(&path).ok()?;
    let nums = |key: &str| -> Vec<f64> {
        let Some(after) = text.split(key).nth(1) else {
            return Vec::new();
        };
        after
            .split(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e' || c == 'E'))
            .filter(|s| !s.is_empty())
            .filter_map(|s| s.parse::<f64>().ok())
            .take(3)
            .collect()
    };
    let cw = nums("\"center_world\"");
    let cd = nums("\"center_depth\"");
    if cw.len() == 3 && !cd.is_empty() {
        Some(Golden {
            center_world: [cw[0] as f32, cw[1] as f32, cw[2] as f32],
            center_depth: cd[0],
        })
    } else {
        None
    }
}

/// Persist the current build's center-pixel result so it can be promoted to a
/// 0.5.6 golden (run this test on a 0.5.6 checkout, then point
/// `BEVY_SENSOR_SPATIAL_GOLDEN` at the file when running on 0.18).
fn write_current_for_capture(output: &RenderOutput, center_depth: f64, center_world: Vec3) {
    let dir = PathBuf::from("test_fixtures/spatial_parity");
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let json = format!(
        "{{\n  \"object\": \"003_cracker_box\",\n  \"viewpoint\": 0,\n  \
         \"resolution\": [{}, {}],\n  \
         \"center_depth\": {},\n  \
         \"center_world\": [{}, {}, {}]\n}}\n",
        output.width, output.height, center_depth, center_world.x, center_world.y, center_world.z
    );
    let path = dir.join("current.json");
    if std::fs::write(&path, json).is_ok() {
        println!("wrote current center-pixel result to {}", path.display());
        println!("  -> on a 0.5.6 checkout this becomes the golden; rename and point");
        println!("     BEVY_SENSOR_SPATIAL_GOLDEN at it when validating 0.18.");
    }
}
