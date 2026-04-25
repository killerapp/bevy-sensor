//! Example: Per-step persistent rendering for surface-policy feedback loops.
//!
//! Closes issue #65. Demonstrates the `PersistentRenderer` API and benchmarks
//! it against the two existing rendering paths to give a concrete speedup
//! number for downstream consumers (neocortx surface-policy port).
//!
//! Usage:
//! ```sh
//! cargo run --example persistent_per_step --release
//! cargo run --example persistent_per_step --release -- --steps 100 --object 003_cracker_box
//! ```
//!
//! Compares three paths over `STEPS` single-viewpoint renders against one
//! object:
//!
//!   1. `render_to_buffer` per call          — fresh App every time (worst).
//!   2. Fresh `RenderSession` per call       — current best for one-off shape.
//!   3. `PersistentRenderer::render` per call — scene held loaded (proposed).
//!
//! For a surface-policy episode (50–200 motor steps × 30 episodes × 10 objs),
//! the per-step path dominates wall-clock — this is the case #65 was filed for.

use bevy_sensor::{
    batch::BatchRenderRequest, generate_viewpoints, render_to_buffer, BatchRenderConfig,
    ObjectRotation, PersistentRenderer, RenderConfig, RenderSession, ViewpointConfig,
};
use std::path::PathBuf;
use std::time::Instant;

fn parse_args() -> (usize, String) {
    let mut steps = 50_usize;
    let mut object = "003_cracker_box".to_string();
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--steps" => {
                if let Some(v) = args.next() {
                    steps = v.parse().unwrap_or(steps);
                }
            }
            "--object" => {
                if let Some(v) = args.next() {
                    object = v;
                }
            }
            _ => {}
        }
    }
    (steps, object)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (steps, object_id) = parse_args();
    let object_dir = PathBuf::from(format!("/tmp/ycb/{object_id}"));
    let render_config = RenderConfig::tbp_default();

    println!("PersistentRenderer per-step throughput bench (issue #65)");
    println!("=========================================================");
    println!("  Object:     {object_id}");
    println!(
        "  Resolution: {}x{}",
        render_config.width, render_config.height
    );
    println!("  Steps:      {steps}");
    println!();

    if !object_dir.exists() {
        eprintln!("Object directory not found: {object_dir:?}");
        eprintln!(
            "Run: bevy_sensor::ycb::download_models(\"/tmp/ycb\", Subset::Representative).await?;"
        );
        return Err("Object directory not found".into());
    }

    let viewpoint_config = ViewpointConfig {
        radius: 0.5,
        yaw_count: 8,
        pitch_angles_deg: vec![-30.0, 0.0, 30.0],
    };
    let viewpoints = generate_viewpoints(&viewpoint_config);
    let rotation = ObjectRotation::identity();

    let viewpoint_for = |i: usize| viewpoints[i % viewpoints.len()];

    // ---- Baseline 1: render_to_buffer per call (fresh App each time) ----
    println!("[1/3] render_to_buffer per call (fresh App per step)");
    let t0 = Instant::now();
    for i in 0..steps {
        let vp = viewpoint_for(i);
        let _ = render_to_buffer(&object_dir, &vp, &rotation, &render_config)?;
    }
    let baseline = t0.elapsed();
    let baseline_per_step = baseline.as_secs_f64() * 1000.0 / steps as f64;
    println!(
        "      total {:.1} s, per-step {:.1} ms",
        baseline.as_secs_f64(),
        baseline_per_step
    );

    // ---- Baseline 2: fresh RenderSession per call ----
    println!("[2/3] fresh RenderSession per call");
    let t1 = Instant::now();
    for i in 0..steps {
        let vp = viewpoint_for(i);
        let mut session = RenderSession::new(&render_config)?;
        let _ = session.render(&[BatchRenderRequest {
            object_dir: object_dir.clone(),
            viewpoint: vp,
            object_rotation: rotation.clone(),
            render_config: render_config.clone(),
        }])?;
    }
    let session_per_call = t1.elapsed();
    let session_per_step = session_per_call.as_secs_f64() * 1000.0 / steps as f64;
    println!(
        "      total {:.1} s, per-step {:.1} ms",
        session_per_call.as_secs_f64(),
        session_per_step
    );

    // ---- The new path: PersistentRenderer ----
    println!("[3/3] PersistentRenderer (scene held loaded)");
    let t2 = Instant::now();
    let mut renderer = PersistentRenderer::new(&object_dir, &render_config)?;
    let init = t2.elapsed();
    let t3 = Instant::now();
    for i in 0..steps {
        let vp = viewpoint_for(i);
        let _ = renderer.render(&vp, &rotation)?;
    }
    let persistent_steps = t3.elapsed();
    let persistent_per_step = persistent_steps.as_secs_f64() * 1000.0 / steps as f64;
    println!(
        "      init {:.0} ms, total {:.1} s, per-step {:.1} ms",
        init.as_secs_f64() * 1000.0,
        persistent_steps.as_secs_f64(),
        persistent_per_step
    );

    // Touch the API to keep symbol used and demonstrate explicit teardown.
    let _ = BatchRenderConfig::default();
    renderer.close();

    println!();
    println!(
        "Speedup vs render_to_buffer per call: {:.2}x",
        baseline_per_step / persistent_per_step
    );
    println!(
        "Speedup vs fresh RenderSession per call: {:.2}x",
        session_per_step / persistent_per_step
    );

    Ok(())
}
