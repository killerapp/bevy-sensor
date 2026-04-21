//! Integration test: Hardware rendering sanity check
//!
//! This test actually renders to GPU to verify rendering support on the current platform.
//! It's ignored by default (CI doesn't run it) but can be run locally with:
//!
//! ```sh
//! cargo test -- --ignored --test render_integration -- --nocapture
//! just test-render-integration  # or use justfile command
//! ```
//!
//! Renders are saved to `test_fixtures/test_renders/` so you can inspect output.
//!
//! This validates:
//! - GPU/rendering backend availability
//! - WebGPU backend selection works
//! - Output data format and dimensions are correct
//! - Depth buffer readback works

use bevy_sensor::{
    backend::detect_platform, batch::BatchRenderRequest, cache::ModelCache, render_batch,
    render_to_buffer, render_to_buffer_cached, BatchRenderConfig, ObjectRotation, RenderConfig,
    RenderOutput, RenderSession, ViewpointConfig,
};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

/// Save render output to test_fixtures/test_renders for inspection
fn save_render_output(output: &RenderOutput, name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let render_dir = PathBuf::from("test_fixtures/test_renders");
    fs::create_dir_all(&render_dir)?;

    // Save RGBA as PNG
    let rgb_image = output.to_rgb_image();
    let img = image::ImageBuffer::from_fn(output.width, output.height, |x, y| {
        let rgb = rgb_image[y as usize][x as usize];
        image::Rgb(rgb)
    });
    let rgba_path = render_dir.join(format!("{}.png", name));
    img.save(&rgba_path)?;
    println!("  Saved RGBA: {}", rgba_path.display());

    // Save depth as binary f64
    let depth_path = render_dir.join(format!("{}.depth", name));
    let depth_bytes: Vec<u8> = output.depth.iter().flat_map(|&d| d.to_le_bytes()).collect();
    fs::write(&depth_path, &depth_bytes)?;
    println!("  Saved depth: {}", depth_path.display());

    Ok(())
}

#[test]
#[ignore] // Skip in CI - run manually to verify rendering works
fn test_render_integration() {
    println!("\n=== Render Integration Test ===");
    println!("Platform: {:?}", detect_platform());

    // Check if YCB models exist
    let object_dir = PathBuf::from("/tmp/ycb/003_cracker_box");
    if !object_dir.exists() {
        println!("⚠ YCB models not found at {:?}", object_dir);
        println!("  Skipping render test (models required)");
        println!("  Run: cargo run --example test_render");
        return;
    }

    // Generate viewpoints and select first one
    let viewpoint_config = ViewpointConfig::default();
    let viewpoints = bevy_sensor::generate_viewpoints(&viewpoint_config);
    assert!(!viewpoints.is_empty(), "No viewpoints generated");

    // Render config - TBP default (64x64)
    let config = RenderConfig::tbp_default();
    println!("Rendering {}x{}", config.width, config.height);

    // Actually render to GPU
    let output = render_to_buffer(
        &object_dir,
        &viewpoints[0],
        &ObjectRotation::identity(),
        &config,
    )
    .expect("Render failed - GPU/backend unavailable or rendering not supported");

    // Validate RGBA output
    assert_eq!(output.width, config.width, "Output width mismatch");
    assert_eq!(output.height, config.height, "Output height mismatch");

    let expected_rgba_size = (config.width * config.height * 4) as usize;
    assert_eq!(
        output.rgba.len(),
        expected_rgba_size,
        "RGBA buffer size mismatch: expected {} bytes, got {}",
        expected_rgba_size,
        output.rgba.len()
    );

    // Validate depth output
    let expected_depth_size = (config.width * config.height) as usize;
    assert_eq!(
        output.depth.len(),
        expected_depth_size,
        "Depth buffer size mismatch: expected {} values, got {}",
        expected_depth_size,
        output.depth.len()
    );

    // Sanity check: depth values should be reasonable (between 0.1 and 10 meters typically)
    let mut has_valid_depth = false;
    for &depth in output.depth.iter() {
        if depth > 0.1 && depth < 10.0 {
            has_valid_depth = true;
            break;
        }
    }
    assert!(has_valid_depth, "No valid depth values in output");

    // Sanity check: RGBA should have some non-zero pixels
    let mut has_color = false;
    for chunk in output.rgba.chunks(4) {
        if chunk[0] > 10 || chunk[1] > 10 || chunk[2] > 10 {
            has_color = true;
            break;
        }
    }
    assert!(has_color, "No color data in output");

    // Validate intrinsics (camera calibration)
    let intrinsics = &output.intrinsics;
    assert!(intrinsics.focal_length[0] > 0.0, "Invalid focal length X");
    assert!(intrinsics.focal_length[1] > 0.0, "Invalid focal length Y");
    assert!(
        intrinsics.principal_point[0] >= 0.0,
        "Invalid principal point X"
    );
    assert!(
        intrinsics.principal_point[1] >= 0.0,
        "Invalid principal point Y"
    );

    println!("✓ Render output valid!");
    println!("  RGBA: {} bytes", output.rgba.len());
    println!(
        "  Depth: {} values ({} bytes)",
        output.depth.len(),
        output.depth.len() * 8
    );
    println!(
        "  Focal length: [{:.2}, {:.2}]",
        intrinsics.focal_length[0], intrinsics.focal_length[1]
    );

    // Save render output for inspection
    if let Err(e) = save_render_output(&output, "test_render_basic") {
        println!("⚠ Failed to save render output: {}", e);
    }

    println!("✓ Integration test passed");
}

#[test]
#[ignore] // Skip in CI - run manually on hardware/render-capable environments
fn test_batch_render_matches_sequential_episode_outputs() {
    println!("\n=== Batch vs Sequential Render Test ===");

    let object_dir = PathBuf::from("/tmp/ycb/003_cracker_box");
    if !object_dir.exists() {
        println!("⚠ Skipping - YCB models not found");
        return;
    }

    let viewpoint_config = ViewpointConfig::default();
    let viewpoints = bevy_sensor::generate_viewpoints(&viewpoint_config);
    let selected_viewpoints: Vec<_> = viewpoints.into_iter().take(3).collect();
    let rotation = ObjectRotation::identity();
    let config = RenderConfig::tbp_default();

    let sequential_start = Instant::now();
    let sequential_outputs: Vec<_> = selected_viewpoints
        .iter()
        .map(|viewpoint| {
            render_to_buffer(&object_dir, viewpoint, &rotation, &config)
                .expect("Sequential render failed")
        })
        .collect();
    let sequential_elapsed = sequential_start.elapsed();

    let batch_requests: Vec<_> = selected_viewpoints
        .iter()
        .map(|viewpoint| BatchRenderRequest {
            object_dir: object_dir.clone(),
            viewpoint: *viewpoint,
            object_rotation: rotation.clone(),
            render_config: config.clone(),
        })
        .collect();

    let batch_start = Instant::now();
    let batch_outputs =
        render_batch(batch_requests, &BatchRenderConfig::default()).expect("Batch render failed");
    let batch_elapsed = batch_start.elapsed();

    assert_eq!(batch_outputs.len(), sequential_outputs.len());

    for (idx, (batch_output, sequential_output)) in batch_outputs
        .iter()
        .zip(sequential_outputs.iter())
        .enumerate()
    {
        assert_eq!(batch_output.request.viewpoint, selected_viewpoints[idx]);
        assert_eq!(batch_output.request.object_rotation, rotation);
        assert_eq!(batch_output.width, sequential_output.width);
        assert_eq!(batch_output.height, sequential_output.height);
        assert_eq!(batch_output.intrinsics, sequential_output.intrinsics);
        assert_eq!(batch_output.rgba, sequential_output.rgba);
        assert_eq!(batch_output.depth.len(), sequential_output.depth.len());

        let max_depth_delta = batch_output
            .depth
            .iter()
            .zip(sequential_output.depth.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_depth_delta <= 1e-9,
            "Depth mismatch at viewpoint {idx}: max delta {max_depth_delta}"
        );
    }

    println!(
        "  Sequential: {:.2}s for {} viewpoints",
        sequential_elapsed.as_secs_f64(),
        sequential_outputs.len()
    );
    println!(
        "  Batch: {:.2}s for {} viewpoints",
        batch_elapsed.as_secs_f64(),
        batch_outputs.len()
    );
    println!("✓ Batch and sequential outputs matched");
}

/// Session-vs-fresh N-batch comparison for the RenderSession PSO-cache hypothesis.
///
/// Measures the architectural question: does holding a `RenderSession` across
/// N batches beat constructing a fresh `App` for each of N `render_batch()` calls?
///
/// # Gate calibration
///
/// The authoritative validation is the downstream canary: neocortx's parity-gate
/// run shows **8.85× end-to-end speedup** (153.46 s → 17.35 s) with 100% accuracy
/// on 30 episodes. See PR #58 comments on this branch for full numbers.
///
/// This in-repo test, however, runs in the cargo-test harness and consistently
/// under-reports the real speedup by ~5× relative to the downstream `exp_ycb`
/// binary on the same branch — `render_batch` takes ~518 ms/call in the test
/// process but ~2500 ms/call in the consumer process, likely a process-level
/// effect (global allocator state, wgpu instance affinity, Windows GPU scheduler
/// behavior under a minimal-crate test binary vs. a multi-crate production
/// binary). Root cause is unresolved; tracked as a follow-up after Phase 1
/// lands.
///
/// Consequence: the in-repo gate is set at **≥1.2×** — high enough above 1.0×
/// to catch the class of regression where session state is accidentally
/// destroyed between calls (which would drop speedup toward 1.0×), low enough
/// to clear observed run-to-run jitter (1 of 5 sampled runs on DX12 came in at
/// 1.4× from thermal / driver scheduling noise — see PR #58). The gate exists
/// to detect regressions, not to re-prove the speedup; 8.85× lives in the
/// downstream canary and that's authoritative.
///
/// # Workload
///
/// 5 distinct YCB objects × 24 viewpoints = 120 total renders per path. Using
/// different objects per iteration ensures each batch hits fresh mesh/texture
/// assets, which mirrors the real parity-gate pattern (not the artificial case
/// of re-rendering the same object).
///
/// Requires native GPU (WSL2 cannot run).
#[test]
#[ignore]
fn test_session_vs_fresh_n_batch_smoke() {
    println!("\n=== RenderSession vs fresh N-batch smoke gate ===");

    // Use N DIFFERENT objects so each batch is a cache-miss on the DX12 driver's
    // cross-Device PSO cache — matches the real neocortx parity-gate workload
    // (10 unique objects × 3 rotations = 30 unique PSO keys).
    let object_ids = [
        "003_cracker_box",
        "004_sugar_box",
        "005_tomato_soup_can",
        "006_mustard_bottle",
        "025_mug",
    ];
    let object_dirs: Vec<PathBuf> = object_ids
        .iter()
        .map(|id| PathBuf::from(format!("/tmp/ycb/{id}")))
        .collect();
    if !object_dirs[0].exists() {
        println!("⚠ Skipping - YCB models not found at /tmp/ycb/");
        return;
    }

    let n: usize = object_ids.len();
    let viewpoint_config = ViewpointConfig::default();
    let viewpoints = bevy_sensor::generate_viewpoints(&viewpoint_config);
    let selected: Vec<_> = viewpoints.into_iter().take(24).collect();
    let rotation = ObjectRotation::identity();
    let config = RenderConfig::tbp_default();
    let batch_config = BatchRenderConfig::default();

    // Build one request-list per object.
    let per_object_requests: Vec<Vec<BatchRenderRequest>> = object_dirs
        .iter()
        .map(|object_dir| {
            selected
                .iter()
                .map(|vp| BatchRenderRequest {
                    object_dir: object_dir.clone(),
                    viewpoint: *vp,
                    object_rotation: rotation.clone(),
                    render_config: config.clone(),
                })
                .collect()
        })
        .collect();

    println!(
        "  Workload: {} distinct objects × {} viewpoints = {} total renders per path",
        n,
        selected.len(),
        n * selected.len()
    );

    // Session path: one App, N render() calls across N different objects.
    let session_start = Instant::now();
    let mut session = RenderSession::new(&config).expect("session init failed");
    let session_new_ms = session_start.elapsed().as_secs_f64() * 1000.0;
    let mut session_render_total_ms = 0.0_f64;
    let mut session_per_call_ms: Vec<f64> = Vec::with_capacity(n);
    let mut session_outputs_last = Vec::new();
    for (i, requests) in per_object_requests.iter().enumerate() {
        let t = Instant::now();
        let outs = session
            .render(requests)
            .unwrap_or_else(|e| panic!("session render {i} ({}) failed: {e:?}", object_ids[i]));
        let call_ms = t.elapsed().as_secs_f64() * 1000.0;
        session_per_call_ms.push(call_ms);
        session_render_total_ms += call_ms;
        if i == n - 1 {
            session_outputs_last = outs;
        }
    }
    let session_total_ms = session_new_ms + session_render_total_ms;

    // Fresh path: N independent render_batch() calls across the same N objects.
    let fresh_start = Instant::now();
    let mut fresh_per_call_ms: Vec<f64> = Vec::with_capacity(n);
    let mut fresh_outputs_last = Vec::new();
    for (i, requests) in per_object_requests.iter().enumerate() {
        let t = Instant::now();
        let outs = render_batch(requests.clone(), &batch_config)
            .unwrap_or_else(|e| panic!("fresh render_batch {i} ({}) failed: {e:?}", object_ids[i]));
        let call_ms = t.elapsed().as_secs_f64() * 1000.0;
        fresh_per_call_ms.push(call_ms);
        if i == n - 1 {
            fresh_outputs_last = outs;
        }
    }
    let fresh_total_ms = fresh_start.elapsed().as_secs_f64() * 1000.0;

    // Correctness: both paths must produce byte-identical output on the last iteration.
    assert_eq!(
        session_outputs_last.len(),
        fresh_outputs_last.len(),
        "output count mismatch"
    );
    for (idx, (sess, fresh)) in session_outputs_last
        .iter()
        .zip(fresh_outputs_last.iter())
        .enumerate()
    {
        assert_eq!(sess.width, fresh.width);
        assert_eq!(sess.height, fresh.height);
        assert_eq!(
            sess.rgba, fresh.rgba,
            "RGBA mismatch at viewpoint {idx} between session and fresh"
        );
        let max_depth_delta = sess
            .depth
            .iter()
            .zip(fresh.depth.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_depth_delta <= 1e-9,
            "Depth mismatch at viewpoint {idx}: max delta {max_depth_delta}"
        );
    }

    let speedup = fresh_total_ms / session_total_ms.max(1e-3);

    println!(
        "  Session:   {:.1} ms total (new {:.1} ms + {} × render avg {:.1} ms)",
        session_total_ms,
        session_new_ms,
        n,
        session_render_total_ms / n as f64,
    );
    print!("    per-call: ");
    for (i, ms) in session_per_call_ms.iter().enumerate() {
        print!("[{}]={:.0} ms  ", object_ids[i], ms);
    }
    println!();
    println!(
        "  Fresh:     {:.1} ms total ({} × render_batch avg {:.1} ms)",
        fresh_total_ms,
        n,
        fresh_total_ms / n as f64,
    );
    print!("    per-call: ");
    for (i, ms) in fresh_per_call_ms.iter().enumerate() {
        print!("[{}]={:.0} ms  ", object_ids[i], ms);
    }
    println!();
    println!("  Speedup:   {:.1}×", speedup);

    // Gate is ≥1.2× in the test binary — regression detection only. The
    // downstream canary (neocortx parity-gate) shows 8.85× end-to-end; the
    // test-binary under-reports by ~5× for reasons documented on the docstring.
    // 1.2× is chosen over 1.5× because DX12 thermal / scheduling jitter was
    // observed to dip to 1.4× on 1 of 5 sampled runs on otherwise-correct code.
    assert!(
        speedup >= 1.2,
        "session-vs-fresh speedup was only {:.1}× at N={} × 24 vp \
         (session {:.1} ms, fresh {:.1} ms); in-repo gate requires ≥1.2× \
         (downstream canary shows 8.85× — see #58). Speedup < 1.2× suggests \
         session state is being destroyed between calls.",
        speedup,
        n,
        session_total_ms,
        fresh_total_ms
    );

    println!("✓ RenderSession N-batch smoke gate PASSED");
}

/// Pixel-exact correctness gate for `RenderSession` against the authoritative
/// per-request `render_to_buffer()` path.
///
/// Companion to `test_batch_render_matches_sequential_episode_outputs` (PR #42),
/// but for the persistent-session path. Ensures that:
///
///   1. The per-group scene swap (`render()` call → despawn SessionScene +
///      reset RenderState + spawn new SceneRoot) produces identical output to
///      a freshly-built `App` per viewpoint.
///   2. No state bleeds between `render()` calls on the same session: run two
///      back-to-back calls on different objects, compare each against the
///      sequential reference for that object.
///
/// Failure modes this catches:
///   - Scene-swap leaves stale Mesh3d entities → old object bleeds into new render.
///   - RenderState reset misses a field → capture_ready stays true, capture fires
///     before the new scene is instantiated.
///   - Asset handle not refreshed → old mesh/texture rendered against new rotation.
///
/// Requires native GPU (WSL2 cannot run).
#[test]
#[ignore]
fn test_render_session_matches_sequential_across_objects() {
    println!("\n=== RenderSession vs sequential pixel-exact gate ===");

    let object_ids = ["003_cracker_box", "005_tomato_soup_can"];
    let object_dirs: Vec<PathBuf> = object_ids
        .iter()
        .map(|id| PathBuf::from(format!("/tmp/ycb/{id}")))
        .collect();
    if !object_dirs[0].exists() {
        println!("⚠ Skipping - YCB models not found at /tmp/ycb/");
        return;
    }

    let viewpoint_config = ViewpointConfig::default();
    let viewpoints = bevy_sensor::generate_viewpoints(&viewpoint_config);
    let selected: Vec<_> = viewpoints.into_iter().take(3).collect();
    let rotation = ObjectRotation::identity();
    let config = RenderConfig::tbp_default();

    // Reference: per-request render_to_buffer() for each (object, viewpoint).
    let reference_outputs: Vec<Vec<RenderOutput>> = object_dirs
        .iter()
        .map(|object_dir| {
            selected
                .iter()
                .map(|vp| {
                    render_to_buffer(object_dir, vp, &rotation, &config)
                        .expect("sequential render failed")
                })
                .collect()
        })
        .collect();

    // Session path: single session, two render() calls with different objects.
    let mut session = RenderSession::new(&config).expect("session init failed");
    let session_outputs: Vec<Vec<_>> = object_dirs
        .iter()
        .map(|object_dir| {
            let requests: Vec<_> = selected
                .iter()
                .map(|vp| BatchRenderRequest {
                    object_dir: object_dir.clone(),
                    viewpoint: *vp,
                    object_rotation: rotation.clone(),
                    render_config: config.clone(),
                })
                .collect();
            session.render(&requests).expect("session render failed")
        })
        .collect();

    // Compare pixel-exact + depth-epsilon per (object, viewpoint).
    for (obj_idx, object_id) in object_ids.iter().enumerate() {
        let refs = &reference_outputs[obj_idx];
        let sess = &session_outputs[obj_idx];
        assert_eq!(
            refs.len(),
            sess.len(),
            "output count mismatch for object {object_id}"
        );
        for (vp_idx, (reference, session)) in refs.iter().zip(sess.iter()).enumerate() {
            assert_eq!(session.width, reference.width);
            assert_eq!(session.height, reference.height);
            assert_eq!(session.intrinsics, reference.intrinsics);
            assert_eq!(
                session.rgba, reference.rgba,
                "RGBA mismatch for object {object_id} viewpoint {vp_idx}"
            );
            assert_eq!(session.depth.len(), reference.depth.len());
            let max_depth_delta = session
                .depth
                .iter()
                .zip(reference.depth.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                max_depth_delta <= 1e-9,
                "Depth mismatch for object {object_id} viewpoint {vp_idx}: \
                 max delta {max_depth_delta}"
            );
        }
        println!("  ✓ {object_id}: {} viewpoints pixel-exact", refs.len());
    }

    println!("✓ RenderSession pixel-exact gate PASSED");
}

#[test]
#[ignore] // Skip in CI
fn test_render_multiple_viewpoints() {
    println!("\n=== Multiple Viewpoint Render Test ===");

    let object_dir = PathBuf::from("/tmp/ycb/003_cracker_box");
    if !object_dir.exists() {
        println!("⚠ Skipping - YCB models not found");
        return;
    }

    let viewpoint_config = ViewpointConfig::default();
    let viewpoints = bevy_sensor::generate_viewpoints(&viewpoint_config);
    let config = RenderConfig::tbp_default();

    println!("Rendering {} viewpoints...", viewpoints.len().min(3));

    // Render first 3 viewpoints to verify consistency
    for (i, viewpoint) in viewpoints.iter().take(3).enumerate() {
        let output = render_to_buffer(&object_dir, viewpoint, &ObjectRotation::identity(), &config)
            .expect("Render failed");

        assert_eq!(output.width, config.width);
        assert_eq!(output.height, config.height);
        assert_eq!(
            output.rgba.len(),
            (config.width * config.height * 4) as usize
        );
        assert_eq!(output.depth.len(), (config.width * config.height) as usize);

        // Save each viewpoint
        if let Err(e) = save_render_output(&output, &format!("test_viewpoint_{}", i)) {
            println!("    ⚠ Failed to save: {}", e);
        }

        println!("  ✓ Viewpoint {} rendered successfully", i);
    }

    println!("✓ Multiple viewpoint test passed");
}

#[test]
#[ignore] // Skip in CI
fn test_render_with_rotation() {
    println!("\n=== Render with Object Rotation Test ===");

    let object_dir = PathBuf::from("/tmp/ycb/003_cracker_box");
    if !object_dir.exists() {
        println!("⚠ Skipping - YCB models not found");
        return;
    }

    let viewpoint_config = ViewpointConfig::default();
    let viewpoints = bevy_sensor::generate_viewpoints(&viewpoint_config);
    let config = RenderConfig::tbp_default();

    // Get TBP benchmark rotations (3 rotations)
    let rotations = ObjectRotation::tbp_benchmark_rotations();
    println!("Rendering with {} rotations...", rotations.len());

    for (rot_idx, rotation) in rotations.iter().enumerate() {
        let output = render_to_buffer(&object_dir, &viewpoints[0], rotation, &config)
            .expect("Render with rotation failed");

        assert_eq!(output.width, config.width);
        assert_eq!(output.height, config.height);

        // Save each rotation
        if let Err(e) = save_render_output(&output, &format!("test_rotation_{}", rot_idx)) {
            println!("    ⚠ Failed to save: {}", e);
        }

        println!(
            "  ✓ Rotation {} rendered successfully (yaw: {}°)",
            rot_idx, rotation.yaw
        );
    }

    println!("✓ Rotation test passed");
}

#[test]
#[ignore] // Skip in CI
fn test_render_with_cache() {
    println!("\n=== Render with Cache Test ===");

    let object_dir = PathBuf::from("/tmp/ycb/003_cracker_box");
    if !object_dir.exists() {
        println!("⚠ Skipping - YCB models not found");
        return;
    }

    let viewpoint_config = ViewpointConfig::default();
    let viewpoints = bevy_sensor::generate_viewpoints(&viewpoint_config);
    let config = RenderConfig::tbp_default();

    let mut cache = ModelCache::new();

    // Render first viewpoint with cache
    let output1 = render_to_buffer_cached(
        &object_dir,
        &viewpoints[0],
        &ObjectRotation::identity(),
        &config,
        &mut cache,
    )
    .expect("First cached render failed");

    println!("  ✓ First render completed");
    println!(
        "    Cache stats: {} scenes, {} textures",
        cache.scene_count(),
        cache.texture_count()
    );

    assert_eq!(cache.scene_count(), 1, "Should have 1 cached scene");
    assert_eq!(cache.texture_count(), 1, "Should have 1 cached texture");

    // Render second viewpoint - should use cache
    let output2 = render_to_buffer_cached(
        &object_dir,
        &viewpoints[1],
        &ObjectRotation::identity(),
        &config,
        &mut cache,
    )
    .expect("Second cached render failed");

    println!("  ✓ Second render completed");
    println!(
        "    Cache stats: {} scenes, {} textures",
        cache.scene_count(),
        cache.texture_count()
    );

    // Cache should still have same entries (not duplicated)
    assert_eq!(cache.scene_count(), 1, "Should still have 1 cached scene");
    assert_eq!(
        cache.texture_count(),
        1,
        "Should still have 1 cached texture"
    );

    // Both renders should produce valid output
    assert_eq!(output1.width, config.width);
    assert_eq!(output1.height, config.height);
    assert_eq!(output2.width, config.width);
    assert_eq!(output2.height, config.height);

    // Outputs should be different (different viewpoints)
    assert_ne!(
        output1.rgba, output2.rgba,
        "Different viewpoints should produce different RGBA"
    );

    if let Err(e) = save_render_output(&output1, "test_cache_vp0") {
        println!("    ⚠ Failed to save output1: {}", e);
    }
    if let Err(e) = save_render_output(&output2, "test_cache_vp1") {
        println!("    ⚠ Failed to save output2: {}", e);
    }

    // Test cache clear
    cache.clear();
    assert_eq!(cache.scene_count(), 0, "Cache should be empty after clear");
    assert_eq!(
        cache.texture_count(),
        0,
        "Cache should be empty after clear"
    );

    println!("✓ Cache test passed");
}

#[test]
#[ignore] // Skip in CI
fn test_cache_with_multiple_viewpoints() {
    println!("\n=== Cache with Multiple Viewpoints Test ===");

    let object_dir = PathBuf::from("/tmp/ycb/005_tomato_soup_can");
    if !object_dir.exists() {
        println!("⚠ Skipping - YCB model not found");
        return;
    }

    let viewpoint_config = ViewpointConfig::default();
    let viewpoints = bevy_sensor::generate_viewpoints(&viewpoint_config);
    let config = RenderConfig::tbp_default();

    let mut cache = ModelCache::new();

    // Render 5 viewpoints with cache
    let render_count = 5.min(viewpoints.len());
    println!("Rendering {} viewpoints with cache...", render_count);

    let mut outputs = Vec::new();
    for (i, viewpoint) in viewpoints.iter().take(render_count).enumerate() {
        let output = render_to_buffer_cached(
            &object_dir,
            viewpoint,
            &ObjectRotation::identity(),
            &config,
            &mut cache,
        )
        .expect("Cached render failed");

        outputs.push(output);

        if i == 0 {
            println!(
                "  Initial cache size: {} scenes, {} textures",
                cache.scene_count(),
                cache.texture_count()
            );
        }
    }

    // Cache should only have 1 scene and 1 texture (tomato soup can)
    assert_eq!(cache.scene_count(), 1, "Should cache 1 scene");
    assert_eq!(cache.texture_count(), 1, "Should cache 1 texture");

    // All outputs should be valid
    for (i, output) in outputs.iter().enumerate() {
        assert_eq!(output.width, config.width, "Output {} width mismatch", i);
        assert_eq!(output.height, config.height, "Output {} height mismatch", i);
        assert!(!output.rgba.is_empty(), "Output {} has no RGBA data", i);
        assert!(!output.depth.is_empty(), "Output {} has no depth data", i);
    }

    println!("  ✓ All {} renders successful", render_count);
    println!(
        "  ✓ Cache maintained 1 scene and 1 texture across {} viewpoints",
        render_count
    );
    println!("✓ Multiple viewpoints cache test passed");
}
