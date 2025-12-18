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
    backend::detect_platform, cache::ModelCache, render_to_buffer, render_to_buffer_cached,
    ObjectRotation, RenderConfig, RenderOutput, ViewpointConfig,
};
use std::fs;
use std::path::PathBuf;

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
