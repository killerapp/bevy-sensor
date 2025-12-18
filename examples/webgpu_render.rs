//! Example: WebGPU backend rendering for cross-platform compatibility
//!
//! This example demonstrates how to explicitly select and use the WebGPU backend
//! for rendering. This is particularly useful for:
//! - WSL2 environments (where Vulkan window surfaces don't work)
//! - Restricted environments without native GPU driver support
//! - Testing backend compatibility across platforms
//!
//! The WebGPU backend works via fallback to OpenGL/WebGL2 on systems without
//! native WebGPU drivers, making it highly portable.
//!
//! Usage:
//! ```sh
//! # Use automatic backend selection (default)
//! cargo run --example webgpu_render --release
//!
//! # Force WebGPU backend explicitly
//! export WGPU_BACKEND=webgpu
//! cargo run --example webgpu_render --release
//!
//! # Enable debug logging to see which backend is selected
//! export WGPU_LOG=warn
//! cargo run --example webgpu_render --release
//! ```

use bevy_sensor::{
    backend::{BackendConfig, detect_platform},
    render_to_buffer, RenderConfig, ViewpointConfig, ObjectRotation,
};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("WebGPU Backend Rendering Example");
    println!("================================\n");

    // Show detected platform
    let platform = detect_platform();
    println!("Detected platform: {:?}\n", platform);

    // Configure backend based on environment/platform
    let backend_config = BackendConfig::new();
    let selected_backend = backend_config.selected_backend();

    println!("Backend configuration:");
    println!("  Selected: {}", selected_backend.name());
    println!("  Fallbacks: {:?}",
        backend_config.fallbacks.iter().map(|b| b.name()).collect::<Vec<_>>()
    );
    println!("  Force headless: {}", backend_config.force_headless);
    println!();

    // Apply backend configuration to environment
    backend_config.apply_env();

    // Example: Force WebGPU for explicit testing
    println!("Note: To force WebGPU backend, use:");
    println!("  export WGPU_BACKEND=webgpu\n");

    // Check if object directory exists
    let object_dir = PathBuf::from("/tmp/ycb/003_cracker_box");
    if !object_dir.exists() {
        eprintln!("Object directory not found: {:?}", object_dir);
        eprintln!("Please download YCB dataset first:");
        eprintln!("  cargo run --example test_render");
        return Err("Object directory not found".into());
    }

    // Render configuration
    let render_config = RenderConfig::tbp_default();
    let viewpoint_config = ViewpointConfig::default();
    let viewpoints = bevy_sensor::generate_viewpoints(&viewpoint_config);

    println!("Rendering configuration:");
    println!("  Resolution: {}x{}", render_config.width, render_config.height);
    println!("  Viewpoints: {}", viewpoints.len());
    println!("  Object: 003_cracker_box\n");

    // Render single viewpoint
    println!("Rendering with {} backend...", selected_backend.name());
    let output = render_to_buffer(
        &object_dir,
        &viewpoints[0],
        &ObjectRotation::identity(),
        &render_config,
    )?;

    println!("✓ Render successful!");
    println!();
    println!("Output information:");
    println!("  Image size: {}x{} pixels", output.width, output.height);
    println!("  RGBA data: {} bytes", output.rgba.len());
    println!("  Depth data: {} values ({} bytes)",
        output.depth.len(),
        output.depth.len() * 8
    );

    // Show camera intrinsics (high precision with f64)
    let intrinsics = &output.intrinsics;
    println!("\nCamera intrinsics:");
    println!("  Focal length: [{:.2}, {:.2}]",
        intrinsics.focal_length[0], intrinsics.focal_length[1]
    );
    println!("  Principal point: [{:.2}, {:.2}]",
        intrinsics.principal_point[0], intrinsics.principal_point[1]
    );

    // Example: Convert to image formats
    let rgb_image = output.to_rgb_image();
    let depth_image = output.to_depth_image();
    println!("\nImage format conversions:");
    println!("  RGB image: {}x{} pixels", rgb_image.len(), rgb_image[0].len());
    println!("  Depth image: {}x{} floats", depth_image.len(), depth_image[0].len());

    println!("\n✓ WebGPU rendering example complete!");
    println!("\nTo render multiple viewpoints efficiently, see examples/batch_render_neocortx.rs");

    Ok(())
}
