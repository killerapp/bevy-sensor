//! Test the render_to_buffer API with actual YCB models.
//!
//! Run with: cargo run --example test_render

use bevy_sensor::ycb::{download_models, models_exist, Subset};
use bevy_sensor::{
    generate_viewpoints, render_to_buffer, ObjectRotation, RenderConfig, ViewpointConfig,
};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let ycb_dir = "/tmp/ycb";

    // Download if needed
    if !models_exist(ycb_dir) {
        println!("Downloading YCB models (representative subset)...");
        download_models(ycb_dir, Subset::Representative).await?;
    } else {
        println!("YCB models already present at {}", ycb_dir);
    }

    // Test render
    let object_dir = Path::new(ycb_dir).join("003_cracker_box");
    println!("Object dir: {:?}", object_dir);

    let config = RenderConfig::tbp_default();
    let viewpoints = generate_viewpoints(&ViewpointConfig::default());
    let rotation = ObjectRotation::identity();

    println!(
        "Testing render_to_buffer with {}x{} resolution...",
        config.width, config.height
    );
    let output = render_to_buffer(&object_dir, &viewpoints[0], &rotation, &config)?;

    println!(
        "Success! Got {}x{} image with {} RGBA bytes, {} depth values",
        output.width,
        output.height,
        output.rgba.len(),
        output.depth.len()
    );

    // Check if we got real data (not all gray placeholder)
    let unique_colors: std::collections::HashSet<_> =
        output.rgba.chunks(4).map(|c| (c[0], c[1], c[2])).collect();
    println!("Unique colors in image: {}", unique_colors.len());

    Ok(())
}
