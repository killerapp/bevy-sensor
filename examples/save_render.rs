//! Save rendered images to disk.
//!
//! This example renders a YCB object and saves both RGB and depth images.
//! Run with: cargo run --example save_render

use bevy_sensor::ycb::{download_models, models_exist, Subset};
use bevy_sensor::{
    generate_viewpoints, render_to_buffer, ObjectRotation, RenderConfig, ViewpointConfig,
};
use image::{GrayImage, RgbImage};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Initialize backend for proper GPU detection
    bevy_sensor::initialize();

    let ycb_dir = "/tmp/ycb";
    let output_dir = Path::new("output/renders");

    // Create output directory
    std::fs::create_dir_all(output_dir)?;

    // Download if needed
    if !models_exist(ycb_dir) {
        println!("Downloading YCB models (representative subset)...");
        download_models(ycb_dir, Subset::Representative).await?;
    } else {
        println!("YCB models already present at {}", ycb_dir);
    }

    // Render the cracker box
    let object_dir = Path::new(ycb_dir).join("003_cracker_box");
    let config = RenderConfig::tbp_default();
    let viewpoints = generate_viewpoints(&ViewpointConfig::default());
    let rotation = ObjectRotation::identity();

    println!("Rendering {} viewpoints...", viewpoints.len());

    for (i, viewpoint) in viewpoints.iter().enumerate().take(3) {
        println!("  Viewpoint {}/{}...", i + 1, 3);
        let output = render_to_buffer(&object_dir, viewpoint, &rotation, &config)?;

        // Save RGB image
        let rgb_path = output_dir.join(format!("cracker_box_view{:02}_rgb.png", i));
        let rgb_image: RgbImage = RgbImage::from_fn(output.width, output.height, |x, y| {
            let rgba = output.get_rgba(x, y).unwrap_or([0, 0, 0, 255]);
            image::Rgb([rgba[0], rgba[1], rgba[2]])
        });
        rgb_image.save(&rgb_path)?;
        println!("    Saved RGB: {}", rgb_path.display());

        // Save depth image (normalized to grayscale)
        let depth_path = output_dir.join(format!("cracker_box_view{:02}_depth.png", i));
        let depth_min = output.depth.iter().cloned().fold(f64::INFINITY, f64::min);
        let depth_max = output
            .depth
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let depth_range = depth_max - depth_min;

        let depth_image: GrayImage = GrayImage::from_fn(output.width, output.height, |x, y| {
            let depth = output.get_depth(x, y).unwrap_or(depth_max);
            let normalized = if depth_range > 0.0 {
                ((depth - depth_min) / depth_range * 255.0) as u8
            } else {
                128
            };
            image::Luma([normalized])
        });
        depth_image.save(&depth_path)?;
        println!("    Saved depth: {}", depth_path.display());
    }

    println!("\nDone! Check {} for output images.", output_dir.display());

    Ok(())
}
