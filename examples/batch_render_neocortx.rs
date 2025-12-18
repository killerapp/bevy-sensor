//! Example: Batch rendering for neocortx sensorimotor learning.
//!
//! This example demonstrates the batch rendering API for efficiently rendering
//! multiple viewpoints of YCB objects. The batch API achieves significant speedups
//! by queuing renders and processing them together.
//!
//! Usage:
//! ```sh
//! cargo run --example batch_render_neocortx --release
//! ```

use bevy_sensor::{
    create_batch_renderer, generate_viewpoints, queue_render_request, render_next_in_batch,
    BatchRenderConfig, BatchRenderRequest, ObjectRotation, RenderConfig, RenderStatus,
    ViewpointConfig,
};
use std::path::PathBuf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Batch Rendering Example for neocortx");
    println!("====================================\n");

    // Configuration
    let object_dir = PathBuf::from("/tmp/ycb/003_cracker_box");
    let render_config = RenderConfig::tbp_default();

    // Check if object exists
    if !object_dir.exists() {
        eprintln!("Object directory not found: {:?}", object_dir);
        eprintln!("Please download YCB dataset first:");
        eprintln!(
            "  bevy_sensor::ycb::download_models(\"/tmp/ycb\", Subset::Representative).await?;"
        );
        return Err("Object directory not found".into());
    }

    // Generate viewpoints and rotations (TBP benchmark)
    let viewpoint_config = ViewpointConfig::default();
    let viewpoints = generate_viewpoints(&viewpoint_config);
    let rotations = ObjectRotation::tbp_benchmark_rotations();

    println!(
        "Configuration:\n  Object: 003_cracker_box\n  Resolution: {}x{}\n  Viewpoints: {}\n  Rotations: {}",
        render_config.width,
        render_config.height,
        viewpoints.len(),
        rotations.len()
    );
    println!("  Total renders: {}\n", viewpoints.len() * rotations.len());

    // Create batch renderer
    let batch_config = BatchRenderConfig::default();
    let mut renderer = create_batch_renderer(&batch_config)?;

    println!("Queuing renders...");
    let start_queue = Instant::now();

    // Queue all renders
    let mut total_queued = 0;
    for rotation_idx in 0..rotations.len() {
        for viewpoint_idx in 0..viewpoints.len() {
            queue_render_request(
                &mut renderer,
                BatchRenderRequest {
                    object_dir: object_dir.clone(),
                    viewpoint: viewpoints[viewpoint_idx],
                    object_rotation: rotations[rotation_idx].clone(),
                    render_config: render_config.clone(),
                },
            )?;
            total_queued += 1;
        }
    }

    let queue_time = start_queue.elapsed();
    println!(
        "Queued {} renders in {:.3}s\n",
        total_queued,
        queue_time.as_secs_f64()
    );

    // Execute renders and collect results
    println!("Executing batch renders...");
    let start_render = Instant::now();

    let mut success_count = 0;
    let mut partial_count = 0;
    let mut failure_count = 0;
    let mut max_rgba_size = 0;
    let mut max_depth_size = 0;

    loop {
        match render_next_in_batch(&mut renderer, 500)? {
            Some(output) => {
                // Update statistics
                match output.status {
                    RenderStatus::Success => {
                        success_count += 1;
                    }
                    RenderStatus::PartialFailure => {
                        partial_count += 1;
                    }
                    RenderStatus::Failed => {
                        failure_count += 1;
                    }
                }

                max_rgba_size = max_rgba_size.max(output.rgba.len());
                max_depth_size = max_depth_size.max(output.depth.len());

                // Progress indicator
                let total = success_count + partial_count + failure_count;
                if total % 8 == 0 {
                    print!(".");
                    if total % 72 == 0 {
                        println!(" {} / {}", total, total_queued);
                    }
                }
            }
            None => break, // All renders complete
        }
    }

    let render_time = start_render.elapsed();
    println!("\n\nResults:");
    println!("  Successful: {}", success_count);
    println!("  Partial failures: {}", partial_count);
    println!("  Total failures: {}", failure_count);
    println!("\nPerformance:");
    println!(
        "  Queue time: {:.3}s ({:.2}µs per request)",
        queue_time.as_secs_f64(),
        queue_time.as_micros() as f64 / total_queued as f64
    );
    println!(
        "  Render time: {:.3}s ({:.2}ms per render)",
        render_time.as_secs_f64(),
        render_time.as_millis() as f64 / success_count as f64
    );
    println!(
        "  Total time: {:.3}s\n",
        (queue_time + render_time).as_secs_f64()
    );

    // Data format example
    println!("Output Format Example (first successful render):");
    if success_count > 0 {
        let outputs = renderer.take_completed();
        if let Some(output) = outputs.first() {
            println!("  Image size: {}x{}", output.width, output.height);
            println!("  RGBA data: {} bytes", output.rgba.len());
            println!(
                "  Depth data: {} floats ({} bytes)",
                output.depth.len(),
                output.depth.len() * 8
            );
            println!("  Intrinsics:");
            println!(
                "    Focal length: [{:.2}, {:.2}]",
                output.intrinsics.focal_length[0], output.intrinsics.focal_length[1]
            );
            println!(
                "    Principal point: [{:.2}, {:.2}]",
                output.intrinsics.principal_point[0], output.intrinsics.principal_point[1]
            );

            // Show how to convert to neocortx formats
            println!("\n  neocortx Integration:");
            let rgb_image = output.to_rgb_image();
            let depth_image = output.to_depth_image();
            println!(
                "    RGB image: {}x{} pixels",
                rgb_image.len(),
                rgb_image[0].len()
            );
            println!(
                "    Depth image: {}x{} floats",
                depth_image.len(),
                depth_image[0].len()
            );
            println!("    Sample pixel (0,0) RGB: {:?}", rgb_image[0][0]);
            println!("    Sample pixel (0,0) depth: {:.3}m", depth_image[0][0]);
        }
    }

    println!("\n✓ Batch rendering example complete!");
    Ok(())
}
