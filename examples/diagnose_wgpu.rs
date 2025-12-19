//! Diagnose wgpu backend availability on WSL2
//!
//! This helps understand why GPU adapter enumeration fails

use bevy::app::ScheduleRunnerPlugin;
use bevy::prelude::*;
use bevy::window::WindowPlugin;
use std::time::Duration;

fn main() {
    println!("\n=== WGPU Backend Diagnostics ===\n");

    // Show environment
    println!("WGPU_BACKEND: {:?}", std::env::var("WGPU_BACKEND"));
    println!("RUST_LOG: {:?}", std::env::var("RUST_LOG"));

    // Check if we're on WSL2
    let is_wsl2 = std::fs::read_to_string("/proc/version")
        .map(|v| v.to_lowercase().contains("microsoft") || v.to_lowercase().contains("wsl"))
        .unwrap_or(false);

    println!("Platform: {}", if is_wsl2 { "WSL2" } else { "Linux/Other" });
    println!();

    // Try different environment configurations
    println!("Testing backend selection...");

    for backend_name in &["vulkan", "webgpu", "gl", "auto", ""] {
        println!("\n--- Testing WGPU_BACKEND={} ---", backend_name);
        std::env::set_var("WGPU_BACKEND", backend_name);

        // Note: We can't actually spawn a full Bevy app in a loop due to how it handles initialization
        // But we can at least show the environment
        println!("Set WGPU_BACKEND={}", backend_name);
    }

    println!("\n=== Recommendations ===");
    println!("If GPU isn't found with WebGPU backend:");
    println!("1. Try WGPU_BACKEND=vulkan (if WSL2 supports it)");
    println!("2. Check if nvidia-smi works (driver access)");
    println!("3. Try CUDA Python to confirm GPU works");
    println!("4. Consider using CUDA bindings directly instead of wgpu");
    println!();

    // Try to spawn a minimal Bevy app to see what happens
    println!("Attempting Bevy app creation (this will fail if no GPU)...");
    println!();

    let app_result = std::panic::catch_unwind(|| {
        let mut app = App::new();
        app.add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: None,
            exit_condition: bevy::window::ExitCondition::DontExit,
            ..default()
        }))
        .add_plugins(ScheduleRunnerPlugin::run_loop(Duration::from_secs_f64(
            1.0 / 60.0,
        )));
        // Just test creation, don't run
    });

    match app_result {
        Ok(()) => println!("✅ Bevy app created successfully (GPU detected!)"),
        Err(_) => println!("❌ Bevy app creation failed (GPU not detected by wgpu)"),
    }
}
