//! Diagnose wgpu backend availability across platforms
//!
//! This helps understand GPU detection and backend selection

use bevy::app::ScheduleRunnerPlugin;
use bevy::prelude::*;
use bevy::window::WindowPlugin;
use bevy_sensor::backend::{BackendConfig, RenderBackend};
use std::time::Duration;

fn detect_platform() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        return "Windows";
    }
    #[cfg(target_os = "macos")]
    {
        return "macOS";
    }
    #[cfg(target_os = "linux")]
    {
        // Check for WSL2
        if let Ok(version) = std::fs::read_to_string("/proc/version") {
            let v = version.to_lowercase();
            if v.contains("microsoft") || v.contains("wsl") {
                return "WSL2";
            }
        }
        return "Linux";
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    {
        return "Unknown";
    }
}

fn main() {
    println!("\n=== WGPU Backend Diagnostics ===\n");

    // Show environment
    println!("WGPU_BACKEND: {:?}", std::env::var("WGPU_BACKEND"));
    println!("RUST_LOG: {:?}", std::env::var("RUST_LOG"));

    let platform = detect_platform();
    println!("Platform: {}", platform);
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

    println!("\n=== Platform-Specific Recommendations ===");
    match platform {
        "Windows" => {
            println!("Windows detected. Recommended backends:");
            println!("1. WGPU_BACKEND=dx12 (Direct3D 12 - best for Windows)");
            println!("2. WGPU_BACKEND=vulkan (if Vulkan SDK installed)");
            println!("3. WGPU_BACKEND=gl (OpenGL fallback)");
        }
        "WSL2" => {
            println!("WSL2 detected. Recommended backends:");
            println!("1. WGPU_BACKEND=webgpu (best for WSL2)");
            println!("2. WGPU_BACKEND=gl (OpenGL fallback)");
            println!("Note: Vulkan surfaces don't work in WSL2.");
        }
        "Linux" => {
            println!("Linux detected. Recommended backends:");
            println!("1. WGPU_BACKEND=vulkan (best for native Linux)");
            println!("2. WGPU_BACKEND=gl (OpenGL fallback)");
        }
        "macOS" => {
            println!("macOS detected. Recommended backends:");
            println!("1. Metal is auto-selected (default)");
            println!("2. WGPU_BACKEND=gl (OpenGL fallback)");
        }
        _ => {
            println!("Unknown platform. Try:");
            println!("1. WGPU_BACKEND=auto");
            println!("2. WGPU_BACKEND=gl");
        }
    }
    println!();

    // Initialize backend based on platform
    println!("Initializing backend for {}...", platform);
    let config = BackendConfig::new();
    println!("Selected backend: {:?}", config.selected_backend());
    config.apply_env();
    println!("Applied env: WGPU_BACKEND={:?}", std::env::var("WGPU_BACKEND"));
    println!();

    // Try to spawn a minimal Bevy app to see what happens
    println!("Attempting Bevy app creation...");
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
