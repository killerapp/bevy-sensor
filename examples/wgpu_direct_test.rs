//! Direct wgpu adapter enumeration test - bypasses Bevy
//!
//! This isolates whether the issue is in wgpu or Bevy's use of wgpu

fn main() {
    println!("\n=== Direct WGPU Adapter Test ===\n");

    // Environment
    println!("WGPU_BACKEND: {:?}", std::env::var("WGPU_BACKEND"));

    // Check WSL2
    let is_wsl2 = std::fs::read_to_string("/proc/version")
        .map(|v| v.to_lowercase().contains("microsoft") || v.to_lowercase().contains("wsl"))
        .unwrap_or(false);
    println!("Platform: {}", if is_wsl2 { "WSL2" } else { "Linux/Other" });
    println!();

    // Create wgpu instance with all backends
    println!("Testing with all backends enabled...");
    let backends = wgpu::Backends::all();
    println!("Backends mask: {:?}", backends);

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends,
        ..Default::default()
    });

    // Enumerate adapters
    let adapters = instance.enumerate_adapters(backends);

    println!("\nFound {} adapter(s):", adapters.len());
    for (i, adapter) in adapters.iter().enumerate() {
        let info = adapter.get_info();
        println!("  [{}] {} ({:?})", i, info.name, info.backend);
        println!("      Device type: {:?}", info.device_type);
        println!("      Driver: {}", info.driver);
        println!(
            "      Vendor: 0x{:04x}, Device: 0x{:04x}",
            info.vendor, info.device
        );
    }

    // Always test individual backends to understand what's available
    println!("\n=== Individual Backend Analysis ===");
    for (name, backend) in [
        ("Vulkan", wgpu::Backends::VULKAN),
        ("GL", wgpu::Backends::GL),
        ("DX12", wgpu::Backends::DX12),
        ("Metal", wgpu::Backends::METAL),
        ("WebGPU", wgpu::Backends::BROWSER_WEBGPU),
    ] {
        let inst = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: backend,
            ..Default::default()
        });
        let adapters = inst.enumerate_adapters(backend);
        if adapters.is_empty() {
            println!("  {}: No adapters", name);
        } else {
            for a in adapters.iter() {
                let info = a.get_info();
                println!("  {}: {} ({:?})", name, info.name, info.device_type);
            }
        }
    }
    println!();

    if adapters.is_empty() {
        println!("❌ No GPU adapters found with all backends!");
    } else {
        println!("\n✅ GPU adapter(s) found!");

        // Test EACH adapter
        for (i, adapter) in adapters.iter().enumerate() {
            let info = adapter.get_info();
            println!("\n--- Testing adapter {} ({}) ---", i, info.name);

            let result = pollster::block_on(async {
                adapter
                    .request_device(
                        &wgpu::DeviceDescriptor {
                            label: Some("test device"),
                            required_features: wgpu::Features::empty(),
                            required_limits: wgpu::Limits::default(),
                            ..Default::default()
                        },
                        None,
                    )
                    .await
            });

            match result {
                Ok((device, _queue)) => {
                    println!("✅ Device created successfully!");
                    let limits = device.limits();
                    println!(
                        "   Max texture dimension 2D: {}",
                        limits.max_texture_dimension_2d
                    );
                    println!(
                        "   Max compute workgroups: [{}, {}, {}]",
                        limits.max_compute_workgroup_size_x,
                        limits.max_compute_workgroup_size_y,
                        limits.max_compute_workgroup_size_z
                    );
                    println!(
                        "   Max buffer size: {} MB",
                        limits.max_buffer_size / 1024 / 1024
                    );
                }
                Err(e) => {
                    println!("❌ Device request failed: {:?}", e);
                }
            }
        }

        // Special test: explicitly request GL backend with GPU power preference
        println!("\n=== Testing GL Backend with High Power Preference ===");
        let gl_instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::GL,
            ..Default::default()
        });

        let gl_adapter = pollster::block_on(async {
            gl_instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    force_fallback_adapter: false,
                    compatible_surface: None,
                })
                .await
        });

        match gl_adapter {
            Some(adapter) => {
                let info = adapter.get_info();
                println!("GL adapter found: {} ({:?})", info.name, info.device_type);

                let result = pollster::block_on(async {
                    adapter
                        .request_device(
                            &wgpu::DeviceDescriptor {
                                label: Some("gl device"),
                                required_features: wgpu::Features::empty(),
                                required_limits: wgpu::Limits::default(),
                                ..Default::default()
                            },
                            None,
                        )
                        .await
                });

                match result {
                    Ok((device, _queue)) => {
                        println!("✅ GL Device created successfully!");
                        let limits = device.limits();
                        println!("   Max texture 2D: {}", limits.max_texture_dimension_2d);
                    }
                    Err(e) => {
                        println!("❌ GL Device request failed: {:?}", e);
                    }
                }
            }
            None => println!("❌ No GL adapter found"),
        }
    }

    println!();
}
