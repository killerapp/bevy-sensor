# bevy-sensor

A Rust library and CLI for capturing multi-view images (RGBA + Depth) of 3D objects, specifically designed for the [Thousand Brains Project](https://github.com/thousandbrainsproject/tbp.monty) sensor simulation.

This crate serves as the visual sensor module for the [neocortx](https://github.com/killerapp/neocortx) project, providing TBP-compatible sensor data (64x64 resolution, specific camera intrinsics) from YCB dataset models.

## Features

- **TBP-Compatible:** Matches Habitat sensor specifications (resolution, coordinate systems).
- **Multi-View:** Captures objects from spherical viewpoints (yaw/pitch).
- **YCB Integration:** Auto-downloads and caches [YCB Benchmark](https://www.ycbbenchmarks.com/) models.
- **Headless:** Optimized for headless rendering on Linux and WSL2 (via WebGPU).

## Requirements

- **Rust:** 1.70+
- **Bevy:** 0.15+
- **System:** Linux with Vulkan drivers (or WSL2).
- **Tools:** `just` (recommended command runner).

## Quick Start

1.  **Install Just** (Optional but recommended):
    ```bash
    cargo install just
    ```

2.  **Run a Test Render:**
    ```bash
    just render-single 003_cracker_box
    # Models will be automatically downloaded to /tmp/ycb if missing.
    # To use a custom location: cargo run --bin prerender -- --data-dir ./my_models ...
    # Output saved to test_fixtures/renders/
    ```

## Usage

### CLI (Batch Rendering)

Render the standard TBP benchmark set (10 objects):
```bash
just render-tbp-benchmark
```

Render specific objects:
```bash
just render-batch "003_cracker_box,005_tomato_soup_can"
```

### Library (Rust)

Add to your `Cargo.toml`:
```toml
[dependencies]
bevy-sensor = "0.4"
```

Use in your code:
```rust
use bevy_sensor::{render_to_buffer, RenderConfig, ViewpointConfig, ObjectRotation};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Configure
    let config = RenderConfig::tbp_default(); // 64x64, TBP intrinsics
    let viewpoint = bevy_sensor::generate_viewpoints(&ViewpointConfig::default())[0];
    let rotation = ObjectRotation::identity();
    let object_path = Path::new("/tmp/ycb/003_cracker_box");

    // 2. Render to memory (RGBA + Depth)
    let output = render_to_buffer(object_path, &viewpoint, &rotation, &config)?;
    
    println!("Captured {}x{} image", output.width, output.height);
    Ok(())
}
```

## Troubleshooting

### WSL2 Support
WSL2 does not support native Vulkan window surfaces well. This project defaults to the **WebGPU** backend on WSL2, which works reliably for headless rendering.
*   **Fix:** Ensure you have up-to-date GPU drivers on Windows.

### Software Rendering (No GPU)
If you absolutely have no GPU, you can try software rendering (slow, potential artifacts):
```bash
LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe cargo run --release
```

## License

MIT