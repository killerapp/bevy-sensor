# bevy-sensor

> **Note:** When using this project as a local dependency (e.g., from `neocortx`), the repository is typically located at `../bevy-sensor`.

## Project Overview

`bevy-sensor` is a Rust library and CLI tool built with the Bevy game engine. It captures multi-view images (RGBA and Depth) of 3D OBJ models (specifically from the YCB dataset) for sensor simulation.

It is designed to be compatible with the **Thousand Brains Project (TBP)** sensor specifications, producing data for the `neocortx` Rust-based implementation. It aligns with TBP's Habitat sensor regarding resolution (64x64), camera intrinsics, and coordinate systems.

## Tech Stack

*   **Language:** Rust
*   **Engine:** Bevy 0.15+ (Rendering, ECS)
*   **Graphics Backend:** WGPU (Vulkan, WebGPU, Metal, DX12)
*   **Data:** YCB Benchmarks (OBJ models)

## Environment Setup

### Prerequisites
1.  **Rust:** 1.70+
2.  **System Dependencies:**
    *   **Linux:** Vulkan drivers, `libx11-dev`, `libasound2-dev`, `libudev-dev`
    *   **WSL2:** Specific configuration required (see Troubleshooting).
3.  **Just:** Command runner (optional but recommended).

### Environment Variables
*   `WGPU_BACKEND`: Controls the rendering backend. Critical for headless environments.
    *   `vulkan` (Native Linux headless)
    *   `webgpu` (WSL2 compatibility)

## Key Commands

Use `just` for simplified command execution. If `just` is not installed, the corresponding `cargo` commands are listed in the `justfile`.

### Building
```bash
just build          # Build debug
just build-release  # Build release
just check          # Check without building
```

### Testing
```bash
just test           # Run all tests (library + doc)
just test-render-integration # Run GPU-dependent integration tests
```

### Running (CLI)
```bash
# Render default CI objects (cracker box, soup can)
# Models will be automatically downloaded to /tmp/ycb if missing
just render-ci

# Render specific objects
just render-batch "003_cracker_box,005_tomato_soup_can"

# Render single viewpoint (good for debugging)
just render-single 003_cracker_box
```

### YCB Dataset Management
```bash
# Models are managed automatically by the CLI tools.
# These commands are available for manual management:

just ycb-download-representative # Download small subset (3 objects)
just ycb-check                   # Verify dataset presence
```

## Architecture & Codebase

*   **`src/lib.rs`**: Main library entry point. Exports the public API used by `neocortx`.
    *   `SensorConfig`, `RenderConfig`, `ObjectRotation`: Core configuration structs.
    *   `render_to_buffer`: Primary API for headless rendering to memory.
*   **`src/render.rs`**: Low-level Bevy app setup for headless rendering. Handles the "one-shot" app lifecycle for capturing a frame.
*   **`src/batch.rs`**: Optimized batch rendering to reduce Bevy initialization overhead.
*   **`src/backend.rs`**: Logic for selecting the appropriate WGPU backend based on platform (Linux vs WSL2 vs macOS).
*   **`src/main.rs`**: CLI binary entry point.
*   **`examples/`**: Demonstration scripts, especially `batch_render_neocortx.rs`.

## Development Conventions

*   **Formatting:** Run `just fmt` (cargo fmt) before committing.
*   **Linting:** Run `just lint` (cargo clippy) to ensure code quality.
*   **Commits:** Follow **Conventional Commits** (e.g., `feat:`, `fix:`, `docs:`) to support `release-plz` automated versioning.

## Troubleshooting

### WSL2 & Headless Rendering
This is the most common issue. Bevy/WGPU can struggle with the virtualized GPU in WSL2.
*   **Symptoms:** Panics about "No Adapter Found" or window creation failures.
*   **Solution:** The project defaults to `WebGPU` backend on WSL2 for better compatibility. Ensure you have recent drivers.
*   **Fallback:** Use pre-rendered fixtures (`just render-to test_fixtures/renders ...`) on a native Linux machine and commit them if WSL2 rendering is strictly impossible.

### Software Rendering
If no GPU is available, you can force software rendering (llvmpipe), but it is slow and may have shader compatibility issues.
```bash
LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe cargo run
```
