# CLAUDE.md - Development Notes for bevy-sensor

## Project Overview

A Bevy library and CLI that captures multi-view images of 3D OBJ models (YCB dataset) for sensor simulation. This project produces comparable results to the [Thousand Brains Project (TBP)](https://github.com/thousandbrainsproject/tbp.monty) habitat sensor for use in the neocortx Rust-based implementation.

## Quick Reference

```bash
# Build
cargo build --release

# Run tests (38 tests)
cargo test

# Run (requires GPU or proper software rendering)
cargo run --release

# Headless with software rendering (limited - PBR shaders may fail)
LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe \
  xvfb-run -a -s "-screen 0 1280x1024x24" \
  cargo run --release
```

## Architecture

### Library API (`src/lib.rs`)

The library exports these types for use by neocortx:

```rust
use bevy_sensor::{
    // Headless Rendering API (NEW)
    RenderConfig,        // Render settings (resolution, lighting, depth)
    RenderOutput,        // RGBA + depth buffer output
    LightingConfig,      // Configurable lighting
    CameraIntrinsics,    // Camera parameters for 3D projection
    render_to_buffer,    // Render single viewpoint to memory
    render_all_viewpoints, // Batch render all viewpoints

    // File-based Capture (Legacy)
    SensorConfig,        // Full capture configuration
    ViewpointConfig,     // Camera viewpoint settings
    ObjectRotation,      // Object rotation (Euler angles)
    generate_viewpoints, // Generate camera transforms
    CaptureCamera,       // Marker component for camera
    CaptureTarget,       // Marker component for target object
};

// YCB utilities
use bevy_sensor::ycb::{
    download_models,     // Async download of YCB models
    models_exist,        // Check if models downloaded
    object_mesh_path,    // Get OBJ file path
    object_texture_path, // Get texture path
    Subset,              // Representative, Ten, All
    REPRESENTATIVE_OBJECTS,
    TEN_OBJECTS,
};
```

### Headless Rendering API (NEW)

Render directly to memory for neocortx integration:

```rust
use bevy_sensor::{render_to_buffer, RenderConfig, ViewpointConfig, ObjectRotation};
use std::path::Path;

// TBP-compatible 64x64 sensor
let config = RenderConfig::tbp_default();
let viewpoints = bevy_sensor::generate_viewpoints(&ViewpointConfig::default());

// Render single viewpoint
let output = render_to_buffer(
    Path::new("/tmp/ycb/003_cracker_box"),
    &viewpoints[0],
    &ObjectRotation::identity(),
    &config,
)?;

// Access rendered data
let rgba: &[u8] = &output.rgba;           // 64*64*4 bytes
let depth: &[f32] = &output.depth;        // 64*64 floats (meters)
let rgb_image = output.to_rgb_image();    // Vec<Vec<[u8; 3]>> for neocortx
let depth_image = output.to_depth_image(); // Vec<Vec<f32>> for neocortx
```

**RenderConfig options:**

| Constructor | Resolution | Use case |
|-------------|-----------|----------|
| `RenderConfig::tbp_default()` | 64×64 | TBP-compatible sensor |
| `RenderConfig::preview()` | 256×256 | Debugging/visualization |
| `RenderConfig::high_res()` | 512×512 | Detailed captures |

**LightingConfig options:**

| Constructor | Description |
|-------------|-------------|
| `LightingConfig::default()` | Standard 3-point lighting |
| `LightingConfig::bright()` | High visibility |
| `LightingConfig::soft()` | Soft, even lighting |
| `LightingConfig::unlit()` | Ambient only, no shadows |

**CameraIntrinsics:**

```rust
let intrinsics = config.intrinsics();
// intrinsics.focal_length: [f32; 2]     - (fx, fy) in pixels
// intrinsics.principal_point: [f32; 2]  - (cx, cy) center
// intrinsics.image_size: [u32; 2]       - (width, height)

// Project 3D to 2D
let pixel = intrinsics.project(point_3d);  // Option<[f32; 2]>

// Unproject 2D + depth to 3D
let point = intrinsics.unproject([32.0, 32.0], depth);  // Vec3
```

> **Note:** Current implementation returns placeholder data. Full GPU rendering
> requires a display (X11/Wayland). Use Xvfb for headless servers.

### Object Rotation (`ObjectRotation`)

Matches TBP benchmark Euler angle format `[pitch, yaw, roll]`:

```rust
// TBP benchmark: 3 rotations (used for quick experiments)
ObjectRotation::tbp_benchmark_rotations()
// → [[0,0,0], [0,90,0], [0,180,0]]

// TBP known orientations: 14 rotations (6 faces + 8 corners)
ObjectRotation::tbp_known_orientations()
// → 14 orientations used during TBP training
```

### Viewpoint Generation

Uses **spherical coordinates** matching TBP habitat sensor behavior:

```rust
struct ViewpointConfig {
    radius: f32,              // Distance from object (default: 0.5m)
    yaw_count: usize,         // Horizontal positions (default: 8)
    pitch_angles_deg: Vec<f32>, // Elevations (default: [-30°, 0°, +30°])
}
```

**Spherical to Cartesian conversion (Y-up):**
```
x = radius * cos(pitch) * sin(yaw)
y = radius * sin(pitch)
z = radius * cos(pitch) * cos(yaw)
```

### Capture Configurations

| Config | Rotations | Viewpoints | Total |
|--------|-----------|------------|-------|
| `SensorConfig::default()` | 1 | 24 | 24 |
| `SensorConfig::tbp_benchmark()` | 3 | 24 | 72 |
| `SensorConfig::tbp_full_training()` | 14 | 24 | 336 |

### Capture State Machine (`src/main.rs`)

```
SetupRotation → SetupView → WaitSettle (10 frames) → Capture → WaitSave (200 frames) → loop
```

Output: `capture_{rot}_{view}.png` (e.g., `capture_0_0.png` through `capture_2_23.png`)

## TBP Habitat Sensor Alignment

| TBP Feature | Bevy Implementation |
|-------------|---------------------|
| Quaternion rotation | `ObjectRotation::to_quat()` |
| `look_up` / `look_down` | Pitch angles: -30°, 0°, +30° |
| `turn_left` / `turn_right` | 8 yaw positions @ 45° intervals |
| Object rotations [0,0,0], [0,90,0], [0,180,0] | `ObjectRotation::tbp_benchmark_rotations()` |
| 14 known orientations | `ObjectRotation::tbp_known_orientations()` |
| Spherical radius | Configurable (default 0.5m) |

### TBP Reference Implementation

- **Sensors**: `tbp/monty/simulators/habitat/sensors.py`
  - Uses quaternion (w,x,y,z) format, default identity: (1, 0, 0, 0)
  - Position relative to HabitatAgent

- **Actions**: `tbp/monty/frameworks/actions/actions.py`
  - `LookDown`, `LookUp`: rotation_degrees parameter
  - `TurnLeft`, `TurnRight`: yaw rotation
  - `SetYaw`, `SetAgentPitch`: absolute rotations

- **Benchmarks**:
  - 3 known rotations for quick tests: [0,0,0], [0,90,0], [0,180,0]
  - 14 known orientations for full training (cube faces + corners)
  - 10 random rotations for generalization testing

## YCB Dataset Setup

**Programmatic download (via ycbust library):**

```rust
use bevy_sensor::ycb::{download_models, Subset, models_exist};

// Check if models already exist
if !models_exist("/tmp/ycb") {
    // Download representative subset (3 objects) - async
    download_models("/tmp/ycb", Subset::Representative).await?;
}

// Get paths to specific object files
let mesh = bevy_sensor::ycb::object_mesh_path("/tmp/ycb", "003_cracker_box");
let texture = bevy_sensor::ycb::object_texture_path("/tmp/ycb", "003_cracker_box");
```

**Available subsets:**
- `Subset::Representative` - 3 objects (quick testing)
- `Subset::Ten` - 10 objects (TBP benchmark subset)
- `Subset::All` - All 77 YCB objects

The `assets/ycb` symlink points to `/tmp/ycb`.

## Dependencies

```toml
bevy = { version = "0.11", default-features = false, features = [
    "bevy_asset",
    "bevy_core_pipeline",
    "bevy_pbr",
    "bevy_render",
    "bevy_winit",
    "png",
    "x11",
] }
bevy_obj = "0.11"
ycbust = "0.2.3"
```

## Known Limitations

1. **Software rendering (llvmpipe)**:
   - **CRITICAL**: Must disable tonemapping (`Tonemapping::None`) on the camera, otherwise all materials render as magenta/pink (the "missing texture" fallback color). This is due to a bug in Bevy's PBR pipeline with llvmpipe.
   - Run with `WGPU_BACKEND=vulkan DISPLAY=:0` for llvmpipe Vulkan rendering.
   - PBR shaders may fail with `gsamplerCubeArrayShadow` error in some configurations.

2. **Texture loading with bevy_obj**:
   - The bevy_obj scene loader has "limited MTL support" - textures may not load automatically.
   - **Workaround**: Load textures manually with `asset_server.load()` and replace materials after scene spawns.
   - MTL files with trailing spaces in texture paths (e.g., `map_Kd texture_map.png `) may cause texture loading failures.

3. **Asset path**: When running binary directly from `target/release/`, assets must be in `target/release/assets/`. Use `cargo run` to run from project root instead.

4. **WSL2 GPU rendering**: WSL2 supports GPU compute (CUDA) but not windowed Vulkan rendering. Must use llvmpipe software rendering with the fixes above.

## Viewpoint Coordinate Reference

```
View  0-7:  pitch=-30° (below), yaw=0°-315° @ 45° steps, Y=-0.250
View  8-15: pitch=0°   (level), yaw=0°-315° @ 45° steps, Y=0.000
View 16-23: pitch=+30° (above), yaw=0°-315° @ 45° steps, Y=+0.250
```

## Usage from neocortx

```rust
use bevy_sensor::{SensorConfig, ObjectRotation, ViewpointConfig};
use bevy_sensor::ycb::{download_models, Subset, models_exist};

// Ensure YCB models are available
if !models_exist("/tmp/ycb") {
    download_models("/tmp/ycb", Subset::Representative).await?;
}

// Create capture config
let config = SensorConfig {
    viewpoints: ViewpointConfig {
        radius: 0.5,
        yaw_count: 8,
        pitch_angles_deg: vec![-30.0, 0.0, 30.0],
    },
    object_rotations: ObjectRotation::tbp_benchmark_rotations(),
    output_dir: "./captures".to_string(),
    filename_pattern: "ycb_{rot}_{view}.png".to_string(),
};

println!("Total captures: {}", config.total_captures()); // 72
```

## Related Projects

- **neocortx**: Rust-based Thousand Brains implementation (main project)
- **tbp.monty**: Original Python implementation by Thousand Brains Project
- **tbp.tbs_sensorimotor_intelligence**: TBP experiment configs
- **ycbust**: YCB dataset downloader (used as library dependency)

## Resources

- [TBP Benchmark Experiments](https://thousandbrainsproject.readme.io/docs/benchmark-experiments)
- [TBP GitHub](https://github.com/thousandbrainsproject/tbp.monty)
- [YCB Object Dataset](https://www.ycbbenchmarks.com/)
