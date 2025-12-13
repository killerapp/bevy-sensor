# CLAUDE.md - Development Notes for bevy-sensor

## Project Overview

A Bevy application that captures multi-view images of 3D OBJ models (YCB dataset) for sensor simulation. This project aims to produce comparable results to the [Thousand Brains Project (TBP)](https://github.com/thousandbrainsproject/tbp.monty) habitat sensor for use in the neocortx Rust-based implementation.

## Quick Reference

```bash
# Build
cargo build --release

# Run tests
cargo test

# Run (requires GPU or proper software rendering)
cargo run --release

# Headless with software rendering (limited - PBR shaders may fail)
LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe \
  xvfb-run -a -s "-screen 0 1280x1024x24" \
  cargo run --release
```

## Architecture

### Viewpoint Generation (`src/main.rs:21-81`)

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

Default configuration produces **24 viewpoints** (8 yaw × 3 pitch).

### Capture State Machine (`src/main.rs:83-148`)

```
SetupView → WaitSettle (10 frames) → Capture → WaitSave (200 frames) → loop
```

Output: `capture_0.png` through `capture_23.png`

## TBP Habitat Sensor Alignment

| TBP Feature | Bevy Implementation |
|-------------|---------------------|
| Quaternion rotation | `Transform::looking_at()` with Y-up |
| `look_up` / `look_down` | Pitch angles: -30°, 0°, +30° |
| `turn_left` / `turn_right` | 8 yaw positions @ 45° intervals |
| Spherical radius | Configurable (default 0.5m) |
| Distant agent exploration | Full spherical coverage |

### TBP Reference Implementation

- **Sensors**: `tbp/monty/simulators/habitat/sensors.py`
  - Uses quaternion (w,x,y,z) format, default identity: (1, 0, 0, 0)
  - Position relative to HabitatAgent

- **Actions**: `tbp/monty/frameworks/actions/actions.py`
  - `LookDown`, `LookUp`: rotation_degrees parameter
  - `TurnLeft`, `TurnRight`: yaw rotation
  - `SetYaw`, `SetAgentPitch`: absolute rotations

## YCB Dataset Setup

```bash
# Install ycbust
cargo install ycbust

# Download representative YCB models
ycbust --output-dir /tmp/ycb --subset representative

# The assets/ycb symlink points to /tmp/ycb
```

## Dependencies

Minimal Bevy features to avoid system dependencies (alsa, libudev):

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
```

## Known Limitations

1. **Software rendering (llvmpipe)**: PBR shaders may fail with `gsamplerCubeArrayShadow` error. Requires real GPU for full functionality.

2. **Asset path**: When running binary directly from `target/release/`, assets must be in `target/release/assets/`. Use `cargo run` to run from project root instead.

## Viewpoint Coordinate Reference

```
View  0-7:  pitch=-30° (below), yaw=0°-315° @ 45° steps, Y=-0.250
View  8-15: pitch=0°   (level), yaw=0°-315° @ 45° steps, Y=0.000
View 16-23: pitch=+30° (above), yaw=0°-315° @ 45° steps, Y=+0.250
```

## Related Projects

- **neocortx**: Rust-based Thousand Brains implementation (main project)
- **tbp.monty**: Original Python implementation by Thousand Brains Project
- **tbp.tbs_sensorimotor_intelligence**: TBP experiment configs

## Resources

- [TBP Benchmark Experiments](https://thousandbrainsproject.readme.io/docs/benchmark-experiments)
- [TBP GitHub](https://github.com/thousandbrainsproject/tbp.monty)
- [YCB Object Dataset](https://www.ycbbenchmarks.com/)
