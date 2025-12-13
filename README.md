# bevy-sensor

A Bevy library and CLI for capturing multi-view images of 3D OBJ models, designed for [Thousand Brains Project](https://github.com/thousandbrainsproject/tbp.monty) compatible sensor simulation.

## Features

- Multi-viewpoint capture using spherical coordinates
- Object rotation support matching TBP benchmark formats
- Library API for integration with neocortx
- Programmatic YCB model downloads via ycbust
- Pre-configured TBP benchmark and training configurations

## Requirements

- Rust 1.70+
- Bevy 0.11
- GPU (software rendering has limitations)

## Setup

### Getting YCB Models

**Programmatically (recommended):**

```rust
use bevy_sensor::ycb::{download_models, Subset};

// Download representative subset (3 objects)
download_models("/tmp/ycb", Subset::Representative).await?;

// Or download 10 objects for TBP benchmark testing
download_models("/tmp/ycb", Subset::Ten).await?;
```

The `assets/ycb` symlink points to `/tmp/ycb`.

## Usage

### CLI

```bash
cargo run --release
```

Default configuration captures 72 images (3 rotations × 24 viewpoints) matching TBP benchmark format.

### Library

```rust
use bevy_sensor::{SensorConfig, ObjectRotation, ViewpointConfig};

// TBP benchmark: 3 rotations × 24 viewpoints = 72 captures
let config = SensorConfig::tbp_benchmark();

// Full training: 14 rotations × 24 viewpoints = 336 captures
let config = SensorConfig::tbp_full_training();

// Custom configuration
let config = SensorConfig {
    viewpoints: ViewpointConfig {
        radius: 0.5,
        yaw_count: 8,
        pitch_angles_deg: vec![-30.0, 0.0, 30.0],
    },
    object_rotations: ObjectRotation::tbp_benchmark_rotations(),
    output_dir: "./captures".to_string(),
    filename_pattern: "capture_{rot}_{view}.png".to_string(),
};
```

### YCB Utilities

```rust
use bevy_sensor::ycb::{models_exist, object_mesh_path, object_texture_path, REPRESENTATIVE_OBJECTS};

// Check if models are downloaded
if !models_exist("/tmp/ycb") {
    download_models("/tmp/ycb", Subset::Representative).await?;
}

// Get paths to object files
let mesh = object_mesh_path("/tmp/ycb", "003_cracker_box");
let texture = object_texture_path("/tmp/ycb", "003_cracker_box");

// List available objects
for obj in REPRESENTATIVE_OBJECTS {
    println!("{}", obj);
}
```

## TBP Alignment

| TBP Benchmark | bevy-sensor |
|---------------|-------------|
| 3 known rotations `[0,0,0], [0,90,0], [0,180,0]` | `ObjectRotation::tbp_benchmark_rotations()` |
| 14 known orientations (faces + corners) | `ObjectRotation::tbp_known_orientations()` |
| Distant agent look up/down | Pitch angles: -30°, 0°, +30° |
| Turn left/right | 8 yaw positions @ 45° intervals |

## Headless Rendering

```bash
LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe \
  xvfb-run -a -s "-screen 0 1280x1024x24" \
  cargo run --release
```

Note: PBR shaders may fail with software rendering. Real GPU recommended.

## Output

Files are saved as `capture_{rotation}_{viewpoint}.png`:
- Rotation 0: identity `[0,0,0]`
- Rotation 1: 90° yaw `[0,90,0]`
- Rotation 2: 180° yaw `[0,180,0]`
- Viewpoints 0-23: spherical positions around object

## License

MIT
