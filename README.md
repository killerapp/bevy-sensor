# bevy-sensor

A Bevy application that captures multi-view images of 3D OBJ models for sensor simulation and dataset generation.

## Features

- Loads OBJ models with textures
- Automatically captures screenshots from multiple viewpoints around the model
- Configurable camera positions arranged in a circle

## Requirements

- Rust 1.70+
- Bevy 0.11

## Setup

### Getting YCB Models

The easiest way to get YCB dataset models is using [ycbust](https://crates.io/crates/ycbust):

```bash
cargo install ycbust
ycbust --output-dir /tmp/ycb --subset representative
```

This downloads the models to `/tmp/ycb`, which the `assets/ycb` symlink points to.

### Custom Models

Alternatively, place your own OBJ model and textures in the `assets/` directory and update the model path in `src/main.rs`.

### System Dependencies (Linux)

```bash
apt-get install libasound2-dev libudev-dev libwayland-dev libxkbcommon-dev
```

## Usage

```bash
cargo run --release
```

### Headless/CI Environments

For servers without a GPU, use software rendering:

```bash
LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe \
  xvfb-run -a -s "-screen 0 1280x1024x24" \
  ./target/release/bevy-sensor
```

Requires: `apt-get install xvfb mesa-utils libgl1-mesa-dri`

The application will:
1. Load the specified 3D model
2. Cycle through 8 camera viewpoints arranged in a circle
3. Save screenshots as `capture_0.png`, `capture_1.png`, etc.
4. Exit automatically after capturing all views

## Configuration

Edit `src/main.rs` to customize:
- `radius`: Camera distance from the model center
- `height`: Camera height
- `count`: Number of viewpoints to capture

## License

MIT
