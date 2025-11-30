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

1. Place your OBJ model and textures in the `assets/` directory
2. Update the model path in `src/main.rs` if needed

## Usage

```bash
cargo run
```

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
