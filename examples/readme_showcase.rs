//! Generate the README RGB-D showcase image.
//!
//! Usage:
//! ```sh
//! cargo run --example readme_showcase -- --data-dir /tmp/ycb --output-dir docs/images
//! ```

use bevy_sensor::ycb::{download_objects, objects_exist};
use bevy_sensor::{render_to_buffer, ObjectRotation, RenderConfig, Transform, Vec3};
use image::imageops::{overlay, resize, FilterType};
use image::{Rgba, RgbaImage};
use std::path::{Path, PathBuf};

const DEFAULT_DATA_DIR: &str = "/tmp/ycb";
const DEFAULT_OUTPUT_DIR: &str = "docs/images";
const OUTPUT_FILE: &str = "ycb_rgbd_showcase.png";
const CELL_SIZE: u32 = 256;
const PADDING: u32 = 16;
const MAX_COLUMNS: u32 = 3;

struct Options {
    data_dir: PathBuf,
    output_dir: PathBuf,
    objects: Option<Vec<String>>,
}

#[derive(Clone)]
struct ShowcasePreset {
    object_id: String,
    yaw_deg: f32,
    pitch_deg: f32,
    rotation: ObjectRotation,
}

struct Bounds {
    center: Vec3,
    diagonal: f32,
}

struct TilePair {
    rgb: RgbaImage,
    depth: RgbaImage,
}

const DEFAULT_PRESETS: &[(&str, f32, f32, [f64; 3])] = &[
    ("003_cracker_box", 25.0, 14.0, [0.0, 18.0, 0.0]),
    ("006_mustard_bottle", 315.0, 10.0, [0.0, 0.0, 0.0]),
    ("011_banana", 55.0, 20.0, [8.0, 35.0, -18.0]),
    ("025_mug", 220.0, 18.0, [0.0, 210.0, 0.0]),
    ("035_power_drill", 135.0, 12.0, [10.0, 120.0, 0.0]),
    ("077_rubiks_cube", 45.0, 26.0, [18.0, 40.0, 8.0]),
];

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    bevy_sensor::initialize();

    let options = parse_args();
    let presets = build_presets(options.objects);
    let object_ids: Vec<&str> = presets
        .iter()
        .map(|preset| preset.object_id.as_str())
        .collect();

    if !objects_exist(&options.data_dir, &object_ids) {
        println!(
            "Downloading missing YCB objects into {}...",
            options.data_dir.display()
        );
        download_objects(&options.data_dir, &object_ids).await?;
    }

    let render_config = RenderConfig {
        width: CELL_SIZE,
        height: CELL_SIZE,
        zoom: 2.2,
        ..RenderConfig::preview()
    };

    let mut tiles = Vec::with_capacity(presets.len());
    for preset in &presets {
        println!("Rendering {}...", preset.object_id);
        let object_dir = options.data_dir.join(&preset.object_id);
        let bounds = estimate_bounds(&object_dir).unwrap_or(Bounds {
            center: Vec3::ZERO,
            diagonal: 0.18,
        });
        let radius = (bounds.diagonal * 2.1).clamp(0.28, 0.72);
        let camera = camera_transform(bounds.center, radius, preset.yaw_deg, preset.pitch_deg);
        let output = render_to_buffer(&object_dir, &camera, &preset.rotation, &render_config)?;

        tiles.push(TilePair {
            rgb: rgb_tile(
                &output.rgba,
                &output.depth,
                output.width,
                output.height,
                render_config.far_plane,
            ),
            depth: depth_tile(
                &output.depth,
                output.width,
                output.height,
                render_config.far_plane,
            ),
        });
    }

    std::fs::create_dir_all(&options.output_dir)?;
    let sheet = contact_sheet(&tiles);
    let output_path = options.output_dir.join(OUTPUT_FILE);
    sheet.save(&output_path)?;
    println!("Saved {}", output_path.display());

    Ok(())
}

fn parse_args() -> Options {
    let mut data_dir = PathBuf::from(DEFAULT_DATA_DIR);
    let mut output_dir = PathBuf::from(DEFAULT_OUTPUT_DIR);
    let mut objects = None;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--data-dir" => {
                if let Some(value) = args.next() {
                    data_dir = PathBuf::from(value);
                }
            }
            "--output-dir" => {
                if let Some(value) = args.next() {
                    output_dir = PathBuf::from(value);
                }
            }
            "--objects" => {
                if let Some(value) = args.next() {
                    objects = Some(
                        value
                            .split(',')
                            .map(str::trim)
                            .filter(|value| !value.is_empty())
                            .map(ToOwned::to_owned)
                            .collect(),
                    );
                }
            }
            _ => {}
        }
    }

    Options {
        data_dir,
        output_dir,
        objects,
    }
}

fn build_presets(objects: Option<Vec<String>>) -> Vec<ShowcasePreset> {
    if let Some(objects) = objects {
        let viewpoints = [
            (25.0, 14.0),
            (315.0, 10.0),
            (55.0, 20.0),
            (220.0, 18.0),
            (135.0, 12.0),
            (45.0, 26.0),
        ];

        return objects
            .into_iter()
            .enumerate()
            .map(|(index, object_id)| {
                let (yaw_deg, pitch_deg) = viewpoints[index % viewpoints.len()];
                ShowcasePreset {
                    object_id,
                    yaw_deg,
                    pitch_deg,
                    rotation: ObjectRotation::identity(),
                }
            })
            .collect();
    }

    DEFAULT_PRESETS
        .iter()
        .map(|(object_id, yaw_deg, pitch_deg, rotation)| ShowcasePreset {
            object_id: (*object_id).to_string(),
            yaw_deg: *yaw_deg,
            pitch_deg: *pitch_deg,
            rotation: ObjectRotation::from_array(*rotation),
        })
        .collect()
}

fn estimate_bounds(object_dir: &Path) -> Option<Bounds> {
    let mesh_path = object_dir.join("google_16k").join("textured.obj");
    let mesh = std::fs::read_to_string(mesh_path).ok()?;

    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    let mut found_vertex = false;

    for line in mesh.lines() {
        let mut parts = line.split_whitespace();
        if parts.next() != Some("v") {
            continue;
        }

        let x = parts.next()?.parse::<f32>().ok()?;
        let y = parts.next()?.parse::<f32>().ok()?;
        let z = parts.next()?.parse::<f32>().ok()?;
        let vertex = Vec3::new(x, y, z);
        min = min.min(vertex);
        max = max.max(vertex);
        found_vertex = true;
    }

    found_vertex.then(|| {
        let size = max - min;
        Bounds {
            center: (min + max) * 0.5,
            diagonal: size.length(),
        }
    })
}

fn camera_transform(center: Vec3, radius: f32, yaw_deg: f32, pitch_deg: f32) -> Transform {
    let yaw = yaw_deg.to_radians();
    let pitch = pitch_deg.to_radians();
    let eye = center
        + Vec3::new(
            radius * pitch.cos() * yaw.sin(),
            radius * pitch.sin(),
            radius * pitch.cos() * yaw.cos(),
        );

    Transform::from_translation(eye).looking_at(center, Vec3::Y)
}

fn rgb_tile(rgba: &[u8], depth: &[f64], width: u32, height: u32, far_plane: f32) -> RgbaImage {
    let mut tile = RgbaImage::from_pixel(width, height, background_pixel());
    let far = far_plane as f64 * 0.99;

    for y in 0..height {
        for x in 0..width {
            let pixel_index = (y * width + x) as usize;
            if depth.get(pixel_index).copied().unwrap_or(far_plane as f64) >= far {
                continue;
            }

            let rgba_index = pixel_index * 4;
            if rgba_index + 3 < rgba.len() {
                tile.put_pixel(
                    x,
                    y,
                    Rgba([
                        rgba[rgba_index],
                        rgba[rgba_index + 1],
                        rgba[rgba_index + 2],
                        255,
                    ]),
                );
            }
        }
    }

    tile
}

fn depth_tile(depth: &[f64], width: u32, height: u32, far_plane: f32) -> RgbaImage {
    let mut min_depth = f64::INFINITY;
    let mut max_depth = f64::NEG_INFINITY;
    let far = far_plane as f64 * 0.99;

    for value in depth.iter().copied().filter(|value| *value < far) {
        min_depth = min_depth.min(value);
        max_depth = max_depth.max(value);
    }

    let range = (max_depth - min_depth).max(1e-9);
    let mut tile = RgbaImage::from_pixel(width, height, background_pixel());

    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            let Some(value) = depth.get(index).copied() else {
                continue;
            };
            if value >= far {
                continue;
            }

            let normalized = ((value - min_depth) / range).clamp(0.0, 1.0);
            tile.put_pixel(x, y, depth_color(1.0 - normalized));
        }
    }

    tile
}

fn contact_sheet(tiles: &[TilePair]) -> RgbaImage {
    let columns = MAX_COLUMNS.min(tiles.len().max(1) as u32);
    let groups = (tiles.len() as u32).div_ceil(columns);
    let rows = groups * 2;
    let width = columns * CELL_SIZE + (columns + 1) * PADDING;
    let height = rows * CELL_SIZE + (rows + 1) * PADDING;
    let mut sheet = RgbaImage::from_pixel(width, height, sheet_pixel());

    for (index, tile_pair) in tiles.iter().enumerate() {
        let index = index as u32;
        let column = index % columns;
        let group = index / columns;
        let x = PADDING + column * (CELL_SIZE + PADDING);
        let rgb_y = PADDING + group * 2 * (CELL_SIZE + PADDING);
        let depth_y = rgb_y + CELL_SIZE + PADDING;

        let rgb = resize(&tile_pair.rgb, CELL_SIZE, CELL_SIZE, FilterType::Lanczos3);
        let depth = resize(&tile_pair.depth, CELL_SIZE, CELL_SIZE, FilterType::Lanczos3);

        overlay(&mut sheet, &rgb, i64::from(x), i64::from(rgb_y));
        overlay(&mut sheet, &depth, i64::from(x), i64::from(depth_y));
    }

    sheet
}

fn depth_color(value: f64) -> Rgba<u8> {
    let value = value.clamp(0.0, 1.0);
    let (start, end, t) = if value < 0.5 {
        ([33.0, 76.0, 164.0], [24.0, 190.0, 166.0], value * 2.0)
    } else {
        (
            [24.0, 190.0, 166.0],
            [250.0, 204.0, 76.0],
            (value - 0.5) * 2.0,
        )
    };

    Rgba([
        lerp(start[0], end[0], t) as u8,
        lerp(start[1], end[1], t) as u8,
        lerp(start[2], end[2], t) as u8,
        255,
    ])
}

fn lerp(start: f64, end: f64, t: f64) -> f64 {
    start + (end - start) * t
}

fn background_pixel() -> Rgba<u8> {
    Rgba([244, 246, 249, 255])
}

fn sheet_pixel() -> Rgba<u8> {
    Rgba([226, 231, 237, 255])
}
