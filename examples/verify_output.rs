//! Verify a serialized render output file.
//!
//! Run with: cargo run --example verify_output -- /tmp/bevy_sensor_render_*.bin

use std::fs::File;
use std::io::Read;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let path = if args.len() > 1 {
        &args[1]
    } else {
        // Find the most recent file
        let paths: Vec<_> = std::fs::read_dir("/tmp")
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_name()
                    .to_string_lossy()
                    .starts_with("bevy_sensor_render_")
            })
            .collect();
        if paths.is_empty() {
            eprintln!("No render output files found");
            return;
        }
        // Just use first one for now
        println!("Using: {:?}", paths[0].path());
        &paths[0].path().to_string_lossy().to_string()
    };

    let mut file = File::open(path).expect("Failed to open file");
    let mut data = Vec::new();
    file.read_to_end(&mut data).expect("Failed to read file");

    println!("File size: {} bytes", data.len());

    let mut cursor = 0;

    let read_u32 = |data: &[u8], cursor: &mut usize| -> u32 {
        let val = u32::from_le_bytes(data[*cursor..*cursor + 4].try_into().unwrap());
        *cursor += 4;
        val
    };

    let read_f32 = |data: &[u8], cursor: &mut usize| -> f32 {
        let val = f32::from_le_bytes(data[*cursor..*cursor + 4].try_into().unwrap());
        *cursor += 4;
        val
    };

    let width = read_u32(&data, &mut cursor);
    let height = read_u32(&data, &mut cursor);
    let rgba_len = read_u32(&data, &mut cursor) as usize;
    let depth_len = read_u32(&data, &mut cursor) as usize;

    println!("Image: {}x{}", width, height);
    println!("RGBA bytes: {}", rgba_len);
    println!("Depth values: {}", depth_len);

    let rgba = &data[cursor..cursor + rgba_len];
    cursor += rgba_len;

    // Count unique colors
    let mut colors: std::collections::HashSet<(u8, u8, u8)> = std::collections::HashSet::new();
    for chunk in rgba.chunks(4) {
        colors.insert((chunk[0], chunk[1], chunk[2]));
    }
    println!("Unique colors: {}", colors.len());

    // Sample some pixels
    println!("Sample pixels (RGBA):");
    for i in [0, 100, 1000, 5000, 10000, 20000] {
        if i * 4 + 4 <= rgba.len() {
            println!(
                "  Pixel {}: [{}, {}, {}, {}]",
                i,
                rgba[i * 4],
                rgba[i * 4 + 1],
                rgba[i * 4 + 2],
                rgba[i * 4 + 3]
            );
        }
    }

    // Read and display some depth values
    let mut depths = Vec::new();
    for _ in 0..depth_len {
        depths.push(read_f32(&data, &mut cursor));
    }
    println!(
        "\nDepth range: {:?} to {:?}",
        depths.iter().cloned().reduce(f32::min),
        depths.iter().cloned().reduce(f32::max)
    );

    // Intrinsics
    let fx = read_f32(&data, &mut cursor);
    let fy = read_f32(&data, &mut cursor);
    let cx = read_f32(&data, &mut cursor);
    let cy = read_f32(&data, &mut cursor);
    let iw = read_u32(&data, &mut cursor);
    let ih = read_u32(&data, &mut cursor);
    println!(
        "\nIntrinsics: fx={}, fy={}, cx={}, cy={}, size={}x{}",
        fx, fy, cx, cy, iw, ih
    );

    // Camera
    let tx = read_f32(&data, &mut cursor);
    let ty = read_f32(&data, &mut cursor);
    let tz = read_f32(&data, &mut cursor);
    let rx = read_f32(&data, &mut cursor);
    let ry = read_f32(&data, &mut cursor);
    let rz = read_f32(&data, &mut cursor);
    let rw = read_f32(&data, &mut cursor);
    println!("Camera pos: ({}, {}, {})", tx, ty, tz);
    println!("Camera rot: ({}, {}, {}, {})", rx, ry, rz, rw);

    // Object rotation
    let pitch = read_f32(&data, &mut cursor);
    let yaw = read_f32(&data, &mut cursor);
    let roll = read_f32(&data, &mut cursor);
    println!(
        "Object rotation: pitch={}, yaw={}, roll={}",
        pitch, yaw, roll
    );

    println!("\nBytes read: {} / {}", cursor, data.len());
    if cursor == data.len() {
        println!("✓ All bytes consumed successfully!");
    } else {
        println!("✗ {} extra bytes remaining", data.len() - cursor);
    }
}
