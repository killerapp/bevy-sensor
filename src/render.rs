//! Headless rendering implementation using Bevy.
//!
//! This module provides rendering for capturing RGBA images from YCB objects.
//! A window is briefly opened to render the scene, then `ScreenshotManager`
//! captures the frame before the window closes.
//!
//! # Current Status
//!
//! - **RGBA**: Working via `ScreenshotManager.take_screenshot()` callback
//! - **Depth**: Placeholder only (uniform camera distance per pixel)
//!
//! # Running Requirements
//!
//! On WSL2 or systems without hardware GPU rendering:
//! ```bash
//! WGPU_BACKEND=vulkan DISPLAY=:0 cargo run --example test_render
//! ```
//!
//! For CI/headless servers, use Xvfb or software rendering (llvmpipe).
//!
//! # Architecture Notes
//!
//! Bevy's `App::run()` does not return cleanly in all configurations. This
//! implementation uses a watchdog thread that monitors for completion and
//! calls `std::process::exit(0)` once the render output is serialized to
//! a temp file. The main thread reads this file after the process would
//! normally exit.

use bevy::asset::LoadState;
use bevy::core_pipeline::prepass::{DepthPrepass, NormalPrepass};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::prelude::*;
use bevy::render::view::screenshot::ScreenshotManager;
use bevy::window::{PresentMode, WindowPlugin, WindowResolution};
use bevy_obj::ObjPlugin;
use std::fs::File;
use std::io::{Read as IoRead, Write as IoWrite};
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::{ObjectRotation, RenderConfig, RenderError, RenderOutput};

/// Internal state for tracking render progress
#[derive(Resource, Default)]
struct RenderState {
    frame_count: u32,
    scene_loaded: bool,
    texture_loaded: bool,
    capture_ready: bool,
    screenshot_requested: bool,
    captured: bool,
    exit_requested: bool,
    exit_frame_count: u32,
    rgba_data: Option<Vec<u8>>,
    depth_data: Option<Vec<f32>>,
    image_width: u32,
    image_height: u32,
}

/// Shared buffer for screenshot callback to write into
#[derive(Resource, Clone)]
struct SharedImageBuffer(Arc<Mutex<Option<(Vec<u8>, u32, u32)>>>);

/// Configuration passed to the Bevy app
#[derive(Resource, Clone)]
struct RenderRequest {
    mesh_path: String,
    texture_path: String,
    camera_transform: Transform,
    object_rotation: ObjectRotation,
    config: RenderConfig,
}

/// Marker for the rendered object
#[derive(Component)]
struct RenderedObject;

/// Marker for the render camera
#[derive(Component)]
struct RenderCamera;

/// Handle for the loaded texture
#[derive(Resource)]
struct LoadedTexture(Handle<Image>);

/// Handle for the loaded scene
#[derive(Resource)]
struct LoadedScene(Handle<Scene>);

/// Shared output for extracting render results
#[derive(Resource, Clone)]
struct SharedOutput(Arc<Mutex<Option<RenderOutput>>>);

/// Perform headless rendering of a YCB object.
///
/// This spins up a minimal Bevy app, renders frames until assets are loaded,
/// then extracts the rendered frame via screenshot.
///
/// Note: Bevy's App::run() may not exit cleanly. A watchdog thread monitors
/// for results and terminates the process once the render is complete.
#[allow(dead_code)]
pub fn render_headless(
    object_dir: &Path,
    camera_transform: &Transform,
    object_rotation: &ObjectRotation,
    config: &RenderConfig,
) -> Result<RenderOutput, RenderError> {
    let mesh_path = object_dir.join("google_16k/textured.obj");
    let texture_path = object_dir.join("google_16k/texture_map.png");

    if !mesh_path.exists() {
        return Err(RenderError::MeshNotFound(mesh_path.display().to_string()));
    }
    if !texture_path.exists() {
        return Err(RenderError::TextureNotFound(
            texture_path.display().to_string(),
        ));
    }

    let request = RenderRequest {
        mesh_path: mesh_path.display().to_string(),
        texture_path: texture_path.display().to_string(),
        camera_transform: *camera_transform,
        object_rotation: object_rotation.clone(),
        config: config.clone(),
    };

    let shared_output: SharedOutput = SharedOutput(Arc::new(Mutex::new(None)));
    let output_clone = shared_output.clone();
    let output_poll = shared_output.clone();

    // Shared buffer for screenshot callback
    let shared_image: SharedImageBuffer = SharedImageBuffer(Arc::new(Mutex::new(None)));
    let image_clone = shared_image.clone();

    // Create a temp file path for output serialization
    let temp_path =
        std::env::temp_dir().join(format!("bevy_sensor_render_{}.bin", std::process::id()));
    let temp_path_clone = temp_path.clone();

    // Spawn watchdog thread that monitors for results and exits process when ready
    std::thread::spawn(move || {
        let timeout = std::time::Duration::from_secs(60);
        let start = std::time::Instant::now();
        let poll_interval = std::time::Duration::from_millis(100);

        loop {
            // Check if we have a result
            if let Ok(guard) = output_poll.0.lock() {
                if let Some(output) = guard.as_ref() {
                    eprintln!(
                        "Watchdog: Output detected! {}x{} rgba_len={}",
                        output.width,
                        output.height,
                        output.rgba.len()
                    );
                    // Serialize output to temp file
                    let data = serialize_output(output);
                    eprintln!(
                        "Watchdog: Serialized {} bytes to {:?}",
                        data.len(),
                        temp_path_clone
                    );
                    match File::create(&temp_path_clone) {
                        Ok(mut file) => {
                            if let Err(e) = file.write_all(&data) {
                                eprintln!("Watchdog: Failed to write: {}", e);
                            } else {
                                eprintln!("Watchdog: Written successfully");
                            }
                        }
                        Err(e) => {
                            eprintln!("Watchdog: Failed to create file: {}", e);
                        }
                    }
                    // Give a moment for file to flush
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    eprintln!("Watchdog: Exiting with code 0");
                    // Exit the process - App::run() won't return otherwise
                    std::process::exit(0);
                }
            }

            if start.elapsed() > timeout {
                eprintln!("Watchdog: Timeout!");
                std::process::exit(1); // Timeout
            }

            std::thread::sleep(poll_interval);
        }
    });

    // Run Bevy app on main thread (required by winit)
    App::new()
        .add_plugins(
            DefaultPlugins
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        resolution: WindowResolution::new(
                            config.width as f32,
                            config.height as f32,
                        ),
                        present_mode: PresentMode::AutoNoVsync,
                        title: "bevy-sensor render".into(),
                        ..default()
                    }),
                    ..default()
                })
                .disable::<bevy::log::LogPlugin>(),
        )
        .add_plugins(ObjPlugin)
        .insert_resource(request)
        .insert_resource(output_clone)
        .insert_resource(image_clone)
        .init_resource::<RenderState>()
        .add_systems(Startup, setup_scene)
        .add_systems(
            Update,
            (
                check_assets_loaded,
                apply_materials,
                request_screenshot,
                check_screenshot_ready,
                extract_and_exit,
            )
                .chain(),
        )
        .run();

    // If we get here, try to read from temp file (unlikely since watchdog exits)
    if temp_path.exists() {
        if let Ok(output) = read_output_from_file(&temp_path) {
            let _ = std::fs::remove_file(&temp_path);
            return Ok(output);
        }
    }

    Err(RenderError::RenderFailed(
        "Render did not complete".to_string(),
    ))
}

/// Serialize RenderOutput to bytes for IPC
fn serialize_output(output: &RenderOutput) -> Vec<u8> {
    let mut data = Vec::new();

    // Header: width, height, rgba_len, depth_len
    data.extend_from_slice(&output.width.to_le_bytes());
    data.extend_from_slice(&output.height.to_le_bytes());
    data.extend_from_slice(&(output.rgba.len() as u32).to_le_bytes());
    data.extend_from_slice(&(output.depth.len() as u32).to_le_bytes());

    // RGBA data
    data.extend_from_slice(&output.rgba);

    // Depth data (as f32 bytes)
    for d in &output.depth {
        data.extend_from_slice(&d.to_le_bytes());
    }

    // Intrinsics
    data.extend_from_slice(&output.intrinsics.focal_length[0].to_le_bytes());
    data.extend_from_slice(&output.intrinsics.focal_length[1].to_le_bytes());
    data.extend_from_slice(&output.intrinsics.principal_point[0].to_le_bytes());
    data.extend_from_slice(&output.intrinsics.principal_point[1].to_le_bytes());
    data.extend_from_slice(&output.intrinsics.image_size[0].to_le_bytes());
    data.extend_from_slice(&output.intrinsics.image_size[1].to_le_bytes());

    // Camera transform (translation + rotation quaternion)
    let t = output.camera_transform.translation;
    let r = output.camera_transform.rotation;
    data.extend_from_slice(&t.x.to_le_bytes());
    data.extend_from_slice(&t.y.to_le_bytes());
    data.extend_from_slice(&t.z.to_le_bytes());
    data.extend_from_slice(&r.x.to_le_bytes());
    data.extend_from_slice(&r.y.to_le_bytes());
    data.extend_from_slice(&r.z.to_le_bytes());
    data.extend_from_slice(&r.w.to_le_bytes());

    // Object rotation
    let or = &output.object_rotation;
    data.extend_from_slice(&or.pitch.to_le_bytes());
    data.extend_from_slice(&or.yaw.to_le_bytes());
    data.extend_from_slice(&or.roll.to_le_bytes());

    data
}

/// Read RenderOutput from serialized file
fn read_output_from_file(path: &std::path::Path) -> Result<RenderOutput, RenderError> {
    let mut file = File::open(path).map_err(|e| RenderError::RenderFailed(e.to_string()))?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)
        .map_err(|e| RenderError::RenderFailed(e.to_string()))?;

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

    let rgba = data[cursor..cursor + rgba_len].to_vec();
    cursor += rgba_len;

    let mut depth = Vec::with_capacity(depth_len);
    for _ in 0..depth_len {
        depth.push(read_f32(&data, &mut cursor));
    }

    let focal_length = [read_f32(&data, &mut cursor), read_f32(&data, &mut cursor)];
    let principal_point = [read_f32(&data, &mut cursor), read_f32(&data, &mut cursor)];
    let image_size = [read_u32(&data, &mut cursor), read_u32(&data, &mut cursor)];

    let tx = read_f32(&data, &mut cursor);
    let ty = read_f32(&data, &mut cursor);
    let tz = read_f32(&data, &mut cursor);
    let rx = read_f32(&data, &mut cursor);
    let ry = read_f32(&data, &mut cursor);
    let rz = read_f32(&data, &mut cursor);
    let rw = read_f32(&data, &mut cursor);

    let pitch = read_f32(&data, &mut cursor);
    let yaw = read_f32(&data, &mut cursor);
    let roll = read_f32(&data, &mut cursor);

    Ok(RenderOutput {
        rgba,
        depth,
        width,
        height,
        intrinsics: crate::CameraIntrinsics {
            focal_length,
            principal_point,
            image_size,
        },
        camera_transform: Transform {
            translation: Vec3::new(tx, ty, tz),
            rotation: Quat::from_xyzw(rx, ry, rz, rw),
            scale: Vec3::ONE,
        },
        object_rotation: ObjectRotation { pitch, yaw, roll },
    })
}

/// Setup the scene with camera, lighting, and object
fn setup_scene(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    request: Res<RenderRequest>,
    mut _materials: ResMut<Assets<StandardMaterial>>,
) {
    // Camera with depth prepass
    commands.spawn((
        Camera3dBundle {
            camera: Camera {
                hdr: true,
                ..default()
            },
            transform: request.camera_transform,
            tonemapping: Tonemapping::None, // Accurate colors for software rendering
            ..default()
        },
        DepthPrepass,
        NormalPrepass,
        RenderCamera,
    ));

    // Ambient light (from config)
    let lighting = &request.config.lighting;
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: lighting.ambient_brightness,
    });

    // Key light (from config)
    if lighting.key_light_intensity > 0.0 {
        commands.spawn(PointLightBundle {
            point_light: PointLight {
                intensity: lighting.key_light_intensity,
                shadows_enabled: lighting.shadows_enabled,
                ..default()
            },
            transform: Transform::from_xyz(
                lighting.key_light_position[0],
                lighting.key_light_position[1],
                lighting.key_light_position[2],
            ),
            ..default()
        });
    }

    // Fill light (from config)
    if lighting.fill_light_intensity > 0.0 {
        commands.spawn(PointLightBundle {
            point_light: PointLight {
                intensity: lighting.fill_light_intensity,
                shadows_enabled: lighting.shadows_enabled,
                ..default()
            },
            transform: Transform::from_xyz(
                lighting.fill_light_position[0],
                lighting.fill_light_position[1],
                lighting.fill_light_position[2],
            ),
            ..default()
        });
    }

    // Load the scene
    let scene_handle: Handle<Scene> = asset_server.load(&request.mesh_path);
    commands.insert_resource(LoadedScene(scene_handle.clone()));

    // Load the texture
    let texture_handle: Handle<Image> = asset_server.load(&request.texture_path);
    commands.insert_resource(LoadedTexture(texture_handle.clone()));

    // Create material with texture (will be applied later)
    let _material = _materials.add(StandardMaterial {
        base_color_texture: Some(texture_handle),
        unlit: true,
        ..default()
    });

    // Spawn the scene with rotation
    commands.spawn((
        SceneBundle {
            scene: scene_handle,
            transform: Transform::from_rotation(request.object_rotation.to_quat()),
            ..default()
        },
        RenderedObject,
    ));

    info!("Scene setup complete");
}

/// Check if assets are loaded
fn check_assets_loaded(
    mut state: ResMut<RenderState>,
    asset_server: Res<AssetServer>,
    scene: Option<Res<LoadedScene>>,
    texture: Option<Res<LoadedTexture>>,
) {
    if state.scene_loaded && state.texture_loaded {
        return;
    }

    if let Some(scene) = scene {
        match asset_server.get_load_state(&scene.0) {
            LoadState::Loaded => {
                state.scene_loaded = true;
                info!("Scene loaded");
            }
            LoadState::Failed => {
                error!("Scene failed to load");
            }
            _ => {}
        }
    }

    if let Some(texture) = texture {
        match asset_server.get_load_state(&texture.0) {
            LoadState::Loaded => {
                state.texture_loaded = true;
                info!("Texture loaded");
            }
            LoadState::Failed => {
                error!("Texture failed to load");
            }
            _ => {}
        }
    }
}

/// Apply materials to loaded meshes
fn apply_materials(
    mut state: ResMut<RenderState>,
    texture: Option<Res<LoadedTexture>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut mesh_query: Query<&mut Handle<StandardMaterial>, With<Handle<Mesh>>>,
) {
    if !state.scene_loaded || !state.texture_loaded || state.capture_ready {
        return;
    }

    state.frame_count += 1;

    // Wait a few frames for everything to settle
    if state.frame_count < 10 {
        return;
    }

    let Some(tex) = texture else { return };

    // Create textured material
    let textured_material = materials.add(StandardMaterial {
        base_color_texture: Some(tex.0.clone()),
        unlit: true,
        ..default()
    });

    // Apply to all meshes
    let mut count = 0;
    for mut mat_handle in mesh_query.iter_mut() {
        *mat_handle = textured_material.clone();
        count += 1;
    }

    if count > 0 {
        info!("Applied texture to {} meshes", count);
    }

    // Wait more frames after applying materials
    if state.frame_count >= 30 {
        state.capture_ready = true;
        info!("Ready to capture");
    }
}

/// Request a screenshot capture
fn request_screenshot(
    mut state: ResMut<RenderState>,
    main_window: Query<Entity, With<bevy::window::PrimaryWindow>>,
    mut screenshot_manager: ResMut<ScreenshotManager>,
    shared_image: Res<SharedImageBuffer>,
) {
    if !state.capture_ready || state.screenshot_requested {
        return;
    }

    let Ok(window_entity) = main_window.get_single() else {
        error!("No primary window found for screenshot");
        return;
    };

    // Clone the Arc for the callback
    let image_buffer = shared_image.0.clone();

    // Request screenshot with callback
    info!("Requesting screenshot from window {:?}", window_entity);
    if let Err(e) = screenshot_manager.take_screenshot(window_entity, move |image| {
        // This callback runs on AsyncComputeTaskPool thread
        // eprintln!("Screenshot callback triggered!");
        // Extract RGBA data from the Image
        let width = image.texture_descriptor.size.width;
        let height = image.texture_descriptor.size.height;

        // Convert image data to RGBA8
        let rgba_data = match image.texture_descriptor.format {
            bevy::render::render_resource::TextureFormat::Rgba8UnormSrgb
            | bevy::render::render_resource::TextureFormat::Rgba8Unorm => {
                // Already in RGBA8 format
                image.data.clone()
            }
            bevy::render::render_resource::TextureFormat::Bgra8UnormSrgb
            | bevy::render::render_resource::TextureFormat::Bgra8Unorm => {
                // Convert BGRA to RGBA
                let mut rgba = image.data.clone();
                for chunk in rgba.chunks_exact_mut(4) {
                    chunk.swap(0, 2); // Swap B and R
                }
                rgba
            }
            _other => {
                // Unknown texture format - use raw data
                image.data.clone()
            }
        };

        // Store in shared buffer
        // eprintln!("Storing {}x{} image ({} bytes) in shared buffer", width, height, rgba_data.len());
        if let Ok(mut guard) = image_buffer.lock() {
            *guard = Some((rgba_data, width, height));
            // eprintln!("Image stored in shared buffer");
        } else {
            // eprintln!("Failed to lock shared buffer!");
        }
    }) {
        error!("Failed to request screenshot: {:?}", e);
        return;
    }

    state.screenshot_requested = true;
    info!("Screenshot requested");
}

/// Check if screenshot callback has completed
fn check_screenshot_ready(
    mut state: ResMut<RenderState>,
    shared_image: Res<SharedImageBuffer>,
    request: Res<RenderRequest>,
) {
    if !state.screenshot_requested || state.captured {
        return;
    }

    // Check if callback has written data
    if let Ok(guard) = shared_image.0.lock() {
        if let Some((rgba_data, width, height)) = guard.as_ref() {
            // eprintln!("Found image data in shared buffer: {}x{}", width, height);
            state.rgba_data = Some(rgba_data.clone());
            state.image_width = *width;
            state.image_height = *height;

            // DEPTH BUFFER LIMITATION:
            // Bevy 0.11's ScreenshotManager only captures the color buffer.
            // True depth buffer extraction requires custom render graph nodes
            // to copy from ViewPrepassTextures::depth to a CPU-readable staging buffer.
            //
            // For now, use camera distance as a uniform placeholder. This means:
            // - Point cloud reconstruction will place all pixels at the same depth
            // - TBP algorithms expecting per-pixel depth will need this fixed
            //
            // TODO: Implement proper depth extraction via:
            // 1. Custom render node that reads DepthPrepass output
            // 2. GPU-to-CPU texture copy with staging buffer
            // 3. Or render depth-to-color shader and capture that
            let camera_dist = request.camera_transform.translation.length();
            let pixel_count = (*width * *height) as usize;
            state.depth_data = Some(vec![camera_dist; pixel_count]);

            state.captured = true;
        }
    }
}

/// Extract results and exit
fn extract_and_exit(
    mut state: ResMut<RenderState>,
    request: Res<RenderRequest>,
    shared_output: Res<SharedOutput>,
    mut commands: Commands,
    windows: Query<Entity, With<bevy::window::Window>>,
) {
    // Handle delayed exit after closing window
    if state.exit_requested {
        state.exit_frame_count += 1;
        // After a few frames with no window, Bevy should exit
        return;
    }

    if !state.captured {
        return;
    }

    if let (Some(rgba), Some(depth)) = (&state.rgba_data, &state.depth_data) {
        // eprintln!("extract_and_exit: have rgba ({} bytes) and depth ({} values)", rgba.len(), depth.len());
        // Use actual captured dimensions (may differ from config if window was resized)
        let width = state.image_width;
        let height = state.image_height;

        // Compute intrinsics based on actual dimensions
        let config = &request.config;
        let intrinsics = crate::CameraIntrinsics {
            focal_length: [width as f32 * config.zoom, height as f32 * config.zoom],
            principal_point: [width as f32 / 2.0, height as f32 / 2.0],
            image_size: [width, height],
        };

        let output = RenderOutput {
            rgba: rgba.clone(),
            depth: depth.clone(),
            width,
            height,
            intrinsics,
            camera_transform: request.camera_transform,
            object_rotation: request.object_rotation.clone(),
        };

        if let Ok(mut guard) = shared_output.0.lock() {
            *guard = Some(output);
            // eprintln!("Output stored in shared_output");
        }

        // Close all windows to trigger app exit
        // eprintln!("Closing windows to trigger exit...");
        for window_entity in windows.iter() {
            commands.entity(window_entity).despawn();
        }
        state.exit_requested = true;
    }
}
