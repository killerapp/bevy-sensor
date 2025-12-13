//! Headless rendering implementation using Bevy.
//!
//! This module provides the actual rendering implementation for capturing
//! RGBA and depth data from YCB objects without displaying a window.
//!
//! NOTE: True headless rendering (without any display) is complex in Bevy.
//! This implementation opens a small window, renders, and extracts pixels.
//! For CI/headless servers, use Xvfb or software rendering (llvmpipe).

use bevy::asset::LoadState;
use bevy::core_pipeline::prepass::{DepthPrepass, NormalPrepass};
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::prelude::*;
use bevy::render::view::screenshot::ScreenshotManager;
use bevy::window::{PresentMode, WindowPlugin, WindowResolution};
use bevy_obj::ObjPlugin;
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
    captured: bool,
    rgba_data: Option<Vec<u8>>,
    depth_data: Option<Vec<f32>>,
}

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
/// then extracts placeholder pixel data.
///
/// NOTE: Currently returns placeholder data. Full GPU pixel extraction
/// requires render-to-texture which is complex in Bevy 0.11.
/// The API is designed for when this is implemented.
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

    // Build and run the Bevy app
    App::new()
        // Minimal plugins for rendering
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
        // Resources
        .insert_resource(request)
        .insert_resource(output_clone)
        .init_resource::<RenderState>()
        // Systems
        .add_systems(Startup, setup_scene)
        .add_systems(
            Update,
            (
                check_assets_loaded,
                apply_materials,
                capture_frame,
                extract_and_exit,
            )
                .chain(),
        )
        .run();

    // Extract result
    let output = shared_output
        .0
        .lock()
        .map_err(|e| RenderError::RenderFailed(format!("Lock failed: {}", e)))?
        .take()
        .ok_or_else(|| RenderError::RenderFailed("No output captured".to_string()))?;

    Ok(output)
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

    // Ambient light
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 0.3,
    });

    // Key light
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 1500.0,
            shadows_enabled: false,
            ..default()
        },
        transform: Transform::from_xyz(4.0, 8.0, 4.0),
        ..default()
    });

    // Fill light
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 500.0,
            shadows_enabled: false,
            ..default()
        },
        transform: Transform::from_xyz(-4.0, 2.0, -4.0),
        ..default()
    });

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

/// Capture the rendered frame
fn capture_frame(
    mut state: ResMut<RenderState>,
    request: Res<RenderRequest>,
    _main_window: Query<Entity, With<bevy::window::PrimaryWindow>>,
    _screenshot_manager: ResMut<ScreenshotManager>,
) {
    if !state.capture_ready || state.captured {
        return;
    }

    // Generate placeholder data
    // TODO: Implement actual GPU pixel extraction via render-to-texture
    // This requires setting up a custom render target and reading back the texture
    let config = &request.config;
    let pixel_count = (config.width * config.height) as usize;

    // Placeholder RGBA (gray)
    let rgba = vec![128u8; pixel_count * 4];

    // Placeholder depth based on camera distance
    let camera_dist = request.camera_transform.translation.length();
    let depth = vec![camera_dist; pixel_count];

    state.rgba_data = Some(rgba);
    state.depth_data = Some(depth);
    state.captured = true;

    info!("Frame captured (placeholder data - actual GPU readback not yet implemented)");
}

/// Extract results and exit
fn extract_and_exit(
    state: Res<RenderState>,
    request: Res<RenderRequest>,
    shared_output: Res<SharedOutput>,
    mut exit: EventWriter<bevy::app::AppExit>,
) {
    if !state.captured {
        return;
    }

    if let (Some(rgba), Some(depth)) = (&state.rgba_data, &state.depth_data) {
        let config = &request.config;
        let output = RenderOutput {
            rgba: rgba.clone(),
            depth: depth.clone(),
            width: config.width,
            height: config.height,
            intrinsics: config.intrinsics(),
            camera_transform: request.camera_transform,
            object_rotation: request.object_rotation.clone(),
        };

        if let Ok(mut guard) = shared_output.0.lock() {
            *guard = Some(output);
        }

        info!("Output extracted, exiting");
        exit.send(bevy::app::AppExit);
    }
}
