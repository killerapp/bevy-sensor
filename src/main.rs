use bevy::asset::LoadState;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::prelude::*;
use bevy::render::view::screenshot::ScreenshotManager;
use bevy::window::PrimaryWindow;
use bevy_obj::ObjPlugin;
use std::f32::consts::PI;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(ObjPlugin)
        .init_resource::<CaptureState>()
        .insert_resource(generate_viewpoints())
        .add_systems(Startup, setup)
        .add_systems(Update, (replace_materials, capture_sequence).chain())
        .run();
}

#[derive(Resource)]
struct Viewpoints(Vec<Transform>);

/// Configuration for viewpoint generation matching TBP habitat sensor behavior.
/// Uses spherical coordinates to capture objects from multiple elevations.
#[derive(Clone)]
struct ViewpointConfig {
    /// Distance from camera to object center (meters)
    radius: f32,
    /// Number of horizontal positions (yaw angles) around the object
    yaw_count: usize,
    /// Elevation angles in degrees (pitch). Positive = above, negative = below.
    /// TBP distant agent uses up/down/left/right movement for exploration.
    pitch_angles_deg: Vec<f32>,
}

impl Default for ViewpointConfig {
    fn default() -> Self {
        Self {
            radius: 0.5,
            yaw_count: 8,
            // Three elevations: below (-30°), level (0°), above (+30°)
            // This matches TBP's look_up/look_down capability
            pitch_angles_deg: vec![-30.0, 0.0, 30.0],
        }
    }
}

fn generate_viewpoints() -> Viewpoints {
    generate_viewpoints_with_config(ViewpointConfig::default())
}

/// Generate camera viewpoints using spherical coordinates.
///
/// Spherical coordinate system (matching TBP habitat sensor conventions):
/// - Yaw: horizontal rotation around Y-axis (0° to 360°)
/// - Pitch: elevation angle from horizontal plane (-90° to +90°)
/// - Radius: distance from origin (object center)
///
/// This produces viewpoints that cover the object from multiple angles and elevations,
/// similar to how TBP's distant agent explores objects with look_up/look_down/turn_left/turn_right.
fn generate_viewpoints_with_config(config: ViewpointConfig) -> Viewpoints {
    let mut views = Vec::new();

    for pitch_deg in &config.pitch_angles_deg {
        let pitch = pitch_deg.to_radians();

        for i in 0..config.yaw_count {
            let yaw = (i as f32) * 2.0 * PI / (config.yaw_count as f32);

            // Spherical to Cartesian conversion (Y-up coordinate system)
            // x = r * cos(pitch) * sin(yaw)
            // y = r * sin(pitch)
            // z = r * cos(pitch) * cos(yaw)
            let x = config.radius * pitch.cos() * yaw.sin();
            let y = config.radius * pitch.sin();
            let z = config.radius * pitch.cos() * yaw.cos();

            let transform = Transform::from_xyz(x, y, z).looking_at(Vec3::ZERO, Vec3::Y);
            views.push(transform);
        }
    }
    Viewpoints(views)
}

#[derive(Resource, Default)]
struct CaptureState {
    view_index: usize,
    frame_counter: u32,
    step: CaptureStep,
    startup_frames: u32, // Wait for assets to load
}

#[derive(Resource)]
struct TextureHandle(Handle<Image>);

#[derive(Resource)]
struct TexturedMaterial(Handle<StandardMaterial>);

#[derive(Resource, Default)]
#[allow(dead_code)]
struct MaterialsReplaced(bool);

#[derive(Default)]
enum CaptureStep {
    #[default]
    WaitForAssets, // Initial state - wait for textures to load
    SetupView,
    WaitSettle,
    Capture,
    WaitSave,
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    _meshes: ResMut<Assets<Mesh>>,
) {
    // Camera - spawned with initial transform, will be moved by system
    // Disable tonemapping for software rendering compatibility
    commands.spawn((Camera3dBundle {
        transform: Transform::from_xyz(0.0, 0.3, 0.5).looking_at(Vec3::ZERO, Vec3::Y),
        tonemapping: Tonemapping::None,
        ..default()
    },));

    // Light
    commands.spawn(PointLightBundle {
        point_light: PointLight {
            intensity: 1500.0,
            shadows_enabled: false,
            ..default()
        },
        transform: Transform::from_xyz(4.0, 8.0, 4.0),
        ..default()
    });

    // Ambient light
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 0.5,
    });

    // Load scene (for geometry) and texture separately
    let scene_handle: Handle<Scene> =
        asset_server.load("ycb/003_cracker_box/google_16k/textured.obj");
    let texture_handle: Handle<Image> =
        asset_server.load("ycb/003_cracker_box/google_16k/texture_map.png");

    println!("Loading scene from: ycb/003_cracker_box/google_16k/textured.obj");
    println!("Loading texture from: ycb/003_cracker_box/google_16k/texture_map.png");

    // Create unlit material with the texture
    let textured_material = materials.add(StandardMaterial {
        base_color_texture: Some(texture_handle.clone()),
        unlit: true, // Unlit so texture colors are accurate
        ..default()
    });

    // Store handles for later use
    commands.insert_resource(TextureHandle(texture_handle));
    commands.insert_resource(TexturedMaterial(textured_material));
    commands.insert_resource(MaterialsReplaced::default());

    commands.spawn(SceneBundle {
        scene: scene_handle,
        transform: Transform::from_scale(Vec3::splat(1.0)),
        ..default()
    });
}

/// Replace all materials in the scene with our manually-loaded textured material
#[allow(clippy::too_many_arguments, clippy::manual_is_multiple_of)]
fn replace_materials(
    _replaced: ResMut<MaterialsReplaced>,
    textured_mat: Option<Res<TexturedMaterial>>,
    texture_handle: Option<Res<TextureHandle>>,
    asset_server: Res<AssetServer>,
    mut mesh_query: Query<(Entity, &mut Handle<StandardMaterial>), With<Handle<Mesh>>>,
    all_entities: Query<Entity>,
    mesh_entities: Query<Entity, With<Handle<Mesh>>>,
    state: Res<CaptureState>,
) {
    // Wait for texture to be loaded
    let Some(tex_handle) = texture_handle else {
        return;
    };
    let load_state = asset_server.get_load_state(&tex_handle.0);
    if load_state != LoadState::Loaded {
        return;
    }

    let Some(mat) = textured_mat else { return };

    // Keep replacing materials every frame until capture starts
    // This ensures we catch scene entities as they spawn
    if !matches!(state.step, CaptureStep::WaitForAssets) {
        return;
    }

    // Debug: count entities
    let total_entities = all_entities.iter().count();
    let mesh_entity_count = mesh_entities.iter().count();
    let mat_entity_count = mesh_query.iter().count();

    if state.startup_frames % 30 == 0 {
        println!(
            "DEBUG: {} total entities, {} with mesh, {} with mesh+material",
            total_entities, mesh_entity_count, mat_entity_count
        );
    }

    // Replace all materials
    let mut count = 0;
    for (entity, mut material_handle) in mesh_query.iter_mut() {
        if *material_handle != mat.0 {
            println!("Replacing material on entity {:?}", entity);
            *material_handle = mat.0.clone();
            count += 1;
        }
    }

    if count > 0 {
        println!("Replaced {} materials with textured material", count);
    }
}

fn capture_sequence(
    mut state: ResMut<CaptureState>,
    viewpoints: Res<Viewpoints>,
    mut camera_query: Query<&mut Transform, With<Camera3d>>,
    main_window: Query<Entity, With<PrimaryWindow>>,
    mut screenshot_manager: ResMut<ScreenshotManager>,
    asset_server: Res<AssetServer>,
    texture_handle: Option<Res<TextureHandle>>,
) {
    match state.step {
        CaptureStep::WaitForAssets => {
            state.startup_frames += 1;

            // Check actual asset load state
            if let Some(handle) = &texture_handle {
                let load_state = asset_server.get_load_state(&handle.0);
                if state.startup_frames % 30 == 0 {
                    println!(
                        "Frame {}: Texture load state: {:?}",
                        state.startup_frames, load_state
                    );
                }

                match load_state {
                    LoadState::Loaded => {
                        let path = asset_server.get_handle_path(&handle.0);
                        println!("Texture loaded. Path: {:?}", path);

                        // Add extra wait for dependent assets after texture is loaded
                        // Wait 60 frames for render pipeline to process materials
                        if state.startup_frames < 60 {
                            // Continue waiting
                        } else {
                            println!(
                                "Texture loaded after {} frames, starting capture...",
                                state.startup_frames
                            );
                            state.step = CaptureStep::SetupView;
                        }
                    }
                    LoadState::Failed => {
                        println!("ERROR: Texture failed to load!");
                        std::process::exit(1);
                    }
                    _ => {
                        // Still loading, continue waiting (max 300 frames = 5 sec)
                        if state.startup_frames >= 300 {
                            println!(
                                "WARNING: Asset loading timeout, proceeding anyway. State: {:?}",
                                load_state
                            );
                            state.step = CaptureStep::SetupView;
                        }
                    }
                }
            } else if state.startup_frames >= 120 {
                println!("Assets loaded (no handle check), starting capture sequence...");
                state.step = CaptureStep::SetupView;
            }
        }
        CaptureStep::SetupView => {
            // Capture all viewpoints
            if state.view_index >= viewpoints.0.len() {
                // Wait a bit before exiting to ensure last save
                state.frame_counter += 1;
                if state.frame_counter > 50 {
                    println!("All views captured. Exiting.");
                    std::process::exit(0);
                }
                return;
            }

            if let Ok(mut transform) = camera_query.get_single_mut() {
                *transform = viewpoints.0[state.view_index];
                println!("Moved to view {}", state.view_index);
                state.frame_counter = 0;
                state.step = CaptureStep::WaitSettle;
            }
        }
        CaptureStep::WaitSettle => {
            state.frame_counter += 1;
            if state.frame_counter > 10 {
                // Wait 10 frames for things to settle
                state.step = CaptureStep::Capture;
            }
        }
        CaptureStep::Capture => {
            let path = format!("capture_{}.png", state.view_index);
            if let Ok(window_entity) = main_window.get_single() {
                screenshot_manager
                    .save_screenshot_to_disk(window_entity, &path)
                    .unwrap();
                println!("Requested screenshot save to {}", path);
                state.frame_counter = 0;
                state.step = CaptureStep::WaitSave;
            }
        }
        CaptureStep::WaitSave => {
            state.frame_counter += 1;
            if state.frame_counter > 30 {
                // Wait 30 frames for save to complete
                state.view_index += 1;
                state.step = CaptureStep::SetupView;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_viewpoints_count() {
        let viewpoints = generate_viewpoints();
        // 8 yaw positions × 3 pitch angles = 24 viewpoints
        assert_eq!(viewpoints.0.len(), 24);
    }

    #[test]
    fn test_viewpoints_cover_multiple_elevations() {
        let viewpoints = generate_viewpoints();
        let config = ViewpointConfig::default();

        // Group viewpoints by pitch angle
        let mut elevations: Vec<f32> = viewpoints.0.iter().map(|t| t.translation.y).collect();

        // Remove near-duplicates and sort
        elevations.sort_by(|a, b| a.partial_cmp(b).unwrap());
        elevations.dedup_by(|a, b| (*a - *b).abs() < 0.01);

        // Should have 3 distinct elevation levels
        assert_eq!(
            elevations.len(),
            config.pitch_angles_deg.len(),
            "Expected {} elevation levels, got {}",
            config.pitch_angles_deg.len(),
            elevations.len()
        );
    }

    #[test]
    fn test_viewpoints_at_correct_spherical_radius() {
        let viewpoints = generate_viewpoints();
        let config = ViewpointConfig::default();

        for (i, transform) in viewpoints.0.iter().enumerate() {
            // Spherical radius: sqrt(x² + y² + z²)
            let actual_radius = transform.translation.length();
            assert!(
                (actual_radius - config.radius).abs() < 0.001,
                "Viewpoint {} has incorrect spherical radius: {} (expected {})",
                i,
                actual_radius,
                config.radius
            );
        }
    }

    #[test]
    fn test_viewpoints_looking_at_origin() {
        let viewpoints = generate_viewpoints();
        for (i, transform) in viewpoints.0.iter().enumerate() {
            let forward = transform.forward();
            let to_origin = (Vec3::ZERO - transform.translation).normalize();
            let dot = forward.dot(to_origin);
            assert!(
                dot > 0.99,
                "Viewpoint {} not looking at origin, dot product: {}",
                i,
                dot
            );
        }
    }

    #[test]
    fn test_viewpoints_pitch_angles_correct() {
        let config = ViewpointConfig::default();
        let viewpoints = generate_viewpoints_with_config(config.clone());

        for (pitch_idx, pitch_deg) in config.pitch_angles_deg.iter().enumerate() {
            let pitch_rad = pitch_deg.to_radians();
            let expected_y = config.radius * pitch_rad.sin();

            for yaw_idx in 0..config.yaw_count {
                let view_idx = pitch_idx * config.yaw_count + yaw_idx;
                let actual_y = viewpoints.0[view_idx].translation.y;

                assert!(
                    (actual_y - expected_y).abs() < 0.001,
                    "Viewpoint {} (pitch={}, yaw={}) has incorrect Y: {} (expected {})",
                    view_idx,
                    pitch_deg,
                    yaw_idx,
                    actual_y,
                    expected_y
                );
            }
        }
    }

    #[test]
    fn test_viewpoints_yaw_distribution() {
        let config = ViewpointConfig {
            radius: 1.0,
            yaw_count: 4,
            pitch_angles_deg: vec![0.0], // Single elevation for simpler testing
        };
        let viewpoints = generate_viewpoints_with_config(config);

        // At pitch=0, y=0 and positions should be on XZ plane
        // Expected positions at yaw = 0°, 90°, 180°, 270°
        let expected_positions = [
            (0.0, 0.0, 1.0),  // yaw=0° → z=1, x=0
            (1.0, 0.0, 0.0),  // yaw=90° → x=1, z=0
            (0.0, 0.0, -1.0), // yaw=180° → z=-1, x=0
            (-1.0, 0.0, 0.0), // yaw=270° → x=-1, z=0
        ];

        for (i, (ex, ey, ez)) in expected_positions.iter().enumerate() {
            let pos = viewpoints.0[i].translation;
            assert!(
                (pos.x - ex).abs() < 0.001
                    && (pos.y - ey).abs() < 0.001
                    && (pos.z - ez).abs() < 0.001,
                "Viewpoint {} at wrong position: ({}, {}, {}) expected ({}, {}, {})",
                i,
                pos.x,
                pos.y,
                pos.z,
                ex,
                ey,
                ez
            );
        }
    }

    #[test]
    fn test_custom_config() {
        let config = ViewpointConfig {
            radius: 1.0,
            yaw_count: 4,
            pitch_angles_deg: vec![-45.0, 0.0, 45.0, 90.0],
        };
        let viewpoints = generate_viewpoints_with_config(config);

        // 4 yaw × 4 pitch = 16 viewpoints
        assert_eq!(viewpoints.0.len(), 16);
    }

    #[test]
    fn test_capture_state_default() {
        let state = CaptureState::default();
        assert_eq!(state.view_index, 0);
        assert_eq!(state.frame_counter, 0);
        assert!(matches!(state.step, CaptureStep::WaitForAssets));
    }
}
