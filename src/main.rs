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
        .add_systems(Update, capture_sequence)
        .run();
}

#[derive(Resource)]
struct Viewpoints(Vec<Transform>);

fn generate_viewpoints() -> Viewpoints {
    let mut views = Vec::new();
    let radius = 0.5;
    let height = 0.3;
    let count = 8;

    for i in 0..count {
        let angle = (i as f32) * 2.0 * PI / (count as f32);
        let x = radius * angle.cos();
        let z = radius * angle.sin();
        let transform = Transform::from_xyz(x, height, z).looking_at(Vec3::ZERO, Vec3::Y);
        views.push(transform);
    }
    Viewpoints(views)
}

#[derive(Resource, Default)]
struct CaptureState {
    view_index: usize,
    frame_counter: u32,
    step: CaptureStep,
}

#[derive(Default)]
enum CaptureStep {
    #[default]
    SetupView,
    WaitSettle,
    Capture,
    WaitSave,
}

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Camera - spawned with initial transform, will be moved by system
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.0, 0.3, 0.5).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });

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

    // Model
    let mesh_handle = asset_server.load("ycb/003_cracker_box/google_16k/textured.obj");
    let texture_handle = asset_server.load("ycb/003_cracker_box/google_16k/texture_map.png");

    let material_handle = materials.add(StandardMaterial {
        base_color_texture: Some(texture_handle),
        unlit: false,
        ..default()
    });

    commands.spawn(PbrBundle {
        mesh: mesh_handle,
        material: material_handle,
        transform: Transform::from_scale(Vec3::splat(1.0)), 
        ..default()
    });
}

fn capture_sequence(
    mut state: ResMut<CaptureState>,
    viewpoints: Res<Viewpoints>,
    mut camera_query: Query<&mut Transform, With<Camera3d>>,
    main_window: Query<Entity, With<PrimaryWindow>>,
    mut screenshot_manager: ResMut<ScreenshotManager>,
) {
    match state.step {
        CaptureStep::SetupView => {
            if state.view_index >= viewpoints.0.len() {
                // Wait a bit before exiting to ensure last save
                state.frame_counter += 1;
                if state.frame_counter > 200 {
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
            if state.frame_counter > 10 { // Wait 10 frames for things to settle
                state.step = CaptureStep::Capture;
            }
        }
        CaptureStep::Capture => {
            let path = format!("capture_{}.png", state.view_index);
            if let Ok(window_entity) = main_window.get_single() {
                screenshot_manager.save_screenshot_to_disk(window_entity, &path).unwrap();
                println!("Requested screenshot save to {}", path);
                state.frame_counter = 0;
                state.step = CaptureStep::WaitSave;
            }
        }
        CaptureStep::WaitSave => {
            state.frame_counter += 1;
            if state.frame_counter > 200 { // Wait 200 frames for save to complete
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
        assert_eq!(viewpoints.0.len(), 8);
    }

    #[test]
    fn test_viewpoints_all_at_correct_height() {
        let viewpoints = generate_viewpoints();
        let expected_height = 0.3;
        for (i, transform) in viewpoints.0.iter().enumerate() {
            assert!(
                (transform.translation.y - expected_height).abs() < 0.001,
                "Viewpoint {} has incorrect height: {}",
                i,
                transform.translation.y
            );
        }
    }

    #[test]
    fn test_viewpoints_at_correct_radius() {
        let viewpoints = generate_viewpoints();
        let expected_radius = 0.5;
        for (i, transform) in viewpoints.0.iter().enumerate() {
            let actual_radius =
                (transform.translation.x.powi(2) + transform.translation.z.powi(2)).sqrt();
            assert!(
                (actual_radius - expected_radius).abs() < 0.001,
                "Viewpoint {} has incorrect radius: {}",
                i,
                actual_radius
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
    fn test_capture_state_default() {
        let state = CaptureState::default();
        assert_eq!(state.view_index, 0);
        assert_eq!(state.frame_counter, 0);
        assert!(matches!(state.step, CaptureStep::SetupView));
    }
}
