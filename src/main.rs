use bevy::prelude::*;
use bevy_obj::ObjPlugin;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(ObjPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, rotate_model)
        .run();
}

#[derive(Component)]
struct Rotator;

fn setup(
    mut commands: Commands,
    asset_server: Res<AssetServer>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Camera
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
    // We load the mesh and the texture manually to ensure it looks correct.
    let mesh_handle = asset_server.load("ycb/003_cracker_box/google_16k/textured.obj");
    let texture_handle = asset_server.load("ycb/003_cracker_box/google_16k/texture_map.png");

    let material_handle = materials.add(StandardMaterial {
        base_color_texture: Some(texture_handle),
        unlit: false,
        ..default()
    });

    commands.spawn((
        PbrBundle {
            mesh: mesh_handle,
            material: material_handle,
            transform: Transform::from_scale(Vec3::splat(1.0)), 
            ..default()
        },
        Rotator,
    ));
}

fn rotate_model(mut query: Query<&mut Transform, With<Rotator>>, time: Res<Time>) {
    for mut transform in &mut query {
        transform.rotate_y(time.delta_seconds() * 0.5);
    }
}
