use bevy_sensor::ycb::{self, REPRESENTATIVE_OBJECTS, TBP_STANDARD_OBJECTS};
use bevy_sensor::{
    REPRESENTATIVE_OBJECTS as REEXPORTED_REPRESENTATIVE_OBJECTS,
    TBP_STANDARD_OBJECTS as REEXPORTED_TBP_STANDARD_OBJECTS,
};
use std::fs;
use std::path::Path;

#[test]
fn public_ycb_constants_expose_expected_downstream_sets() {
    assert_eq!(REPRESENTATIVE_OBJECTS.len(), 3);
    assert_eq!(TBP_STANDARD_OBJECTS.len(), 10);
    assert!(REPRESENTATIVE_OBJECTS.contains(&"003_cracker_box"));
    assert!(TBP_STANDARD_OBJECTS.contains(&"025_mug"));
    assert!(TBP_STANDARD_OBJECTS.contains(&"024_bowl"));
    assert_eq!(REPRESENTATIVE_OBJECTS, REEXPORTED_REPRESENTATIVE_OBJECTS);
    assert_eq!(TBP_STANDARD_OBJECTS, REEXPORTED_TBP_STANDARD_OBJECTS);
}

#[test]
fn public_ycb_download_objects_is_callable_without_internal_imports() {
    let future = ycb::download_objects(Path::new("."), &["003_cracker_box"]);
    drop(future);
}

#[test]
fn public_ycb_objects_exist_validates_requested_set() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mesh = ycb::object_mesh_path(dir.path(), "003_cracker_box");
    let texture = ycb::object_texture_path(dir.path(), "003_cracker_box");
    fs::create_dir_all(mesh.parent().expect("mesh parent")).expect("create object dir");
    fs::write(&mesh, b"").expect("write mesh marker");
    fs::write(&texture, b"").expect("write texture marker");

    assert!(ycb::objects_exist(dir.path(), &["003_cracker_box"]));
    assert!(!ycb::objects_exist(
        dir.path(),
        &["003_cracker_box", "004_sugar_box"]
    ));
    assert_eq!(
        ycb::missing_objects(dir.path(), &["003_cracker_box", "004_sugar_box"]),
        vec!["004_sugar_box".to_string()]
    );
}

#[test]
fn public_ycb_models_exist_requires_representative_set() {
    let dir = tempfile::tempdir().expect("tempdir");
    let mesh = ycb::object_mesh_path(dir.path(), "003_cracker_box");
    let texture = ycb::object_texture_path(dir.path(), "003_cracker_box");
    fs::create_dir_all(mesh.parent().expect("mesh parent")).expect("create object dir");
    fs::write(&mesh, b"").expect("write mesh marker");
    fs::write(&texture, b"").expect("write texture marker");

    assert!(!ycb::models_exist(dir.path()));
}
