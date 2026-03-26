use bevy_sensor::ycb::{self, REPRESENTATIVE_OBJECTS, TBP_STANDARD_OBJECTS};
use bevy_sensor::{
    REPRESENTATIVE_OBJECTS as REEXPORTED_REPRESENTATIVE_OBJECTS,
    TBP_STANDARD_OBJECTS as REEXPORTED_TBP_STANDARD_OBJECTS,
};
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
