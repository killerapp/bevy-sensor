use bevy_sensor::benchmark::{
    fixed_orbit_frame_count, neocortx_targeting_policy, percentile, summarize_timing,
    BenchmarkRenderPath, BenchmarkWorkload,
};
use bevy_sensor::{ObjectRotation, TargetingPolicy};

#[test]
fn fixed_orbit_workload_counts_neocortx_frames() {
    assert_eq!(fixed_orbit_frame_count(3, 3, 24), 216);
    assert_eq!(fixed_orbit_frame_count(10, 3, 24), 720);

    let workload = BenchmarkWorkload::fixed_orbit(
        "fixed-orbit-3",
        vec![
            "003_cracker_box".to_string(),
            "005_tomato_soup_can".to_string(),
            "011_banana".to_string(),
        ],
        3,
        24,
    );

    assert_eq!(workload.render_path, BenchmarkRenderPath::FixedOrbitSession);
    assert_eq!(workload.total_frames, 216);
    assert_eq!(workload.steps_per_object, None);
}

#[test]
fn persistent_workload_counts_per_step_frames() {
    let workload = BenchmarkWorkload::persistent_steps(
        "persistent-surface",
        vec!["025_mug".to_string(), "003_cracker_box".to_string()],
        120,
    );

    assert_eq!(workload.render_path, BenchmarkRenderPath::PersistentSteps);
    assert_eq!(workload.total_frames, 240);
    assert_eq!(workload.steps_per_object, Some(120));
}

#[test]
fn neocortx_targeting_policy_matches_current_downstream_behavior() {
    assert_eq!(
        neocortx_targeting_policy(&ObjectRotation::new(0.0, 90.0, 0.0)),
        TargetingPolicy::Origin
    );
    assert_eq!(
        neocortx_targeting_policy(&ObjectRotation::new(45.0, 90.0, 0.0)),
        TargetingPolicy::MeshCenter
    );
    assert_eq!(
        neocortx_targeting_policy(&ObjectRotation::new(0.0, 90.0, 10.0)),
        TargetingPolicy::MeshCenter
    );
}

#[test]
fn timing_summary_reports_marker_percentiles_and_total_throughput() {
    let markers = [10.0, 20.0, 30.0, 40.0];
    let summary = summarize_timing(&markers, 8, 200.0);

    assert_eq!(summary.total_frames, 8);
    assert_eq!(summary.measurement_count, 4);
    assert_eq!(summary.frames_per_second, 40.0);
    assert_eq!(summary.mean_ms_per_frame, 25.0);
    assert_eq!(summary.p50_ms_per_frame, 30.0);
    assert_eq!(summary.p95_ms_per_frame, 40.0);
    assert_eq!(summary.min_ms_per_frame, 10.0);
    assert_eq!(summary.max_ms_per_frame, 40.0);
}

#[test]
fn percentile_ignores_non_finite_samples() {
    let samples = [f64::NAN, 4.0, 1.0, f64::INFINITY, 2.0];

    assert_eq!(percentile(&samples, 0.0), Some(1.0));
    assert_eq!(percentile(&samples, 0.5), Some(2.0));
    assert_eq!(percentile(&samples, 1.0), Some(4.0));
    assert_eq!(percentile(&[f64::NAN], 0.5), None);
}
