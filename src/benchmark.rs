//! Benchmark helpers for renderer throughput artifacts.
//!
//! This module intentionally keeps the reusable pieces small and deterministic:
//! workload sizing, NeoCortx-compatible targeting policy selection, and timing
//! summaries. GPU work and file output live in `src/bin/render_benchmark.rs`.

use crate::{ObjectRotation, TargetingPolicy};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Renderer path exercised by a benchmark run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BenchmarkRenderPath {
    /// Fixed-orbit episodes rendered through one retained `RenderSession`.
    FixedOrbitSession,
    /// Per-step camera updates rendered through `PersistentRenderer`.
    PersistentSteps,
}

impl BenchmarkRenderPath {
    /// Stable label for reports and result tables.
    pub fn label(self) -> &'static str {
        match self {
            Self::FixedOrbitSession => "fixed-orbit-session",
            Self::PersistentSteps => "persistent-steps",
        }
    }
}

/// Serializable workload description included in benchmark artifacts.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkWorkload {
    pub name: String,
    pub render_path: BenchmarkRenderPath,
    pub objects: Vec<String>,
    pub rotations_per_object: usize,
    pub viewpoints_per_rotation: usize,
    pub steps_per_object: Option<usize>,
    pub total_frames: usize,
}

impl BenchmarkWorkload {
    pub fn fixed_orbit(
        name: impl Into<String>,
        objects: Vec<String>,
        rotations_per_object: usize,
        viewpoints_per_rotation: usize,
    ) -> Self {
        let total_frames =
            fixed_orbit_frame_count(objects.len(), rotations_per_object, viewpoints_per_rotation);
        Self {
            name: name.into(),
            render_path: BenchmarkRenderPath::FixedOrbitSession,
            objects,
            rotations_per_object,
            viewpoints_per_rotation,
            steps_per_object: None,
            total_frames,
        }
    }

    pub fn persistent_steps(
        name: impl Into<String>,
        objects: Vec<String>,
        steps_per_object: usize,
    ) -> Self {
        let total_frames = objects.len().saturating_mul(steps_per_object);
        Self {
            name: name.into(),
            render_path: BenchmarkRenderPath::PersistentSteps,
            objects,
            rotations_per_object: 1,
            viewpoints_per_rotation: 0,
            steps_per_object: Some(steps_per_object),
            total_frames,
        }
    }
}

/// Total captures in a fixed-orbit NeoCortx-style workload.
pub fn fixed_orbit_frame_count(
    object_count: usize,
    rotations_per_object: usize,
    viewpoints_per_rotation: usize,
) -> usize {
    object_count
        .saturating_mul(rotations_per_object)
        .saturating_mul(viewpoints_per_rotation)
}

/// Current NeoCortx targeting behavior for fixed-orbit YCB episodes.
///
/// Yaw-only rotations preserve the historical origin orbit. Pitched or rolled
/// rotations target the rotated mesh center so the object stays in frame.
pub fn neocortx_targeting_policy(rotation: &ObjectRotation) -> TargetingPolicy {
    if rotation.pitch.abs() > f64::EPSILON || rotation.roll.abs() > f64::EPSILON {
        TargetingPolicy::MeshCenter
    } else {
        TargetingPolicy::Origin
    }
}

/// Convert a duration to milliseconds as f64.
pub fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

/// Timing summary over one or more per-frame markers.
///
/// Fixed-orbit session runs measure one marker per object/rotation group
/// (`elapsed_group_ms / viewpoints`). Persistent runs measure one marker per
/// rendered step.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TimingSummary {
    pub total_frames: usize,
    pub measurement_count: usize,
    pub total_ms: f64,
    pub frames_per_second: f64,
    pub mean_ms_per_frame: f64,
    pub p50_ms_per_frame: f64,
    pub p95_ms_per_frame: f64,
    pub min_ms_per_frame: f64,
    pub max_ms_per_frame: f64,
}

impl TimingSummary {
    pub fn empty() -> Self {
        Self {
            total_frames: 0,
            measurement_count: 0,
            total_ms: 0.0,
            frames_per_second: 0.0,
            mean_ms_per_frame: 0.0,
            p50_ms_per_frame: 0.0,
            p95_ms_per_frame: 0.0,
            min_ms_per_frame: 0.0,
            max_ms_per_frame: 0.0,
        }
    }
}

/// Summarize per-frame timing markers.
pub fn summarize_timing(
    per_frame_ms_markers: &[f64],
    total_frames: usize,
    total_ms: f64,
) -> TimingSummary {
    if per_frame_ms_markers.is_empty() || total_frames == 0 {
        return TimingSummary::empty();
    }

    let mean_ms_per_frame = total_ms / total_frames as f64;
    let frames_per_second = if total_ms > 0.0 {
        total_frames as f64 / (total_ms / 1000.0)
    } else {
        0.0
    };

    TimingSummary {
        total_frames,
        measurement_count: per_frame_ms_markers.len(),
        total_ms,
        frames_per_second,
        mean_ms_per_frame,
        p50_ms_per_frame: percentile(per_frame_ms_markers, 0.50).unwrap_or(0.0),
        p95_ms_per_frame: percentile(per_frame_ms_markers, 0.95).unwrap_or(0.0),
        min_ms_per_frame: per_frame_ms_markers
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min),
        max_ms_per_frame: per_frame_ms_markers
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max),
    }
}

/// Nearest-rank percentile over finite samples.
pub fn percentile(samples: &[f64], quantile: f64) -> Option<f64> {
    let mut values: Vec<f64> = samples
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if values.is_empty() {
        return None;
    }

    values.sort_by(|a, b| a.total_cmp(b));
    let q = quantile.clamp(0.0, 1.0);
    let rank = (q * (values.len().saturating_sub(1)) as f64).ceil() as usize;
    values.get(rank).copied()
}
