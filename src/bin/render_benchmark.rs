//! Renderer throughput benchmark for NeoCortx-shaped workloads.
//!
//! Example:
//!   cargo run --release --bin render_benchmark -- --workload fixed-orbit-smoke
//!   cargo run --release --bin render_benchmark -- --workload fixed-orbit-3
//!   cargo run --release --bin render_benchmark -- --workload persistent-smoke

use bevy_sensor::benchmark::{
    duration_ms, neocortx_targeting_policy, summarize_timing, BenchmarkRenderPath,
    BenchmarkWorkload, TimingSummary,
};
use bevy_sensor::ycb;
use bevy_sensor::{
    generate_targeted_viewpoints, BatchRenderRequest, ObjectRotation, PersistentRenderer,
    RenderConfig, RenderHealth, RenderOutput, RenderSession, TargetingPolicy, Vec3,
    ViewpointConfig, REPRESENTATIVE_OBJECTS, TBP_STANDARD_OBJECTS,
};
use image::imageops::{overlay, resize, FilterType};
use image::{Rgba, RgbaImage};
use serde::{Deserialize, Serialize};
use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_DATA_DIR: &str = "/tmp/ycb";
const DEFAULT_OUTPUT_DIR: &str = "output/benchmarks";
const DEFAULT_SAMPLE_LIMIT: usize = 12;
const VISUAL_CELL_SIZE: u32 = 160;
const VISUAL_PADDING: u32 = 12;
const VISUAL_COLUMNS: u32 = 4;
const PERSISTENT_SURFACE_DEFAULT_OBJECT: &str = "025_mug";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WorkloadKind {
    FixedOrbitSmoke,
    FixedOrbit3,
    FixedOrbit10,
    PersistentSmoke,
    PersistentSurface,
}

impl WorkloadKind {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "fixed-orbit-smoke" | "smoke" => Some(Self::FixedOrbitSmoke),
            "fixed-orbit-3" | "neocortx-3" => Some(Self::FixedOrbit3),
            "fixed-orbit-10" | "neocortx-10" => Some(Self::FixedOrbit10),
            "persistent-smoke" => Some(Self::PersistentSmoke),
            "persistent-surface" | "surface" => Some(Self::PersistentSurface),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::FixedOrbitSmoke => "fixed-orbit-smoke",
            Self::FixedOrbit3 => "fixed-orbit-3",
            Self::FixedOrbit10 => "fixed-orbit-10",
            Self::PersistentSmoke => "persistent-smoke",
            Self::PersistentSurface => "persistent-surface",
        }
    }

    fn render_path(self) -> BenchmarkRenderPath {
        match self {
            Self::FixedOrbitSmoke | Self::FixedOrbit3 | Self::FixedOrbit10 => {
                BenchmarkRenderPath::FixedOrbitSession
            }
            Self::PersistentSmoke | Self::PersistentSurface => BenchmarkRenderPath::PersistentSteps,
        }
    }

    fn default_objects(self) -> Vec<String> {
        match self {
            Self::FixedOrbitSmoke | Self::PersistentSmoke => {
                vec![REPRESENTATIVE_OBJECTS[0].to_string()]
            }
            Self::FixedOrbit3 => REPRESENTATIVE_OBJECTS
                .iter()
                .map(|object| (*object).to_string())
                .collect(),
            Self::FixedOrbit10 => TBP_STANDARD_OBJECTS
                .iter()
                .map(|object| (*object).to_string())
                .collect(),
            Self::PersistentSurface => vec![PERSISTENT_SURFACE_DEFAULT_OBJECT.to_string()],
        }
    }

    fn default_rotation_limit(self) -> usize {
        match self {
            Self::FixedOrbitSmoke | Self::PersistentSmoke | Self::PersistentSurface => 1,
            Self::FixedOrbit3 | Self::FixedOrbit10 => 3,
        }
    }

    fn default_viewpoint_limit(self) -> usize {
        match self {
            Self::FixedOrbitSmoke => 3,
            Self::FixedOrbit3 | Self::FixedOrbit10 => ViewpointConfig::default().viewpoint_count(),
            Self::PersistentSmoke | Self::PersistentSurface => {
                ViewpointConfig::default().viewpoint_count()
            }
        }
    }

    fn default_steps(self) -> usize {
        match self {
            Self::PersistentSmoke => 12,
            Self::PersistentSurface => 120,
            Self::FixedOrbitSmoke | Self::FixedOrbit3 | Self::FixedOrbit10 => 0,
        }
    }
}

#[derive(Debug)]
struct Options {
    workload: WorkloadKind,
    data_dir: PathBuf,
    output_dir: PathBuf,
    run_name: Option<String>,
    objects: Option<Vec<String>>,
    rotation_schedule: String,
    max_objects: Option<usize>,
    max_rotations: Option<usize>,
    max_viewpoints: Option<usize>,
    steps: Option<usize>,
    sample_limit: usize,
    download_missing: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct BenchmarkArtifact {
    schema_version: String,
    crate_version: String,
    generated_unix_seconds: u64,
    command: Vec<String>,
    workload: BenchmarkWorkload,
    render_config: RenderConfigArtifact,
    environment: EnvironmentArtifact,
    git: GitArtifact,
    summary: RunSummary,
    groups: Vec<GroupMetric>,
    visual_samples: Vec<VisualSampleMetadata>,
    visual_judge: VisualJudgeArtifact,
}

#[derive(Debug, Serialize, Deserialize)]
struct RenderConfigArtifact {
    width: u32,
    height: u32,
    zoom: f32,
    near_plane: f32,
    far_plane: f32,
}

impl From<&RenderConfig> for RenderConfigArtifact {
    fn from(config: &RenderConfig) -> Self {
        Self {
            width: config.width,
            height: config.height,
            zoom: config.zoom,
            near_plane: config.near_plane,
            far_plane: config.far_plane,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct EnvironmentArtifact {
    os: String,
    arch: String,
    wgpu_backend: Option<String>,
    wgpu_power_pref: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GitArtifact {
    commit: Option<String>,
    branch: Option<String>,
    dirty: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
struct RunSummary {
    timing: TimingSummary,
    health: HealthSummary,
}

#[derive(Debug, Serialize, Deserialize)]
struct HealthSummary {
    total_frames: usize,
    center_foreground_frames: usize,
    blank_frames: usize,
    min_foreground_coverage: f64,
    max_nearest_foreground_distance_px: Option<f64>,
}

#[derive(Default)]
struct HealthAccumulator {
    total_frames: usize,
    center_foreground_frames: usize,
    blank_frames: usize,
    min_foreground_coverage: f64,
    max_nearest_foreground_distance_px: Option<f64>,
}

impl HealthAccumulator {
    fn observe(&mut self, health: &RenderHealth) {
        if self.total_frames == 0 {
            self.min_foreground_coverage = health.foreground_coverage;
        } else {
            self.min_foreground_coverage =
                self.min_foreground_coverage.min(health.foreground_coverage);
        }

        self.total_frames += 1;
        if health.center_foreground {
            self.center_foreground_frames += 1;
        }
        if health.foreground_pixel_count == 0 {
            self.blank_frames += 1;
        }
        if let Some(distance) = health.nearest_foreground_distance_px {
            self.max_nearest_foreground_distance_px = Some(
                self.max_nearest_foreground_distance_px
                    .map(|current| current.max(distance))
                    .unwrap_or(distance),
            );
        }
    }

    fn finish(self) -> HealthSummary {
        HealthSummary {
            total_frames: self.total_frames,
            center_foreground_frames: self.center_foreground_frames,
            blank_frames: self.blank_frames,
            min_foreground_coverage: if self.total_frames == 0 {
                0.0
            } else {
                self.min_foreground_coverage
            },
            max_nearest_foreground_distance_px: self.max_nearest_foreground_distance_px,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct GroupMetric {
    object_id: String,
    rotation_index: usize,
    rotation_euler: [f64; 3],
    target_policy: TargetingPolicy,
    frame_count: usize,
    elapsed_ms: f64,
    ms_per_frame: f64,
    frames_per_second: f64,
    center_foreground_frames: usize,
    blank_frames: usize,
    min_foreground_coverage: f64,
    persistent_init_ms: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VisualSampleMetadata {
    sample_index: usize,
    frame_index: usize,
    object_id: String,
    rotation_index: usize,
    viewpoint_index: Option<usize>,
    step_index: Option<usize>,
    rotation_euler: [f64; 3],
    target_policy: TargetingPolicy,
    rgb_file: String,
    depth_file: String,
    center_foreground: bool,
    foreground_coverage: f64,
    nearest_foreground_distance_px: Option<f64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct VisualJudgeArtifact {
    grid_file: String,
    prompt_file: String,
    samples_file: String,
    judge_required: bool,
}

struct VisualTile {
    rgb: RgbaImage,
    depth: RgbaImage,
}

struct VisualSampleInput<'a> {
    frame_index: usize,
    object_id: &'a str,
    rotation_index: usize,
    viewpoint_index: Option<usize>,
    step_index: Option<usize>,
    rotation: &'a ObjectRotation,
    target_policy: &'a TargetingPolicy,
    rgba: &'a [u8],
    depth: &'a [f64],
    width: u32,
    height: u32,
    health: &'a RenderHealth,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("Error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let options = parse_options()?;
    bevy_sensor::initialize();

    let render_config = RenderConfig::tbp_default();
    let viewpoint_config = ViewpointConfig::default();
    let mut objects = options
        .objects
        .clone()
        .unwrap_or_else(|| options.workload.default_objects());
    if let Some(max_objects) = options.max_objects {
        objects.truncate(max_objects);
    }
    if objects.is_empty() {
        return Err("benchmark workload has no objects".into());
    }

    let mut rotations = rotations_from_schedule(&options.rotation_schedule)?;
    let rotation_limit = options
        .max_rotations
        .unwrap_or_else(|| options.workload.default_rotation_limit());
    rotations.truncate(rotation_limit.max(1));
    if rotations.is_empty() {
        return Err("benchmark workload has no rotations".into());
    }

    let viewpoint_limit = options
        .max_viewpoints
        .unwrap_or_else(|| options.workload.default_viewpoint_limit())
        .max(1)
        .min(viewpoint_config.viewpoint_count());
    let steps = options
        .steps
        .unwrap_or_else(|| options.workload.default_steps());

    ensure_ycb_objects(&options.data_dir, &objects, options.download_missing)?;
    let data_dir = fs::canonicalize(&options.data_dir).unwrap_or_else(|_| options.data_dir.clone());

    let workload = match options.workload.render_path() {
        BenchmarkRenderPath::FixedOrbitSession => BenchmarkWorkload::fixed_orbit(
            options.workload.label(),
            objects.clone(),
            rotations.len(),
            viewpoint_limit,
        ),
        BenchmarkRenderPath::PersistentSteps => BenchmarkWorkload::persistent_steps(
            options.workload.label(),
            objects.clone(),
            steps.max(1),
        ),
    };

    let generated_unix_seconds = unix_seconds();
    let run_name = options
        .run_name
        .clone()
        .unwrap_or_else(|| format!("{}-{}", options.workload.label(), generated_unix_seconds));
    let run_dir = options.output_dir.join(run_name);
    let samples_dir = run_dir.join("visual_samples");
    fs::create_dir_all(&samples_dir)?;

    println!("bevy-sensor render benchmark");
    println!("  workload: {}", workload.name);
    println!("  path: {}", workload.render_path.label());
    println!("  objects: {}", workload.objects.join(","));
    println!("  frames: {}", workload.total_frames);
    println!("  output: {}", run_dir.display());

    let mut health = HealthAccumulator::default();
    let mut groups = Vec::new();
    let mut per_frame_ms_markers = Vec::new();
    let mut visual_samples = Vec::new();
    let mut visual_tiles = Vec::new();
    let mut frame_index = 0usize;
    let total_start = Instant::now();

    match options.workload.render_path() {
        BenchmarkRenderPath::FixedOrbitSession => {
            run_fixed_orbit(
                &data_dir,
                &objects,
                &rotations,
                &viewpoint_config,
                viewpoint_limit,
                &render_config,
                workload.total_frames,
                options.sample_limit,
                &samples_dir,
                &mut frame_index,
                &mut health,
                &mut groups,
                &mut per_frame_ms_markers,
                &mut visual_samples,
                &mut visual_tiles,
            )?;
        }
        BenchmarkRenderPath::PersistentSteps => {
            run_persistent_steps(
                &data_dir,
                &objects,
                &rotations[0],
                &viewpoint_config,
                viewpoint_limit,
                steps.max(1),
                &render_config,
                workload.total_frames,
                options.sample_limit,
                &samples_dir,
                &mut frame_index,
                &mut health,
                &mut groups,
                &mut per_frame_ms_markers,
                &mut visual_samples,
                &mut visual_tiles,
            )?;
        }
    }

    let total_ms = duration_ms(total_start.elapsed());
    let timing = summarize_timing(&per_frame_ms_markers, workload.total_frames, total_ms);
    let visual_grid = contact_sheet(&visual_tiles);
    let grid_path = run_dir.join("visual_grid.png");
    visual_grid.save(&grid_path)?;

    let samples_path = run_dir.join("visual_samples.json");
    fs::write(
        &samples_path,
        serde_json::to_string_pretty(&visual_samples)?,
    )?;

    let prompt_path = run_dir.join("visual_judge_prompt.md");
    write_visual_judge_prompt(&prompt_path, &workload, &visual_samples)?;

    let artifact = BenchmarkArtifact {
        schema_version: "bevy-sensor-render-benchmark-v1".to_string(),
        crate_version: env!("CARGO_PKG_VERSION").to_string(),
        generated_unix_seconds,
        command: env::args().collect(),
        workload,
        render_config: RenderConfigArtifact::from(&render_config),
        environment: collect_environment(),
        git: collect_git(),
        summary: RunSummary {
            timing,
            health: health.finish(),
        },
        groups,
        visual_samples,
        visual_judge: VisualJudgeArtifact {
            grid_file: "visual_grid.png".to_string(),
            prompt_file: "visual_judge_prompt.md".to_string(),
            samples_file: "visual_samples.json".to_string(),
            judge_required: true,
        },
    };

    let metrics_path = run_dir.join("metrics.json");
    fs::write(&metrics_path, serde_json::to_string_pretty(&artifact)?)?;
    let report_path = run_dir.join("report.md");
    write_report(&report_path, &artifact)?;

    println!();
    println!("summary:");
    println!(
        "  {:.2} fps, mean {:.2} ms/frame, p50 {:.2} ms, p95 {:.2} ms",
        artifact.summary.timing.frames_per_second,
        artifact.summary.timing.mean_ms_per_frame,
        artifact.summary.timing.p50_ms_per_frame,
        artifact.summary.timing.p95_ms_per_frame
    );
    println!(
        "  center foreground: {}/{} frames, blank frames: {}",
        artifact.summary.health.center_foreground_frames,
        artifact.summary.health.total_frames,
        artifact.summary.health.blank_frames
    );
    println!("  metrics: {}", metrics_path.display());
    println!("  report: {}", report_path.display());
    println!("  visual grid: {}", grid_path.display());
    println!("  judge prompt: {}", prompt_path.display());

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_fixed_orbit(
    data_dir: &Path,
    objects: &[String],
    rotations: &[ObjectRotation],
    viewpoint_config: &ViewpointConfig,
    viewpoint_limit: usize,
    render_config: &RenderConfig,
    total_frames: usize,
    sample_limit: usize,
    samples_dir: &Path,
    frame_index: &mut usize,
    health_accumulator: &mut HealthAccumulator,
    groups: &mut Vec<GroupMetric>,
    per_frame_ms_markers: &mut Vec<f64>,
    visual_samples: &mut Vec<VisualSampleMetadata>,
    visual_tiles: &mut Vec<VisualTile>,
) -> Result<(), Box<dyn Error>> {
    let mut session = RenderSession::new(render_config)?;

    for object_id in objects {
        let object_dir = data_dir.join(object_id);
        for (rotation_index, rotation) in rotations.iter().enumerate() {
            let target_policy = neocortx_targeting_policy(rotation);
            let targeted = generate_targeted_viewpoints(
                &object_dir,
                viewpoint_config,
                rotation,
                &target_policy,
            )?;
            let viewpoints: Vec<_> = targeted
                .viewpoints
                .iter()
                .copied()
                .take(viewpoint_limit)
                .collect();
            let requests: Vec<_> = viewpoints
                .iter()
                .map(|viewpoint| BatchRenderRequest {
                    object_dir: object_dir.clone(),
                    viewpoint: *viewpoint,
                    object_rotation: rotation.clone(),
                    render_config: render_config.clone(),
                    target_point: targeted.target_point,
                    targeting_policy: targeted.policy.clone(),
                })
                .collect();

            let group_start = Instant::now();
            let outputs = session.render(&requests)?;
            let elapsed_ms = duration_ms(group_start.elapsed());
            let frame_count = outputs.len();
            let ms_per_frame = elapsed_ms / frame_count.max(1) as f64;
            let frames_per_second = if elapsed_ms > 0.0 {
                frame_count as f64 / (elapsed_ms / 1000.0)
            } else {
                0.0
            };
            per_frame_ms_markers.push(ms_per_frame);

            let mut group_health = HealthAccumulator::default();
            for (viewpoint_index, output) in outputs.iter().enumerate() {
                group_health.observe(&output.health);
                health_accumulator.observe(&output.health);
                maybe_capture_visual_sample(
                    VisualSampleInput {
                        frame_index: *frame_index,
                        object_id,
                        rotation_index,
                        viewpoint_index: Some(viewpoint_index),
                        step_index: None,
                        rotation,
                        target_policy: &target_policy,
                        rgba: &output.rgba,
                        depth: &output.depth,
                        width: output.width,
                        height: output.height,
                        health: &output.health,
                    },
                    render_config.far_plane,
                    total_frames,
                    sample_limit,
                    samples_dir,
                    visual_samples,
                    visual_tiles,
                )?;
                *frame_index += 1;
            }
            let group_health = group_health.finish();

            groups.push(GroupMetric {
                object_id: object_id.clone(),
                rotation_index,
                rotation_euler: rotation_euler(rotation),
                target_policy,
                frame_count,
                elapsed_ms,
                ms_per_frame,
                frames_per_second,
                center_foreground_frames: group_health.center_foreground_frames,
                blank_frames: group_health.blank_frames,
                min_foreground_coverage: group_health.min_foreground_coverage,
                persistent_init_ms: None,
            });

            println!(
                "  {} r{}: {} frames in {:.1} ms ({:.2} fps)",
                object_id, rotation_index, frame_count, elapsed_ms, frames_per_second
            );
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_persistent_steps(
    data_dir: &Path,
    objects: &[String],
    rotation: &ObjectRotation,
    viewpoint_config: &ViewpointConfig,
    viewpoint_limit: usize,
    steps_per_object: usize,
    render_config: &RenderConfig,
    total_frames: usize,
    sample_limit: usize,
    samples_dir: &Path,
    frame_index: &mut usize,
    health_accumulator: &mut HealthAccumulator,
    groups: &mut Vec<GroupMetric>,
    per_frame_ms_markers: &mut Vec<f64>,
    visual_samples: &mut Vec<VisualSampleMetadata>,
    visual_tiles: &mut Vec<VisualTile>,
) -> Result<(), Box<dyn Error>> {
    for object_id in objects {
        let object_dir = data_dir.join(object_id);
        let target_policy = neocortx_targeting_policy(rotation);
        let targeted =
            generate_targeted_viewpoints(&object_dir, viewpoint_config, rotation, &target_policy)?;
        let viewpoints: Vec<_> = targeted
            .viewpoints
            .iter()
            .copied()
            .take(viewpoint_limit)
            .collect();
        if viewpoints.is_empty() {
            return Err("persistent workload generated no viewpoints".into());
        }

        let init_start = Instant::now();
        let mut renderer = PersistentRenderer::new(&object_dir, render_config)?;
        let init_ms = duration_ms(init_start.elapsed());
        let group_start = Instant::now();
        let mut group_health = HealthAccumulator::default();

        for step_index in 0..steps_per_object {
            let viewpoint_index = step_index % viewpoints.len();
            let render_start = Instant::now();
            let output = renderer
                .render(&viewpoints[viewpoint_index], rotation)?
                .with_targeting(targeted.target_point, targeted.policy.clone());
            let render_ms = duration_ms(render_start.elapsed());
            per_frame_ms_markers.push(render_ms);

            let health = output.health_with_far_plane(render_config.far_plane as f64);
            group_health.observe(&health);
            health_accumulator.observe(&health);
            maybe_capture_visual_sample(
                VisualSampleInput {
                    frame_index: *frame_index,
                    object_id,
                    rotation_index: 0,
                    viewpoint_index: Some(viewpoint_index),
                    step_index: Some(step_index),
                    rotation,
                    target_policy: &target_policy,
                    rgba: &output.rgba,
                    depth: &output.depth,
                    width: output.width,
                    height: output.height,
                    health: &health,
                },
                render_config.far_plane,
                total_frames,
                sample_limit,
                samples_dir,
                visual_samples,
                visual_tiles,
            )?;
            *frame_index += 1;
        }

        let elapsed_ms = duration_ms(group_start.elapsed());
        let ms_per_frame = elapsed_ms / steps_per_object as f64;
        let frames_per_second = if elapsed_ms > 0.0 {
            steps_per_object as f64 / (elapsed_ms / 1000.0)
        } else {
            0.0
        };
        let group_health = group_health.finish();

        groups.push(GroupMetric {
            object_id: object_id.clone(),
            rotation_index: 0,
            rotation_euler: rotation_euler(rotation),
            target_policy,
            frame_count: steps_per_object,
            elapsed_ms,
            ms_per_frame,
            frames_per_second,
            center_foreground_frames: group_health.center_foreground_frames,
            blank_frames: group_health.blank_frames,
            min_foreground_coverage: group_health.min_foreground_coverage,
            persistent_init_ms: Some(init_ms),
        });

        println!(
            "  {} persistent: init {:.1} ms, {} steps in {:.1} ms ({:.2} fps)",
            object_id, init_ms, steps_per_object, elapsed_ms, frames_per_second
        );
    }

    Ok(())
}

fn maybe_capture_visual_sample(
    input: VisualSampleInput<'_>,
    far_plane: f32,
    total_frames: usize,
    sample_limit: usize,
    samples_dir: &Path,
    visual_samples: &mut Vec<VisualSampleMetadata>,
    visual_tiles: &mut Vec<VisualTile>,
) -> Result<(), Box<dyn Error>> {
    if sample_limit == 0 || visual_samples.len() >= sample_limit {
        return Ok(());
    }
    let stride = (total_frames / sample_limit.max(1)).max(1);
    let is_last = input.frame_index + 1 == total_frames;
    if !is_stride_sample(input.frame_index, stride) && !is_last {
        return Ok(());
    }

    let sample_index = visual_samples.len();
    let prefix = format!(
        "{:04}_{}_r{:02}_{}",
        sample_index,
        input.object_id,
        input.rotation_index,
        input
            .step_index
            .map(|step| format!("s{step:04}"))
            .or_else(|| input.viewpoint_index.map(|view| format!("v{view:02}")))
            .unwrap_or_else(|| "frame".to_string())
    );
    let rgb_file = format!("visual_samples/{prefix}_rgb.png");
    let depth_file = format!("visual_samples/{prefix}_depth.png");
    let rgb_path = samples_dir.join(format!("{prefix}_rgb.png"));
    let depth_path = samples_dir.join(format!("{prefix}_depth.png"));

    let rgb = rgb_tile(
        input.rgba,
        input.depth,
        input.width,
        input.height,
        far_plane,
    );
    let depth = depth_tile(input.depth, input.width, input.height, far_plane);
    rgb.save(&rgb_path)?;
    depth.save(&depth_path)?;

    visual_tiles.push(VisualTile {
        rgb: rgb.clone(),
        depth: depth.clone(),
    });
    visual_samples.push(VisualSampleMetadata {
        sample_index,
        frame_index: input.frame_index,
        object_id: input.object_id.to_string(),
        rotation_index: input.rotation_index,
        viewpoint_index: input.viewpoint_index,
        step_index: input.step_index,
        rotation_euler: rotation_euler(input.rotation),
        target_policy: input.target_policy.clone(),
        rgb_file,
        depth_file,
        center_foreground: input.health.center_foreground,
        foreground_coverage: input.health.foreground_coverage,
        nearest_foreground_distance_px: input.health.nearest_foreground_distance_px,
    });

    Ok(())
}

#[allow(clippy::manual_is_multiple_of)]
fn is_stride_sample(frame_index: usize, stride: usize) -> bool {
    frame_index % stride == 0
}

fn rgb_tile(rgba: &[u8], depth: &[f64], width: u32, height: u32, far_plane: f32) -> RgbaImage {
    let mut tile = RgbaImage::from_pixel(width, height, background_pixel());
    let far = far_plane as f64 * 0.99;

    for y in 0..height {
        for x in 0..width {
            let pixel_index = (y * width + x) as usize;
            if depth.get(pixel_index).copied().unwrap_or(far_plane as f64) >= far {
                continue;
            }

            let rgba_index = pixel_index * 4;
            if rgba_index + 3 < rgba.len() {
                tile.put_pixel(
                    x,
                    y,
                    Rgba([
                        rgba[rgba_index],
                        rgba[rgba_index + 1],
                        rgba[rgba_index + 2],
                        255,
                    ]),
                );
            }
        }
    }

    tile
}

fn depth_tile(depth: &[f64], width: u32, height: u32, far_plane: f32) -> RgbaImage {
    let mut min_depth = f64::INFINITY;
    let mut max_depth = f64::NEG_INFINITY;
    let far = far_plane as f64 * 0.99;

    for value in depth.iter().copied().filter(|value| *value < far) {
        min_depth = min_depth.min(value);
        max_depth = max_depth.max(value);
    }

    let range = (max_depth - min_depth).max(1e-9);
    let mut tile = RgbaImage::from_pixel(width, height, background_pixel());

    for y in 0..height {
        for x in 0..width {
            let index = (y * width + x) as usize;
            let Some(value) = depth.get(index).copied() else {
                continue;
            };
            if value >= far {
                continue;
            }

            let normalized = ((value - min_depth) / range).clamp(0.0, 1.0);
            tile.put_pixel(x, y, depth_color(1.0 - normalized));
        }
    }

    tile
}

fn contact_sheet(tiles: &[VisualTile]) -> RgbaImage {
    if tiles.is_empty() {
        return RgbaImage::from_pixel(1, 1, sheet_pixel());
    }

    let columns = VISUAL_COLUMNS.min(tiles.len() as u32);
    let groups = (tiles.len() as u32).div_ceil(columns);
    let rows = groups * 2;
    let width = columns * VISUAL_CELL_SIZE + (columns + 1) * VISUAL_PADDING;
    let height = rows * VISUAL_CELL_SIZE + (rows + 1) * VISUAL_PADDING;
    let mut sheet = RgbaImage::from_pixel(width, height, sheet_pixel());

    for (index, tile_pair) in tiles.iter().enumerate() {
        let index = index as u32;
        let column = index % columns;
        let group = index / columns;
        let x = VISUAL_PADDING + column * (VISUAL_CELL_SIZE + VISUAL_PADDING);
        let rgb_y = VISUAL_PADDING + group * 2 * (VISUAL_CELL_SIZE + VISUAL_PADDING);
        let depth_y = rgb_y + VISUAL_CELL_SIZE + VISUAL_PADDING;

        let rgb = resize(
            &tile_pair.rgb,
            VISUAL_CELL_SIZE,
            VISUAL_CELL_SIZE,
            FilterType::Lanczos3,
        );
        let depth = resize(
            &tile_pair.depth,
            VISUAL_CELL_SIZE,
            VISUAL_CELL_SIZE,
            FilterType::Lanczos3,
        );

        overlay(&mut sheet, &rgb, i64::from(x), i64::from(rgb_y));
        overlay(&mut sheet, &depth, i64::from(x), i64::from(depth_y));
    }

    sheet
}

fn depth_color(value: f64) -> Rgba<u8> {
    let value = value.clamp(0.0, 1.0);
    let (start, end, t) = if value < 0.5 {
        ([33.0, 76.0, 164.0], [24.0, 190.0, 166.0], value * 2.0)
    } else {
        (
            [24.0, 190.0, 166.0],
            [250.0, 204.0, 76.0],
            (value - 0.5) * 2.0,
        )
    };

    Rgba([
        lerp(start[0], end[0], t) as u8,
        lerp(start[1], end[1], t) as u8,
        lerp(start[2], end[2], t) as u8,
        255,
    ])
}

fn lerp(start: f64, end: f64, t: f64) -> f64 {
    start + (end - start) * t
}

fn background_pixel() -> Rgba<u8> {
    Rgba([244, 246, 249, 255])
}

fn sheet_pixel() -> Rgba<u8> {
    Rgba([226, 231, 237, 255])
}

fn write_report(path: &Path, artifact: &BenchmarkArtifact) -> Result<(), Box<dyn Error>> {
    let mut rows = String::new();
    for group in &artifact.groups {
        rows.push_str(&format!(
            "| {} | {} | {} | {} | {:.2} | {:.2} | {}/{} | {} |\n",
            group.object_id,
            group.rotation_index,
            group.frame_count,
            group.target_policy.label(),
            group.ms_per_frame,
            group.frames_per_second,
            group.center_foreground_frames,
            group.frame_count,
            group.blank_frames
        ));
    }

    let report = format!(
        "# bevy-sensor render benchmark\n\n\
         ## Summary\n\n\
         - Workload: `{}`\n\
         - Render path: `{}`\n\
         - Frames: `{}`\n\
         - Measurements: `{}`\n\
         - Throughput: `{:.2}` fps\n\
         - Mean: `{:.2}` ms/frame\n\
         - P50/P95: `{:.2}` / `{:.2}` ms/frame\n\
         - Center-foreground frames: `{}/{}`\n\
         - Blank frames: `{}`\n\n\
         ## Artifacts\n\n\
         - Metrics: `metrics.json`\n\
         - Visual grid: `visual_grid.png`\n\
         - Visual judge prompt: `visual_judge_prompt.md`\n\
         - Visual sample metadata: `visual_samples.json`\n\n\
         ## Groups\n\n\
         | object | rotation | frames | target | ms/frame | fps | center hits | blanks |\n\
         | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |\n\
         {}",
        artifact.workload.name,
        artifact.workload.render_path.label(),
        artifact.summary.timing.total_frames,
        artifact.summary.timing.measurement_count,
        artifact.summary.timing.frames_per_second,
        artifact.summary.timing.mean_ms_per_frame,
        artifact.summary.timing.p50_ms_per_frame,
        artifact.summary.timing.p95_ms_per_frame,
        artifact.summary.health.center_foreground_frames,
        artifact.summary.health.total_frames,
        artifact.summary.health.blank_frames,
        rows
    );

    fs::write(path, report)?;
    Ok(())
}

fn write_visual_judge_prompt(
    path: &Path,
    workload: &BenchmarkWorkload,
    samples: &[VisualSampleMetadata],
) -> Result<(), Box<dyn Error>> {
    let mut sample_lines = String::new();
    for sample in samples {
        let viewpoint = sample
            .viewpoint_index
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_string());
        let step = sample
            .step_index
            .map(|value| value.to_string())
            .unwrap_or_else(|| "none".to_string());
        sample_lines.push_str(&format!(
            "- sample {}: object `{}`, rotation {}, viewpoint {}, step {}, center_foreground={}, coverage={:.4}\n",
            sample.sample_index,
            sample.object_id,
            sample.rotation_index,
            viewpoint,
            step,
            sample.center_foreground,
            sample.foreground_coverage,
        ));
    }

    let prompt = format!(
        "# Visual Judge Prompt\n\n\
         You are judging bevy-sensor render quality for a throughput benchmark. \
         The deterministic metric is throughput; this visual check is a regression guard.\n\n\
         Inspect `visual_grid.png`. Each sample has two tiles: RGB first, depth directly below it. \
         Samples are ordered left-to-right, then top-to-bottom by sample index.\n\n\
         Workload: `{}` (`{}`), total frames: `{}`.\n\n\
         Samples:\n{}\n\
         Return exactly one JSON object with this schema:\n\n\
         ```json\n\
         {{\"verdict\":\"PASS|FAIL\",\"confidence\":0.0,\"issues\":[\"...\"],\"notes\":\"...\"}}\n\
         ```\n\n\
         PASS only if the RGB tiles show visible YCB objects and the depth tiles have coherent \
         object-shaped foreground. FAIL for blank frames, missing objects, severe off-center \
         framing, depth silhouettes that do not match RGB, texture corruption, or obvious \
         orientation/targeting regressions. Ignore small lighting differences.\n",
        workload.name,
        workload.render_path.label(),
        workload.total_frames,
        sample_lines
    );

    fs::write(path, prompt)?;
    Ok(())
}

fn parse_options() -> Result<Options, Box<dyn Error>> {
    let args: Vec<String> = env::args().collect();
    if has_flag(&args, "--help") || has_flag(&args, "-h") {
        print_help();
        std::process::exit(0);
    }

    let workload_name =
        parse_arg(&args, "--workload").unwrap_or_else(|| "fixed-orbit-smoke".to_string());
    let workload = WorkloadKind::parse(&workload_name).ok_or_else(|| {
        format!(
            "invalid --workload '{}'; supported: fixed-orbit-smoke, fixed-orbit-3, fixed-orbit-10, persistent-smoke, persistent-surface",
            workload_name
        )
    })?;

    let data_dir = parse_arg(&args, "--data-dir")
        .or_else(|| env::var("BEVY_SENSOR_YCB_DIR").ok())
        .unwrap_or_else(|| DEFAULT_DATA_DIR.to_string());
    let output_dir =
        parse_arg(&args, "--output-dir").unwrap_or_else(|| DEFAULT_OUTPUT_DIR.to_string());
    let objects = parse_arg(&args, "--objects").map(|value| {
        value
            .split(',')
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>()
    });

    Ok(Options {
        workload,
        data_dir: PathBuf::from(data_dir),
        output_dir: PathBuf::from(output_dir),
        run_name: parse_arg(&args, "--run-name"),
        objects,
        rotation_schedule: parse_arg(&args, "--rotation-schedule")
            .unwrap_or_else(|| "tbp-parity".to_string()),
        max_objects: parse_arg(&args, "--max-objects").and_then(|value| value.parse().ok()),
        max_rotations: parse_arg(&args, "--max-rotations").and_then(|value| value.parse().ok()),
        max_viewpoints: parse_arg(&args, "--max-viewpoints").and_then(|value| value.parse().ok()),
        steps: parse_arg(&args, "--steps").and_then(|value| value.parse().ok()),
        sample_limit: parse_arg(&args, "--sample-limit")
            .and_then(|value| value.parse().ok())
            .unwrap_or(DEFAULT_SAMPLE_LIMIT),
        download_missing: !has_flag(&args, "--no-download"),
    })
}

fn print_help() {
    println!(
        "Usage: cargo run --release --bin render_benchmark -- [options]\n\n\
         Options:\n\
           --workload <name>          fixed-orbit-smoke | fixed-orbit-3 | fixed-orbit-10 | persistent-smoke | persistent-surface\n\
           --data-dir <path>          YCB root (default: BEVY_SENSOR_YCB_DIR or /tmp/ycb)\n\
           --output-dir <path>        Artifact root (default: output/benchmarks)\n\
           --run-name <name>          Stable run directory name\n\
           --objects <a,b,c>          Override workload object list\n\
           --rotation-schedule <name> tbp-parity | tbp-known\n\
           --max-objects <n>          Truncate object list\n\
           --max-rotations <n>        Truncate rotations\n\
           --max-viewpoints <n>       Truncate viewpoint orbit\n\
           --steps <n>                Persistent steps per object\n\
           --sample-limit <n>         Visual samples to save (default: 12)\n\
           --no-download              Fail if required YCB objects are missing"
    );
}

fn parse_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|arg| arg == flag)
        .and_then(|index| args.get(index + 1).cloned())
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|arg| arg == flag)
}

fn rotations_from_schedule(schedule: &str) -> Result<Vec<ObjectRotation>, Box<dyn Error>> {
    match schedule {
        "tbp-parity" | "tbp-benchmark" | "benchmark" => {
            Ok(ObjectRotation::tbp_benchmark_rotations())
        }
        "tbp-known" | "tbp-known-orientations" | "known" | "full" => {
            Ok(ObjectRotation::tbp_known_orientations())
        }
        other => Err(format!(
            "invalid --rotation-schedule '{}'; supported: tbp-parity, tbp-known",
            other
        )
        .into()),
    }
}

fn ensure_ycb_objects(
    ycb_dir: &Path,
    object_ids: &[String],
    download_missing: bool,
) -> Result<(), Box<dyn Error>> {
    let refs: Vec<&str> = object_ids.iter().map(String::as_str).collect();
    let missing = ycb::missing_objects(ycb_dir, &refs);
    if missing.is_empty() {
        return Ok(());
    }
    if !download_missing {
        return Err(format!(
            "missing YCB objects at {}: {:?}",
            ycb_dir.display(),
            missing
        )
        .into());
    }

    println!(
        "missing YCB objects at {}: {:?}",
        ycb_dir.display(),
        missing
    );
    println!("downloading missing objects...");
    let missing_refs: Vec<&str> = missing.iter().map(String::as_str).collect();
    bevy_sensor::ycbust::blocking::download_objects_blocking(
        &missing_refs,
        ycb_dir,
        bevy_sensor::ycbust::DownloadOptions::default(),
    )?;

    let still_missing = ycb::missing_objects(ycb_dir, &refs);
    if !still_missing.is_empty() {
        return Err(format!(
            "YCB download completed but objects are still incomplete: {:?}",
            still_missing
        )
        .into());
    }
    Ok(())
}

fn collect_environment() -> EnvironmentArtifact {
    EnvironmentArtifact {
        os: env::consts::OS.to_string(),
        arch: env::consts::ARCH.to_string(),
        wgpu_backend: env::var("WGPU_BACKEND").ok(),
        wgpu_power_pref: env::var("WGPU_POWER_PREF").ok(),
    }
}

fn collect_git() -> GitArtifact {
    GitArtifact {
        commit: git_output(&["rev-parse", "--short", "HEAD"]),
        branch: git_output(&["branch", "--show-current"]),
        dirty: git_output(&["status", "--porcelain"]).map(|status| !status.trim().is_empty()),
    }
}

fn git_output(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    Some(text.trim().to_string())
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

fn rotation_euler(rotation: &ObjectRotation) -> [f64; 3] {
    [rotation.pitch, rotation.yaw, rotation.roll]
}

#[allow(dead_code)]
fn _render_output_health(output: &RenderOutput, far_plane: f32) -> RenderHealth {
    output.health_with_far_plane(far_plane as f64)
}

#[allow(dead_code)]
fn _vec3_array(value: Vec3) -> [f32; 3] {
    value.to_array()
}
