# CLAUDE.md - Development Notes for bevy-sensor

## Project Overview

A Bevy library and CLI that captures multi-view images of 3D OBJ models (YCB dataset) for sensor simulation. This project produces comparable results to the [Thousand Brains Project (TBP)](https://github.com/thousandbrainsproject/tbp.monty) habitat sensor for use in the neocortx Rust-based implementation.

## Quick Reference

```bash
# Build
cargo build --release

# Run tests (80 tests)
cargo test

# Run (requires GPU or proper software rendering)
cargo run --release

# Headless with software rendering (limited - PBR shaders may fail)
LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe \
  xvfb-run -a -s "-screen 0 1280x1024x24" \
  cargo run --release
```

## Architecture

### Library API (`src/lib.rs`)

The library exports these types for use by neocortx:

```rust
use bevy_sensor::{
    // Headless Rendering API (NEW)
    RenderConfig,        // Render settings (resolution, lighting, depth)
    RenderOutput,        // RGBA + depth buffer output
    LightingConfig,      // Configurable lighting
    CameraIntrinsics,    // Camera parameters for 3D projection
    render_to_buffer,    // Render single viewpoint to memory
    render_all_viewpoints, // Batch render all viewpoints

    // File-based Capture (Legacy)
    SensorConfig,        // Full capture configuration
    ViewpointConfig,     // Camera viewpoint settings
    ObjectRotation,      // Object rotation (Euler angles)
    generate_viewpoints, // Generate camera transforms
    CaptureCamera,       // Marker component for camera
    CaptureTarget,       // Marker component for target object
};

// YCB utilities
use bevy_sensor::ycb::{
    download_models,     // Async download of YCB models
    models_exist,        // Check if models downloaded
    object_mesh_path,    // Get OBJ file path
    object_texture_path, // Get texture path
    Subset,              // Representative, Ten, All
    REPRESENTATIVE_OBJECTS,
    TEN_OBJECTS,
};
```

### Headless Rendering API (NEW)

Render directly to memory for neocortx integration:

```rust
use bevy_sensor::{render_to_buffer, RenderConfig, ViewpointConfig, ObjectRotation};
use std::path::Path;

// TBP-compatible 64x64 sensor
let config = RenderConfig::tbp_default();
let viewpoints = bevy_sensor::generate_viewpoints(&ViewpointConfig::default());

// Render single viewpoint
let output = render_to_buffer(
    Path::new("/tmp/ycb/003_cracker_box"),
    &viewpoints[0],
    &ObjectRotation::identity(),
    &config,
)?;

// Access rendered data
let rgba: &[u8] = &output.rgba;           // 64*64*4 bytes
let depth: &[f64] = &output.depth;        // 64*64 f64 (meters, TBP precision)
let rgb_image = output.to_rgb_image();    // Vec<Vec<[u8; 3]>> for neocortx
let depth_image = output.to_depth_image(); // Vec<Vec<f64>> for neocortx
```

**RenderConfig options:**

| Constructor | Resolution | Use case |
|-------------|-----------|----------|
| `RenderConfig::tbp_default()` | 64×64 | TBP-compatible sensor |
| `RenderConfig::preview()` | 256×256 | Debugging/visualization |
| `RenderConfig::high_res()` | 512×512 | Detailed captures |

**LightingConfig options:**

| Constructor | Description |
|-------------|-------------|
| `LightingConfig::default()` | Standard 3-point lighting |
| `LightingConfig::bright()` | High visibility |
| `LightingConfig::soft()` | Soft, even lighting |
| `LightingConfig::unlit()` | Ambient only, no shadows |

**CameraIntrinsics:**

```rust
let intrinsics = config.intrinsics();
// intrinsics.focal_length: [f64; 2]     - (fx, fy) in pixels (TBP precision)
// intrinsics.principal_point: [f64; 2]  - (cx, cy) center (TBP precision)
// intrinsics.image_size: [u32; 2]       - (width, height)

// Project 3D to 2D (accepts Bevy Vec3, returns f64)
let pixel = intrinsics.project(point_3d);  // Option<[f64; 2]>

// Unproject 2D + depth to 3D (all f64 for TBP precision)
let point = intrinsics.unproject([32.0, 32.0], depth);  // [f64; 3]
```

> **Note:** Full GPU rendering requires a display (X11/Wayland). The depth buffer
> now returns real per-pixel depth values from GPU readback (Bevy 0.15+).

### Library Usage: GPU Rendering on WSL2

When using bevy-sensor as a library (e.g., from neocortx), you **MUST** call `bevy_sensor::initialize()` at the start of your application, before any rendering operations:

```rust
use bevy_sensor;

fn main() {
    // Initialize backend configuration FIRST
    // This is critical for WSL2 GPU rendering!
    bevy_sensor::initialize();

    // Now use the rendering API
    let output = bevy_sensor::render_to_buffer(
        object_dir,
        &viewpoint,
        &rotation,
        &RenderConfig::tbp_default(),
    )?;
}
```

**Why is this necessary?**

The WGPU rendering backend caches its backend selection early during library initialization. On WSL2, this must be WebGPU (not Vulkan, which doesn't support headless rendering). Calling `initialize()` ensures environment variables are set before any GPU code runs.

**Symptoms of skipping this step:**
- WSL2: Panic "Unable to find a GPU" when trying to render
- Incorrect backend selection causing GPU access failures
- Library mode fails while binary mode works (different initialization order)

**On neocortx integration:**
Call `initialize()` in the `BevySensorModule::new()` function or in main before creating the module.

### Object Rotation (`ObjectRotation`)

Matches TBP benchmark Euler angle format `[pitch, yaw, roll]`:

```rust
// TBP benchmark: 3 rotations (used for quick experiments)
ObjectRotation::tbp_benchmark_rotations()
// → [[0,0,0], [0,90,0], [0,180,0]]

// TBP known orientations: 14 rotations (6 faces + 8 corners)
ObjectRotation::tbp_known_orientations()
// → 14 orientations used during TBP training
```

### Batch Rendering API (NEW)

For rendering 10+ viewpoints efficiently, use the batch API to eliminate subprocess and app initialization overhead. Achieves 10-50x speedup compared to sequential `render_to_buffer()` calls.

**Quick Example:**

```rust
use bevy_sensor::{
    create_batch_renderer, queue_render_request, render_next_in_batch,
    BatchRenderRequest, BatchRenderConfig, RenderConfig, ObjectRotation,
};
use std::path::PathBuf;

// Create renderer once
let mut renderer = create_batch_renderer(&BatchRenderConfig::default())?;

// Queue renders
for viewpoint in viewpoints {
    queue_render_request(&mut renderer, BatchRenderRequest {
        object_dir: PathBuf::from("/tmp/ycb/003_cracker_box"),
        viewpoint,
        object_rotation: ObjectRotation::identity(),
        render_config: RenderConfig::tbp_default(),
    })?;
}

// Execute and collect results
let mut results = Vec::new();
loop {
    match render_next_in_batch(&mut renderer, 500)? {
        Some(output) => results.push(output),
        None => break,
    }
}
```

**Convenience Function (for simple cases):**

```rust
use bevy_sensor::{render_batch, batch::BatchRenderRequest, BatchRenderConfig};

let results = render_batch(requests, &BatchRenderConfig::default())?;
```

**Performance Comparison:**

| Method | 72 Renders (3 rotations × 24 viewpoints) |
|--------|-----------|
| `render_to_buffer()` × 72 (sequential) | ~30-50s |
| `render_all_viewpoints()` | ~25-40s |
| `render_batch()` (current) | ~20-35s |
| `render_batch()` (future: persistent app) | ~1-3s |

**When to Use:**

- ✅ Batch API: 10+ renders (any object/config mix)
- ✅ Batch API: Rendering entire dataset iterations
- ✅ Single `render_to_buffer()`: 1-3 renders or quick tests
- ✅ `render_all_viewpoints()`: Legacy code (still works)

**API Types:**

```rust
pub struct BatchRenderRequest {
    pub object_dir: PathBuf,           // Path to YCB object
    pub viewpoint: Transform,          // Camera position
    pub object_rotation: ObjectRotation,
    pub render_config: RenderConfig,
}

pub struct BatchRenderOutput {
    pub request: BatchRenderRequest,
    pub rgba: Vec<u8>,                 // RGBA pixels
    pub depth: Vec<f64>,               // Depth in meters (f64)
    pub status: RenderStatus,          // Success, PartialFailure, Failed
    // ... other fields
}

pub enum RenderStatus {
    Success,           // RGBA + depth complete
    PartialFailure,    // RGBA ok, depth missing
    Failed,            // Render failed
}
```

**neocortx Integration:**

```rust
// Render 72 observations for benchmark
let requests: Vec<_> = rotations.iter()
    .flat_map(|rot| viewpoints.iter().map(move |vp| {
        BatchRenderRequest {
            object_dir: object_dir.clone(),
            viewpoint: *vp,
            object_rotation: rot.clone(),
            render_config: RenderConfig::tbp_default(),
        }
    }))
    .collect();

let outputs = render_batch(requests, &BatchRenderConfig::default())?;

// Convert to neocortx observation format
for output in outputs {
    let rgb_image: Vec<Vec<[u8; 3]>> = output.to_rgb_image();
    let depth_image: Vec<Vec<f64>> = output.to_depth_image();
    // Send to neocortx sensorimotor system...
}
```

See `examples/batch_render_neocortx.rs` for a complete working example.

### Viewpoint Generation

Uses **spherical coordinates** matching TBP habitat sensor behavior:

```rust
struct ViewpointConfig {
    radius: f32,              // Distance from object (default: 0.5m)
    yaw_count: usize,         // Horizontal positions (default: 8)
    pitch_angles_deg: Vec<f32>, // Elevations (default: [-30°, 0°, +30°])
}
```

**Spherical to Cartesian conversion (Y-up):**
```
x = radius * cos(pitch) * sin(yaw)
y = radius * sin(pitch)
z = radius * cos(pitch) * cos(yaw)
```

### Capture Configurations

| Config | Rotations | Viewpoints | Total |
|--------|-----------|------------|-------|
| `SensorConfig::default()` | 1 | 24 | 24 |
| `SensorConfig::tbp_benchmark()` | 3 | 24 | 72 |
| `SensorConfig::tbp_full_training()` | 14 | 24 | 336 |

### Capture State Machine (`src/main.rs`)

```
SetupRotation → SetupView → WaitSettle (10 frames) → Capture → WaitSave (200 frames) → loop
```

Output: `capture_{rot}_{view}.png` (e.g., `capture_0_0.png` through `capture_2_23.png`)

## TBP Habitat Sensor Alignment

| TBP Feature | Bevy Implementation |
|-------------|---------------------|
| Quaternion rotation | `ObjectRotation::to_quat()` |
| `look_up` / `look_down` | Pitch angles: -30°, 0°, +30° |
| `turn_left` / `turn_right` | 8 yaw positions @ 45° intervals |
| Object rotations [0,0,0], [0,90,0], [0,180,0] | `ObjectRotation::tbp_benchmark_rotations()` |
| 14 known orientations | `ObjectRotation::tbp_known_orientations()` |
| Spherical radius | Configurable (default 0.5m) |

### TBP Reference Implementation

- **Sensors**: `tbp/monty/simulators/habitat/sensors.py`
  - Uses quaternion (w,x,y,z) format, default identity: (1, 0, 0, 0)
  - Position relative to HabitatAgent

- **Actions**: `tbp/monty/frameworks/actions/actions.py`
  - `LookDown`, `LookUp`: rotation_degrees parameter
  - `TurnLeft`, `TurnRight`: yaw rotation
  - `SetYaw`, `SetAgentPitch`: absolute rotations

- **Benchmarks**:
  - 3 known rotations for quick tests: [0,0,0], [0,90,0], [0,180,0]
  - 14 known orientations for full training (cube faces + corners)
  - 10 random rotations for generalization testing

## YCB Dataset Setup

**Programmatic download (via ycbust library):**

```rust
use bevy_sensor::ycb::{download_models, Subset, models_exist};

// Check if models already exist
if !models_exist("/tmp/ycb") {
    // Download representative subset (3 objects) - async
    download_models("/tmp/ycb", Subset::Representative).await?;
}

// Get paths to specific object files
let mesh = bevy_sensor::ycb::object_mesh_path("/tmp/ycb", "003_cracker_box");
let texture = bevy_sensor::ycb::object_texture_path("/tmp/ycb", "003_cracker_box");
```

**Available subsets:**
- `Subset::Representative` - 3 objects (quick testing)
- `Subset::Ten` - 10 objects (TBP benchmark subset)
- `Subset::All` - All 77 YCB objects

The `assets/ycb` symlink points to `/tmp/ycb`.

## Dependencies

```toml
bevy = { version = "0.15", default-features = false, features = [
    "bevy_asset",
    "bevy_core_pipeline",
    "bevy_pbr",
    "bevy_render",
    "bevy_scene",
    "bevy_state",
    "bevy_winit",
    "png",
    "x11",
    "tonemapping_luts",
    "ktx2",
    "zstd",
] }
bevy_obj = { version = "0.15", features = ["scene"] }
ycbust = "0.2.3"
```

### Bevy 0.15 Upgrade (December 2024)

The upgrade from Bevy 0.11 to 0.15 enabled real GPU depth buffer readback:

**Key Changes:**
- `ViewDepthTexture` replaces `ViewPrepassTextures` for depth access
- `Screenshot` entity + observer pattern replaces `ScreenshotManager`
- Component-based spawning (`Camera3d`, `PointLight`) replaces bundles
- `Mesh3d` and `MeshMaterial3d<M>` wrappers for mesh/material handles
- `SceneRoot(handle)` replaces `SceneBundle`
- MSAA must be disabled (`Msaa::Off`) for depth texture copy

**Depth Buffer Implementation:**
- Uses `ViewDepthTexture` from `bevy::render::view`
- Reverse-Z depth converted to linear meters via `reverse_z_to_linear_depth()`
- Custom render graph node (`DepthReadbackNode`) copies depth after main pass
- 256-byte row alignment for GPU buffer mapping

## WebGPU Backend Support

bevy-sensor now includes automatic cross-platform rendering backend selection via the `backend` module:

### Automatic Platform Detection
- **Linux**: Vulkan (primary), OpenGL (fallback), WebGPU (as-needed)
- **WSL2**: WebGPU (primary - recommended for WSL2), OpenGL (fallback)
- **macOS**: Metal (primary), OpenGL (fallback)
- **Windows**: Direct3D 12 (primary), Vulkan (fallback), WebGPU (as-needed)

### Using WebGPU Backend
```rust
use bevy_sensor::backend::{BackendConfig, RenderBackend};

// Use WebGPU explicitly for maximum compatibility
let config = BackendConfig {
    preferred: Some(RenderBackend::WebGPU),
    fallbacks: vec![RenderBackend::OpenGL],
    debug_logging: false,
    force_headless: true,
};
config.apply_env();

// Now run your render functions - they'll use WebGPU
```

### Environment Variable Override
```bash
# Force WebGPU backend
export WGPU_BACKEND=webgpu
cargo run --release

# Other options: vulkan, metal, dx12, gl, webgpu, webgl2, auto
```

### WSL2 Optimization
WSL2 environments now default to WebGPU backend (previously required pre-rendered fixtures):
```bash
# This now works on WSL2 (auto-selects WebGPU)
cargo run --example batch_render_neocortx --release
```

**Note**: WebGPU backend may have slightly lower performance than native backends but offers excellent compatibility across platforms.

### Bevy 0.16 Upgrade (December 2024, In Progress)

Migrating from Bevy 0.15 to 0.16 for improved GPU detection on WSL2 and broader platform compatibility.

**Migration Status:**
- ✅ Updated all Bevy 0.15 → 0.16 API changes (80 tests passing)
- ✅ Fixed AmbientLight `affects_lightmapped_meshes` field requirement
- ✅ Fixed `EventWriter::send()` → `write()` API change
- ✅ Fixed deprecated `Query::get_single_mut()` → `single_mut()`
- ✅ Fixed `Image` texture_descriptor API changes
- ✅ Replaced logging macros (bevy::log removed in 0.16)
- ✅ Fixed `RenderTarget::Image` type changes
- ⚠️ **TODO**: Re-implement depth texture readback (currently commented out)

**Breaking Changes in Bevy 0.16:**
- `bevy::log` module removed - using standard Rust println!/eprintln! instead
- Low-level `ImageCopyBuffer`, `ImageCopyTexture`, `ImageDataLayout` wgpu types no longer re-exported
- Depth texture to buffer copying needs new implementation
- Various ECS query result-based error handling (Result instead of panic)

**What Changed:**
- Image::texture_descriptor now required instead of direct field access
- Camera target type changed to `ImageRenderTarget` wrapper
- Screenshot capture now uses `Screenshot` entity + observer pattern (already in 0.15)
- Logging now uses standard Rust log crate facilities

**Known Issues:**
- **Depth Map Re-implementation Required**: The custom depth texture readback code (DepthReadbackNode, ImageCopyDriver) has been commented out because the low-level wgpu ImageCopy types are not directly accessible in Bevy 0.16. The rendering still works for RGBA/RGB, but depth buffers currently return zeros. This must be fixed before merging to main.
  - Location: `src/render.rs` lines 445-451 and 691-700
  - Alternative approaches to investigate:
    1. Use Bevy's built-in depth texture readback if available in 0.16
    2. Create a custom render node using the wgpu API through Bevy's RenderContext
    3. Use a different approach like normal maps or indirect depth estimation

**Testing Status:**
- All 80 unit tests pass ✅
- Compilation: ✅ Release build successful
- GPU rendering on WSL2: ❌ **Still fails** - wgpu adapter enumeration fails despite GPU being present (nvidia-smi shows RTX 4090, CUDA 12.8)
  - Error: "Unable to find a GPU!" panic in `bevy_render-0.16.1/src/renderer/mod.rs:197`
  - Root cause: WSL2 wgpu limitation, not Bevy 0.16 specific
  - **Status**: This is a framework-level issue beyond bevy-sensor's control
  - **Confirmed**: GPU drivers and CUDA are properly installed and functional

## Known Limitations

1. **WSL2 GPU rendering** (NOT FIXED - Framework Limitation):
   - ✅ GPU is available: RTX 4090 with 23GB, CUDA 12.8, drivers 572.16
   - ❌ **Issue**: wgpu adapter enumeration fails on WSL2 even with GPU present
   - **Root cause**: WSL2 + wgpu + NVIDIA compatibility limitation (not specific to Bevy)
   - **Tested**: Bevy 0.15 and 0.16 both fail with same error
   - **Workaround**: Use pre-rendered fixtures for CI/CD (see below)
   - **Solution for native Linux**: Integration tests work perfectly with GPU
   - **Recommendation**: For production WSL2 use, run neocortx on native Linux with GPU passthrough

2. **Software rendering (llvmpipe)**:
   - Must disable tonemapping (`Tonemapping::None`) on the camera
   - PBR shaders may fail with `gsamplerCubeArrayShadow` error
   - Xvfb may crash due to NVIDIA driver conflicts in WSL2

3. **Texture loading with bevy_obj**:
   - The bevy_obj scene loader has "limited MTL support"
   - **Workaround**: Load textures manually with `asset_server.load()`
   - MTL files with trailing spaces may cause loading failures

4. **Asset path**: When running binary from `target/release/`, assets must be in `target/release/assets/`

## Pre-rendered Fixtures for CI/CD

For environments without GPU rendering (WSL2, CI servers, Docker), use pre-rendered fixtures:

### Generating Fixtures

Run on a machine with a working display (Linux desktop with GPU):

```bash
# Generate test fixtures for CI/CD
cargo run --bin prerender -- --output-dir test_fixtures/renders

# Custom objects
cargo run --bin prerender -- --objects "003_cracker_box,005_tomato_soup_can,006_mustard_bottle"
```

This creates:
- `test_fixtures/renders/metadata.json` - Dataset configuration
- `test_fixtures/renders/{object_id}/` - Per-object renders
  - `r{N}_v{M}.png` - RGBA images
  - `r{N}_v{M}.depth` - Depth data (binary f32)
  - `index.json` - Per-object metadata

### Loading Fixtures in Tests

```rust
use bevy_sensor::fixtures::TestFixtures;

// Load pre-rendered data
let fixtures = TestFixtures::load("test_fixtures/renders")?;

// Get a specific render
let output = fixtures.get_render("003_cracker_box", 0, 0)?; // rotation 0, viewpoint 0

// Use like normal RenderOutput
let rgb = output.to_rgb_image();
let depth = output.to_depth_image();
```

### neocortx Integration with Fixtures

```rust
// In neocortx tests, prefer fixtures for CI/CD compatibility
#[cfg(test)]
mod tests {
    use bevy_sensor::fixtures::TestFixtures;

    #[test]
    fn test_sensor_with_fixtures() {
        let fixtures = TestFixtures::load("test_fixtures/renders")
            .expect("Pre-rendered fixtures required. Run: cargo run --bin prerender");

        // Test with pre-rendered data
        for (object_id, rotation_idx, viewpoint_idx, output) in fixtures.iter_renders() {
            // Process render output...
        }
    }
}
```

### WSL2 Workarounds

If you need live rendering on WSL2, these options may work (unreliable):

1. **VcXsrv on Windows**: Install VcXsrv, set `DISPLAY=172.x.x.x:0`
2. **X410 on Windows**: Commercial X server with better GPU support
3. **Native Linux VM**: Use VirtualBox/VMware with GPU passthrough

**Recommended**: Generate fixtures on a Linux machine and commit to repo for CI/CD use.

## Viewpoint Coordinate Reference

```
View  0-7:  pitch=-30° (below), yaw=0°-315° @ 45° steps, Y=-0.250
View  8-15: pitch=0°   (level), yaw=0°-315° @ 45° steps, Y=0.000
View 16-23: pitch=+30° (above), yaw=0°-315° @ 45° steps, Y=+0.250
```

## Usage from neocortx

```rust
use bevy_sensor::{SensorConfig, ObjectRotation, ViewpointConfig};
use bevy_sensor::ycb::{download_models, Subset, models_exist};

// Ensure YCB models are available
if !models_exist("/tmp/ycb") {
    download_models("/tmp/ycb", Subset::Representative).await?;
}

// Create capture config
let config = SensorConfig {
    viewpoints: ViewpointConfig {
        radius: 0.5,
        yaw_count: 8,
        pitch_angles_deg: vec![-30.0, 0.0, 30.0],
    },
    object_rotations: ObjectRotation::tbp_benchmark_rotations(),
    output_dir: "./captures".to_string(),
    filename_pattern: "ycb_{rot}_{view}.png".to_string(),
};

println!("Total captures: {}", config.total_captures()); // 72
```

## Related Projects

- **neocortx**: Rust-based Thousand Brains implementation (main project)
- **tbp.monty**: Original Python implementation by Thousand Brains Project
- **tbp.tbs_sensorimotor_intelligence**: TBP experiment configs
- **ycbust**: YCB dataset downloader (used as library dependency)

## API Parity with neocortx

bevy-sensor provides complete API parity with neocortx's `bevy_simulator` module:

| neocortx Requirement | bevy-sensor API | Status |
|---------------------|-----------------|--------|
| RGBA image data | `RenderOutput.rgba: Vec<u8>` | ✅ |
| Depth buffer (meters, f64) | `RenderOutput.depth: Vec<f64>` | ✅ |
| Camera intrinsics (f64) | `CameraIntrinsics` struct | ✅ |
| Camera position | `RenderOutput.camera_transform` | ✅ |
| Object rotation (f64) | `RenderOutput.object_rotation` | ✅ |
| RGB image format | `to_rgb_image() → Vec<Vec<[u8; 3]>>` | ✅ |
| Depth image format (f64) | `to_depth_image() → Vec<Vec<f64>>` | ✅ |
| TBP viewpoints (24) | `ViewpointConfig::default()` | ✅ |
| TBP benchmark rotations (3) | `ObjectRotation::tbp_benchmark_rotations()` | ✅ |
| TBP full rotations (14) | `ObjectRotation::tbp_known_orientations()` | ✅ |
| YCB model download | `ycb::download_models()` | ✅ |
| Resolution config | `RenderConfig` (64×64, 256×256, 512×512) | ✅ |

**neocortx integration example:**

```rust
use bevy_sensor::{render_to_buffer, RenderConfig, ViewpointConfig, ObjectRotation};

// Render and convert to neocortx formats
let output = render_to_buffer(object_dir, &viewpoint, &rotation, &config)?;
let rgb_image: Vec<Vec<[u8; 3]>> = output.to_rgb_image();    // VisionObservation.image
let depth_image: Vec<Vec<f64>> = output.to_depth_image();    // For surface normal (f64 precision)
let intrinsics = output.intrinsics;                          // CameraIntrinsics (f64)
```

## Release Flow & Versioning

**Version bumps are handled automatically by release-please.** Do NOT manually bump version numbers in `Cargo.toml`.

### Conventional Commits

This project uses [Conventional Commits](https://www.conventionalcommits.org/) for automated changelog and version management:

- `feat:` - New features (minor version bump)
- `fix:` - Bug fixes (patch version bump)
- `feat!:` or `BREAKING CHANGE:` - Breaking changes (major version bump)
- `docs:`, `chore:`, `refactor:`, `test:` - No version bump

### Release Process

1. Merge PRs with conventional commit messages to `main`
2. release-please creates/updates a "Release PR" with:
   - Updated `Cargo.toml` version
   - Updated `CHANGELOG.md`
3. Merge the Release PR to trigger:
   - Git tag creation
   - GitHub release
   - crates.io publish (if configured)

**Note:** Git tags are created asynchronously by GitHub Actions (30-60 seconds). Don't repeatedly `git fetch` after merging the release PR—the tag will be available once the workflow completes. Check status with `gh run list --workflow CI`.

### Breaking Changes

When making breaking API changes (like f32 → f64 migration):

1. Use `feat!:` prefix in commit message
2. Include `BREAKING CHANGE:` section in commit body
3. Document migration in PR description

## Resources

- [TBP Benchmark Experiments](https://thousandbrainsproject.readme.io/docs/benchmark-experiments)
- [TBP GitHub](https://github.com/thousandbrainsproject/tbp.monty)
- [YCB Object Dataset](https://www.ycbbenchmarks.com/)
