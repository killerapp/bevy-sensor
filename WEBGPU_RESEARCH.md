# WebGPU Backend Support Research

## Current State (Bevy 0.15)

### Architecture
- Bevy 0.15 uses wgpu for rendering (indirect through bevy_render)
- wgpu automatically selects backend based on platform:
  - Windows: Direct3D 12 (primary), Vulkan (fallback)
  - Linux: Vulkan (primary), OpenGL (fallback)
  - macOS: Metal (primary)
  - Web: WebGL2, WebGPU

### Environment Variables for wgpu
- `WGPU_BACKEND`: Force specific backend (vulkan, metal, dx12, gl, webgpu, auto)
- `WGPU_LOG`: Debug logging (trace, debug, info, warn, error)
- `WGPU_FORCE_OFFSCREEN`: Force offscreen rendering mode

### Key Issue: WSL2 GPU Rendering
- WSL2 does NOT support Vulkan window surfaces (GPU windowing)
- `render_to_buffer()` fails with "Invalid surface" error on WSL2
- Environment rendering (LIBGL_ALWAYS_SOFTWARE) has severe performance issues
- **Solution**: Use WebGPU backend which has better fallback support

## Implementation Strategy

### Phase 1: Detection & Configuration
1. Add platform detection utilities (already exist in render.rs as dead code)
2. Create BackendConfig struct with selection strategy
3. Implement backend preference ordering:
   - Native (Vulkan on Linux, Metal on macOS, D3D12 on Windows)
   - WebGPU (fallback for headless/restricted environments)
   - Software (last resort)

### Phase 2: Bevy Integration
1. Add `webgpu` feature flag to Cargo.toml
2. Create backend.rs module that:
   - Detects platform and environment
   - Selects appropriate wgpu backend
   - Configures Bevy to use selected backend
3. Integrate backend selection into render.rs

### Phase 3: Headless Rendering
1. Test WebGPU in headless mode on WSL2
2. Benchmark WebGPU vs Vulkan performance
3. Implement fallback when native backend fails

## References
- wgpu backends: https://github.com/gfx-rs/wgpu/blob/master/wgpu/src/backend.rs
- Bevy rendering: https://github.com/bevyengine/bevy/tree/main/crates/bevy_render
- WSL2 GPU limitations: https://github.com/microsoft/wsl/issues/4822
