# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.2](https://github.com/killerapp/bevy-sensor/compare/v0.5.1...v0.5.2) - 2026-04-25

### Added

- *(render)* PersistentRenderer for per-step feedback loops ([#66](https://github.com/killerapp/bevy-sensor/pull/66))

## [0.5.1](https://github.com/killerapp/bevy-sensor/compare/v0.5.0...v0.5.1) - 2026-04-23

### Added

- feat!(ycb): bump ycbust 0.3 -> 0.4.0 + labeled contract tests ([#62](https://github.com/killerapp/bevy-sensor/pull/62))

## [0.5.0](https://github.com/killerapp/bevy-sensor/compare/v0.4.10...v0.5.0) - 2026-04-21

### Added

- *(render)* RenderSession — persistent App for 8.85× parity-gate speedup ([#59](https://github.com/killerapp/bevy-sensor/pull/59))

### Fixed

- *(render)* drop legacy 60-frame scene-warmup gate in apply_materials ([#56](https://github.com/killerapp/bevy-sensor/pull/56))

### Other

- *(render)* reduce batch warmup 3→1 + add BEVY_SENSOR_RENDER_TRACE diagnostics ([#57](https://github.com/killerapp/bevy-sensor/pull/57))

## [0.4.10](https://github.com/killerapp/bevy-sensor/compare/v0.4.9...v0.4.10) - 2026-04-21

### Fixed

- *(render)* apply configured FOV to camera projection and match TBP intrinsics
- *(render)* increase hardcoded render timeout from 60s to 180s
- *(render)* canonicalize object paths before passing to Bevy asset server

### Other

- *(render)* extract RENDER_TIMEOUT_SECS constant from six duplicated sites
- *(render)* replace magic-number assertions with formula invariants

## [0.4.9](https://github.com/killerapp/bevy-sensor/compare/v0.4.8...v0.4.9) - 2026-04-04

### Fixed

- *(render)* guard backend env init with OnceLock to prevent repeated wgpu env writes

### Other

- Add headless throughput smoke test
- Deduplicate single-render headless app setup
- release v0.4.8

## [0.4.8](https://github.com/killerapp/bevy-sensor/compare/v0.4.7...v0.4.8) - 2026-03-28

### Added

- *(render)* batch homogeneous viewpoint renders

### Fixed

- *(ci)* satisfy clippy on batch continuation system
- *(render)* stabilize batch headless validation

### Other

- *(release)* bump version to 0.4.8

## [0.4.7](https://github.com/killerapp/bevy-sensor/compare/v0.4.6...v0.4.7) - 2026-03-26

### Fixed

- *(ycb)* guard downstream release api
- keep deprecated ycb compatibility without clippy regressions
- *(ycb)* align integration with ycbust 0.3

### Other

- codify neocortx downstream posture

### Added

- add public YCB API regression coverage for `TBP_STANDARD_OBJECTS` and `download_objects`

### Changed

- document the YCB helper surface that downstream crates such as NeoCortx bind to between releases

## [0.4.6](https://github.com/killerapp/bevy-sensor/compare/v0.4.5...v0.4.6) - 2025-12-20

### Added

- auto-download YCB models in single render mode
- automated YCB management and alpha release preparation

### Fixed

- cargo fmt

### Other

- Merge main into prepare-alpha-release, resolve CHANGELOG conflict

### Added
- Automated YCB model downloading in `prerender` single-render mode when models are missing.
- Canonicalize paths for reliable Bevy asset loading.

### Changed
- Updated `README.md` and `GEMINI.md` to document `--data-dir` option.

## [0.4.5](https://github.com/killerapp/bevy-sensor/compare/v0.4.4...v0.4.5) - 2025-12-20

### Fixed

- clippy and fmt issues for CI
- Windows DirectX12 GPU rendering and graceful Bevy app exit

## [0.4.3](https://github.com/killerapp/bevy-sensor/compare/v0.4.2...v0.4.3) - 2025-12-18

### Fixed

- add initialize() function to ensure GPU backend config before WGPU caching ([#27](https://github.com/killerapp/bevy-sensor/pull/27))

## [0.4.2](https://github.com/killerapp/bevy-sensor/compare/v0.4.1...v0.4.2) - 2025-12-18

### Added

- add model caching system for efficient multi-viewpoint rendering ([#25](https://github.com/killerapp/bevy-sensor/pull/25))
- Add WebGPU backend support for cross-platform rendering

### Other

- Merge pull request #22 from killerapp/release-plz-2025-12-18T07-56-05Z
- Update WSL2 GPU limitation - now has CUDA access but Bevy 0.15 can't detect it
- Add render output saving to integration tests and justfile commands
- Add hardware rendering integration tests
- Add WebGPU backend usage example

## [0.4.1](https://github.com/killerapp/bevy-sensor/compare/v0.4.0...v0.4.1) - 2025-12-18

### Other

- Add note about asynchronous git tag creation in release workflow

## [0.4.0](https://github.com/killerapp/bevy-sensor/compare/v0.3.1...v0.4.0) - 2025-12-18

### Other

- Improve error handling with custom RenderError variants

## [0.3.1](https://github.com/killerapp/bevy-sensor/compare/v0.3.0...v0.3.1) - 2025-12-18

### Added

- Add batch rendering API for efficient multi-viewpoint rendering

### Fixed

- Resolve all clippy warnings
- Format code to pass CI checks

## [0.3.0](https://github.com/killerapp/bevy-sensor/compare/v0.2.2...v0.3.0) - 2025-12-17

### Added

- [**breaking**] migrate public API from f32 to f64 for TBP numerical precision ([#11](https://github.com/killerapp/bevy-sensor/pull/11))

## [0.2.2](https://github.com/killerapp/bevy-sensor/compare/v0.2.1...v0.2.2) - 2025-12-15

### Other

- update README for Bevy 0.15 and add ycbust link

## [0.2.1](https://github.com/killerapp/bevy-sensor/compare/v0.2.0...v0.2.1) - 2025-12-14

### Other

- bump ycbust dependency to 0.2.4

## [0.2.0](https://github.com/killerapp/bevy-sensor/compare/v0.1.1...v0.2.0) - 2025-12-14

### Added

- [**breaking**] upgrade to Bevy 0.15 with real GPU depth readback ([#6](https://github.com/killerapp/bevy-sensor/pull/6))

## [0.1.1](https://github.com/killerapp/bevy-sensor/compare/v0.1.0...v0.1.1) - 2025-12-13

### Added

- add LightingConfig and update documentation
- add headless rendering API for neocortx integration

### Other

- apply cargo fmt

## [0.1.0] - 2025-12-13

### Added

- Initial release of bevy-sensor library
- Multi-view image capture of 3D OBJ models
- YCB dataset integration via ycbust dependency
- Configurable viewpoint generation (spherical coordinates)
- TBP-compatible object rotations (benchmark and full training sets)
- Capture state machine for automated image generation
- Support for software rendering (llvmpipe) with proper tonemapping workaround
