# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
