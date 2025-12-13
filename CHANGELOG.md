# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
