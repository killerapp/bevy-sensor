# bevy-sensor justfile - Developer commands for YCB object rendering
#
# Usage: just <command> [args...]
# Run `just` or `just help` to see all available commands

# Default command - show help
default:
    @just --list

# ============================================================================
# Build Commands
# ============================================================================

# Build the library and all binaries (debug)
build:
    cargo build

# Build in release mode
build-release:
    cargo build --release

# Build only the prerender binary
build-prerender:
    cargo build --bin prerender

# Check code without building
check:
    cargo check

# Format code
fmt:
    cargo fmt

# Run clippy linter
lint:
    cargo clippy -- -D warnings

# ============================================================================
# Test Commands
# ============================================================================

# Run all tests
test:
    WGPU_BACKEND=vulkan cargo test -- --test-threads=1

# Run tests with output
test-verbose:
    WGPU_BACKEND=vulkan cargo test -- --test-threads=1 --nocapture

# Run only library tests
test-lib:
    WGPU_BACKEND=vulkan cargo test --lib -- --test-threads=1

# ============================================================================
# Render Commands - Headless GPU rendering on WSL2/Linux
# ============================================================================

# Render a single viewpoint (for testing)
# Usage: just render-single <object> [rotation] [viewpoint] [output_dir]
render-single object rotation="0" viewpoint="0" output="test_fixtures/renders":
    WGPU_BACKEND=vulkan cargo run --bin prerender -- \
        --single-render \
        --object {{object}} \
        --rotation {{rotation}} \
        --viewpoint {{viewpoint}} \
        --output {{output}}

# Batch render default CI test objects (003_cracker_box, 005_tomato_soup_can)
render-ci:
    cargo run --bin prerender

# Batch render specific objects
# Usage: just render-batch "003_cracker_box,005_tomato_soup_can"
render-batch objects:
    cargo run --bin prerender -- --objects {{objects}}

# Batch render all TBP benchmark objects (10 objects)
render-tbp-benchmark:
    cargo run --bin prerender -- --objects "002_master_chef_can,003_cracker_box,004_sugar_box,005_tomato_soup_can,006_mustard_bottle,007_tuna_fish_can,008_pudding_box,009_gelatin_box,010_potted_meat_can,011_banana"

# Render to custom output directory
# Usage: just render-to <dir> [objects]
render-to dir objects="003_cracker_box":
    cargo run --bin prerender -- --output-dir {{dir}} --objects {{objects}}

# Clean render output directory
clean-renders:
    rm -rf test_fixtures/renders

# ============================================================================
# YCB Dataset Commands
# ============================================================================

# Download YCB models (representative subset - 3 objects)
ycb-download-representative:
    cargo run --example test_render -- --download-only

# Check if YCB models are present
ycb-check:
    @if [ -d "/tmp/ycb/003_cracker_box" ]; then \
        echo "YCB models found at /tmp/ycb"; \
        ls -1 /tmp/ycb | head -10; \
        echo "..."; \
    else \
        echo "YCB models not found. Run: just ycb-download-representative"; \
    fi

# List available YCB objects
ycb-list:
    @ls -1 /tmp/ycb 2>/dev/null || echo "YCB models not downloaded. Run: just ycb-download-representative"

# ============================================================================
# Development Commands
# ============================================================================

# Run the main example (requires display or Xvfb)
run-example:
    cargo run --example test_render

# Run with software rendering (llvmpipe) - may have shader issues
run-software:
    LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe \
        xvfb-run -a -s "-screen 0 1280x1024x24" \
        cargo run --release

# Watch for changes and rebuild
watch:
    cargo watch -x check

# Generate documentation
doc:
    cargo doc --open

# Clean build artifacts
clean:
    cargo clean

# Full clean including renders
clean-all: clean clean-renders

# ============================================================================
# CI/CD Commands
# ============================================================================

# Run full CI check (format, lint, test)
ci: fmt lint test

# Pre-commit check
pre-commit: fmt lint check test

# ============================================================================
# Debug Commands
# ============================================================================

# Run prerender with debug output
render-debug object="003_cracker_box":
    WGPU_BACKEND=vulkan RUST_BACKTRACE=1 cargo run --bin prerender -- \
        --single-render \
        --object {{object}} \
        --rotation 0 \
        --viewpoint 0 \
        --output test_fixtures/renders

# Check GPU/Vulkan availability
gpu-check:
    @echo "=== Vulkan Info ==="
    @vulkaninfo --summary 2>/dev/null || echo "vulkaninfo not available"
    @echo ""
    @echo "=== GPU Devices ==="
    @lspci | grep -i vga || echo "No VGA devices found"
    @echo ""
    @echo "=== WGPU Backend ==="
    @echo "Set WGPU_BACKEND=vulkan for headless rendering"

# ============================================================================
# Help
# ============================================================================

# Show detailed help
help:
    @echo "bevy-sensor - YCB Object Multi-View Renderer"
    @echo ""
    @echo "This library renders 3D YCB objects from multiple viewpoints for"
    @echo "sensor simulation in the Thousand Brains Project (TBP)."
    @echo ""
    @echo "QUICK START:"
    @echo "  just ycb-check          # Check if YCB models are downloaded"
    @echo "  just render-ci          # Render default test objects (2 objects, 72 views each)"
    @echo "  just test               # Run all tests"
    @echo ""
    @echo "RENDERING (requires GPU with Vulkan support):"
    @echo "  just render-single 003_cracker_box     # Single viewpoint render"
    @echo "  just render-batch \"obj1,obj2\"          # Batch render specific objects"
    @echo "  just render-tbp-benchmark              # Render all 10 TBP benchmark objects"
    @echo ""
    @echo "CONFIGURATION:"
    @echo "  - Resolution: 64x64 (TBP default)"
    @echo "  - Viewpoints: 24 (8 yaw × 3 pitch angles)"
    @echo "  - Rotations: 3 ([0,0,0], [0,90,0], [0,180,0])"
    @echo "  - Output: test_fixtures/renders/"
    @echo ""
    @echo "ENVIRONMENT VARIABLES:"
    @echo "  WGPU_BACKEND=vulkan    Required for headless rendering on WSL2"
    @echo ""
    @echo "Run 'just --list' to see all available commands."
