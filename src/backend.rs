//! WebGPU and cross-platform backend support for rendering.
//!
//! This module provides utilities for selecting and configuring rendering backends
//! across different platforms. It handles:
//!
//! - Platform detection (Linux, macOS, Windows, WSL2)
//! - Backend preference ordering (native > WebGPU > software)
//! - Environment configuration (WGPU_BACKEND, etc.)
//! - Fallback strategies for restricted environments

use std::env;

/// Detected platform and rendering environment
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    /// Linux with native GPU support
    LinuxNative,
    /// WSL2 (Windows Subsystem for Linux 2)
    WSL2,
    /// macOS
    MacOS,
    /// Windows
    Windows,
    /// Web/WASM target
    Web,
    /// Unknown platform
    Unknown,
}

/// Available rendering backends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderBackend {
    /// Vulkan (Linux, Windows, Android)
    Vulkan,
    /// Metal (macOS, iOS)
    Metal,
    /// Direct3D 12 (Windows)
    DirectX12,
    /// OpenGL (Linux, Windows, macOS)
    OpenGL,
    /// WebGPU (cross-platform, especially good for WSL2 and headless)
    WebGPU,
    /// WebGL2 (Web)
    WebGL2,
    /// Software/CPU rendering (fallback)
    Software,
    /// Automatic selection (wgpu default)
    Auto,
}

impl RenderBackend {
    /// Get the wgpu environment variable value for this backend
    pub fn as_wgpu_env(&self) -> &'static str {
        match self {
            RenderBackend::Vulkan => "vulkan",
            RenderBackend::Metal => "metal",
            RenderBackend::DirectX12 => "dx12",
            RenderBackend::OpenGL => "gl",
            RenderBackend::WebGPU => "webgpu",
            RenderBackend::WebGL2 => "webgl2",
            RenderBackend::Software => "software",
            RenderBackend::Auto => "auto",
        }
    }

    /// Get a human-readable name for this backend
    pub fn name(&self) -> &'static str {
        match self {
            RenderBackend::Vulkan => "Vulkan",
            RenderBackend::Metal => "Metal",
            RenderBackend::DirectX12 => "Direct3D 12",
            RenderBackend::OpenGL => "OpenGL",
            RenderBackend::WebGPU => "WebGPU",
            RenderBackend::WebGL2 => "WebGL2",
            RenderBackend::Software => "Software",
            RenderBackend::Auto => "Auto",
        }
    }
}

/// Configuration for backend selection
#[derive(Debug, Clone)]
pub struct BackendConfig {
    /// Preferred backend (if available)
    pub preferred: Option<RenderBackend>,
    /// Fallback backends in order of preference
    pub fallbacks: Vec<RenderBackend>,
    /// Whether to enable debug logging
    pub debug_logging: bool,
    /// Force headless mode (don't create windows)
    pub force_headless: bool,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl BackendConfig {
    /// Create a new configuration with sensible defaults for the current platform
    pub fn new() -> Self {
        let platform = detect_platform();

        // Select backends based on platform
        let (preferred, fallbacks) = match platform {
            Platform::LinuxNative => (
                Some(RenderBackend::Vulkan),
                vec![RenderBackend::OpenGL, RenderBackend::WebGPU],
            ),
            Platform::WSL2 => (
                Some(RenderBackend::WebGPU),
                vec![RenderBackend::OpenGL, RenderBackend::Software],
            ),
            Platform::MacOS => (
                Some(RenderBackend::Metal),
                vec![RenderBackend::OpenGL, RenderBackend::WebGPU],
            ),
            Platform::Windows => (
                Some(RenderBackend::DirectX12),
                vec![
                    RenderBackend::Vulkan,
                    RenderBackend::OpenGL,
                    RenderBackend::WebGPU,
                ],
            ),
            Platform::Web => (Some(RenderBackend::WebGPU), vec![RenderBackend::WebGL2]),
            Platform::Unknown => (
                Some(RenderBackend::Auto),
                vec![RenderBackend::WebGPU, RenderBackend::OpenGL],
            ),
        };

        Self {
            preferred,
            fallbacks,
            debug_logging: env::var("WGPU_LOG").is_ok(),
            force_headless: true,
        }
    }

    /// Create a configuration optimized for headless rendering
    pub fn headless() -> Self {
        let mut config = Self::new();
        config.force_headless = true;
        config
    }

    /// Create a configuration optimized for WSL2
    pub fn wsl2() -> Self {
        Self {
            preferred: Some(RenderBackend::WebGPU),
            fallbacks: vec![RenderBackend::OpenGL, RenderBackend::Software],
            debug_logging: true,
            force_headless: true,
        }
    }

    /// Get the selected backend, considering environment overrides
    pub fn selected_backend(&self) -> RenderBackend {
        // Check for environment override
        if let Ok(backend_str) = env::var("WGPU_BACKEND") {
            if let Some(backend) = parse_backend(&backend_str) {
                return backend;
            }
        }

        // Use preferred backend
        self.preferred.unwrap_or(RenderBackend::Auto)
    }

    /// Get a list of backends to try in order (preferred + fallbacks)
    pub fn backends_to_try(&self) -> Vec<RenderBackend> {
        let mut backends = vec![self.selected_backend()];
        backends.extend(self.fallbacks.iter().copied());
        backends
    }

    /// Apply this configuration to the environment
    pub fn apply_env(&self) {
        let backend = self.selected_backend();
        env::set_var("WGPU_BACKEND", backend.as_wgpu_env());

        if self.debug_logging {
            env::set_var("WGPU_LOG", "warn");
        }

        if self.force_headless {
            env::set_var("WGPU_FORCE_OFFSCREEN", "1");
        }
    }
}

/// Detect the current platform
pub fn detect_platform() -> Platform {
    #[cfg(target_os = "linux")]
    {
        if is_wsl2() {
            Platform::WSL2
        } else {
            Platform::LinuxNative
        }
    }

    #[cfg(target_os = "macos")]
    {
        Platform::MacOS
    }

    #[cfg(target_os = "windows")]
    {
        Platform::Windows
    }

    #[cfg(target_arch = "wasm32")]
    {
        Platform::Web
    }

    #[cfg(not(any(
        target_os = "linux",
        target_os = "macos",
        target_os = "windows",
        target_arch = "wasm32"
    )))]
    {
        Platform::Unknown
    }
}

/// Check if running on WSL2 (Windows Subsystem for Linux 2)
pub fn is_wsl2() -> bool {
    if let Ok(version) = std::fs::read_to_string("/proc/version") {
        let version_lower = version.to_lowercase();
        return version_lower.contains("microsoft") || version_lower.contains("wsl");
    }
    false
}

/// Check if a display is available for windowed rendering
pub fn has_display() -> bool {
    env::var("DISPLAY").is_ok() || env::var("WAYLAND_DISPLAY").is_ok()
}

/// Parse a backend string from environment variable or user input
fn parse_backend(s: &str) -> Option<RenderBackend> {
    match s.to_lowercase().as_str() {
        "vulkan" => Some(RenderBackend::Vulkan),
        "metal" => Some(RenderBackend::Metal),
        "dx12" | "d3d12" | "directx12" => Some(RenderBackend::DirectX12),
        "gl" | "opengl" => Some(RenderBackend::OpenGL),
        "webgpu" | "web" => Some(RenderBackend::WebGPU),
        "webgl2" | "webgl" => Some(RenderBackend::WebGL2),
        "software" | "cpu" => Some(RenderBackend::Software),
        "auto" => Some(RenderBackend::Auto),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_env_strings() {
        assert_eq!(RenderBackend::Vulkan.as_wgpu_env(), "vulkan");
        assert_eq!(RenderBackend::Metal.as_wgpu_env(), "metal");
        assert_eq!(RenderBackend::WebGPU.as_wgpu_env(), "webgpu");
        assert_eq!(RenderBackend::OpenGL.as_wgpu_env(), "gl");
    }

    #[test]
    fn test_backend_names() {
        assert_eq!(RenderBackend::Vulkan.name(), "Vulkan");
        assert_eq!(RenderBackend::WebGPU.name(), "WebGPU");
        assert_eq!(RenderBackend::DirectX12.name(), "Direct3D 12");
    }

    #[test]
    fn test_parse_backend() {
        assert_eq!(parse_backend("vulkan"), Some(RenderBackend::Vulkan));
        assert_eq!(parse_backend("VULKAN"), Some(RenderBackend::Vulkan));
        assert_eq!(parse_backend("webgpu"), Some(RenderBackend::WebGPU));
        assert_eq!(parse_backend("invalid"), None);
    }

    #[test]
    fn test_backend_config_default() {
        let config = BackendConfig::default();
        assert!(config.selected_backend() != RenderBackend::Auto || config.preferred.is_none());
        assert!(!config.fallbacks.is_empty());
    }

    #[test]
    fn test_backend_config_headless() {
        let config = BackendConfig::headless();
        assert!(config.force_headless);
    }

    #[test]
    fn test_backends_to_try() {
        let config = BackendConfig::wsl2();
        let backends = config.backends_to_try();
        assert!(!backends.is_empty());
        assert_eq!(backends[0], RenderBackend::WebGPU);
    }

    #[test]
    fn test_is_wsl2_detection() {
        // This test will only be meaningful on WSL2 or Linux
        let is_wsl = is_wsl2();
        // We just verify it doesn't panic
        let _ = is_wsl;
    }
}
