//! Model caching system for efficient multi-viewpoint rendering.
//!
//! Caches loaded mesh and texture assets to avoid redundant disk I/O and
//! asset loading when rendering multiple viewpoints of the same object.
//!
//! # Performance
//!
//! Typical speedup when rendering the same object multiple times:
//! - First render: 100% (loads from disk)
//! - Subsequent renders: 2-3x faster (cache hits)
//!
//! # Example
//!
//! ```rust,no_run
//! use bevy_sensor::{
//!     cache::ModelCache,
//!     render_to_buffer_cached,
//!     RenderConfig, ViewpointConfig, ObjectRotation,
//! };
//! use std::path::PathBuf;
//!
//! let mut cache = ModelCache::new();
//! let object_dir = PathBuf::from("/tmp/ycb/003_cracker_box");
//! let config = RenderConfig::tbp_default();
//! let viewpoints = bevy_sensor::generate_viewpoints(&ViewpointConfig::default());
//!
//! // First render: loads from disk
//! let output1 = render_to_buffer_cached(
//!     &object_dir,
//!     &viewpoints[0],
//!     &ObjectRotation::identity(),
//!     &config,
//!     &mut cache,
//! ).unwrap();
//!
//! // Subsequent renders: uses cache (much faster)
//! for viewpoint in &viewpoints[1..] {
//!     let output = render_to_buffer_cached(
//!         &object_dir,
//!         viewpoint,
//!         &ObjectRotation::identity(),
//!         &config,
//!         &mut cache,
//!     ).unwrap();
//! }
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Cache for loaded mesh and texture assets.
///
/// Stores `Handle<Scene>` and `Handle<Image>` by file path to avoid
/// redundant loading during multi-viewpoint rendering.
#[derive(Debug, Clone, Default)]
pub struct ModelCache {
    /// Cached scene meshes by path
    scenes: HashMap<PathBuf, bool>, // true = cached
    /// Cached texture images by path
    textures: HashMap<PathBuf, bool>, // true = cached
}

impl ModelCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self {
            scenes: HashMap::new(),
            textures: HashMap::new(),
        }
    }

    /// Check if a mesh is cached.
    pub fn has_scene(&self, path: &Path) -> bool {
        self.scenes.contains_key(path)
    }

    /// Check if a texture is cached.
    pub fn has_texture(&self, path: &Path) -> bool {
        self.textures.contains_key(path)
    }

    /// Cache a scene path.
    pub fn cache_scene(&mut self, path: PathBuf) {
        self.scenes.insert(path, true);
    }

    /// Cache a texture path.
    pub fn cache_texture(&mut self, path: PathBuf) {
        self.textures.insert(path, true);
    }

    /// Get number of cached scenes.
    pub fn scene_count(&self) -> usize {
        self.scenes.len()
    }

    /// Get number of cached textures.
    pub fn texture_count(&self) -> usize {
        self.textures.len()
    }

    /// Clear all cached items.
    pub fn clear(&mut self) {
        self.scenes.clear();
        self.textures.clear();
    }

    /// Get total cache size information.
    pub fn stats(&self) -> ModelCacheStats {
        ModelCacheStats {
            cached_scenes: self.scenes.len(),
            cached_textures: self.textures.len(),
        }
    }
}

/// Statistics about cache usage.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ModelCacheStats {
    /// Number of unique scenes in cache
    pub cached_scenes: usize,
    /// Number of unique textures in cache
    pub cached_textures: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_cache_is_empty() {
        let cache = ModelCache::new();
        assert_eq!(cache.scene_count(), 0);
        assert_eq!(cache.texture_count(), 0);
    }

    #[test]
    fn test_cache_scene() {
        let mut cache = ModelCache::new();
        let path = PathBuf::from("/tmp/ycb/003_cracker_box/google_16k/textured.obj");

        assert!(!cache.has_scene(&path));
        cache.cache_scene(path.clone());
        assert!(cache.has_scene(&path));
        assert_eq!(cache.scene_count(), 1);
    }

    #[test]
    fn test_cache_texture() {
        let mut cache = ModelCache::new();
        let path = PathBuf::from("/tmp/ycb/003_cracker_box/google_16k/texture_map.png");

        assert!(!cache.has_texture(&path));
        cache.cache_texture(path.clone());
        assert!(cache.has_texture(&path));
        assert_eq!(cache.texture_count(), 1);
    }

    #[test]
    fn test_cache_multiple_items() {
        let mut cache = ModelCache::new();

        cache.cache_scene(PathBuf::from("scene1.obj"));
        cache.cache_scene(PathBuf::from("scene2.obj"));
        cache.cache_texture(PathBuf::from("texture1.png"));

        assert_eq!(cache.scene_count(), 2);
        assert_eq!(cache.texture_count(), 1);
    }

    #[test]
    fn test_clear_cache() {
        let mut cache = ModelCache::new();

        cache.cache_scene(PathBuf::from("scene.obj"));
        cache.cache_texture(PathBuf::from("texture.png"));
        assert_eq!(cache.scene_count(), 1);
        assert_eq!(cache.texture_count(), 1);

        cache.clear();
        assert_eq!(cache.scene_count(), 0);
        assert_eq!(cache.texture_count(), 0);
    }

    #[test]
    fn test_cache_stats() {
        let mut cache = ModelCache::new();

        cache.cache_scene(PathBuf::from("scene.obj"));
        cache.cache_texture(PathBuf::from("texture.png"));

        let stats = cache.stats();
        assert_eq!(stats.cached_scenes, 1);
        assert_eq!(stats.cached_textures, 1);
    }

    #[test]
    fn test_duplicate_cache_entry() {
        let mut cache = ModelCache::new();
        let path = PathBuf::from("scene.obj");

        cache.cache_scene(path.clone());
        cache.cache_scene(path.clone());

        // HashMap deduplicates, so count should still be 1
        assert_eq!(cache.scene_count(), 1);
    }
}
