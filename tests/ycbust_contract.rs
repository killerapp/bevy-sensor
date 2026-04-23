//! ycbust surface contract tests.
//!
//! These tests pin the subset of the `ycbust` public API that `bevy-sensor`
//! depends on. They are **offline** (no network), and failures after bumping
//! the `ycbust` version indicate either:
//!
//!   (a) a compatible API change we should propagate here, or
//!   (b) an upstream regression to coordinate with the ycbust team
//!       (<https://github.com/Agentic-Insights/ycbust>).
//!
//! Conventions:
//! - All tests prefixed `ycbust_contract_` so `cargo test ycbust_contract`
//!   runs the whole suite.
//! - Pin format / layout / enum shape. Do NOT pin implementation behavior
//!   (e.g. download ordering) — that's upstream's business.
//! - Live network coverage lives in `ycbust_s3_integration.rs` behind the
//!   `ycbust-s3` feature, `#[ignore]`d by default.

use std::path::Path;
use ycbust::{
    get_tgz_url, object_mesh_path, object_texture_path, validate_objects, DownloadOptions,
    ObjectValidation, Subset, YcbError, GOOGLE_16K_MESH_RELATIVE, GOOGLE_16K_TEXTURE_RELATIVE,
    REPRESENTATIVE_OBJECTS, TBP_SIMILAR_OBJECTS, TBP_STANDARD_OBJECTS,
};

// -----------------------------------------------------------------------------
// Subset enum
// -----------------------------------------------------------------------------

#[test]
fn ycbust_contract_subset_variants_present() {
    // bevy-sensor routes user config through these variants; removing any of
    // them is a breaking change for downstream callers of `ycb::download_models`.
    let _ = Subset::Representative;
    let _ = Subset::TbpStandard;
    let _ = Subset::TbpSimilar;
    let _ = Subset::All;
    // `Subset::Ten` is deprecated upstream but still exists; bevy-sensor no
    // longer re-exports it and no longer depends on the variant.
}

#[test]
fn ycbust_contract_subset_default_is_representative() {
    assert_eq!(Subset::default(), Subset::Representative);
}

// -----------------------------------------------------------------------------
// Object constants — content is part of the contract because bevy-sensor's
// `models_exist()` and prerender fixtures hardcode specific object ids.
// -----------------------------------------------------------------------------

#[test]
fn ycbust_contract_representative_objects_content_stable() {
    assert_eq!(
        REPRESENTATIVE_OBJECTS,
        &["003_cracker_box", "004_sugar_box", "005_tomato_soup_can"],
    );
}

#[test]
fn ycbust_contract_tbp_standard_objects_shape_stable() {
    assert_eq!(TBP_STANDARD_OBJECTS.len(), 10);
    // Spot-check TBP canonical entries.
    assert!(TBP_STANDARD_OBJECTS.contains(&"025_mug"));
    assert!(TBP_STANDARD_OBJECTS.contains(&"011_banana"));
    assert!(TBP_STANDARD_OBJECTS.contains(&"058_golf_ball"));
}

#[test]
fn ycbust_contract_tbp_similar_objects_shape_stable() {
    assert_eq!(TBP_SIMILAR_OBJECTS.len(), 10);
    assert!(TBP_SIMILAR_OBJECTS.contains(&"003_cracker_box"));
    assert!(TBP_SIMILAR_OBJECTS.contains(&"051_large_clamp"));
}

// -----------------------------------------------------------------------------
// URL / path helpers — bevy-sensor's `render.rs` and `download_objects`
// construct paths from these. Breaking the layout breaks asset loading.
// -----------------------------------------------------------------------------

#[test]
fn ycbust_contract_get_tgz_url_google_16k_format() {
    // bevy-sensor's `ycb::download_objects` passes the result of this call
    // straight to `download_file`. Host + `google/{id}_google_16k.tgz` is
    // the shape we rely on.
    let url = get_tgz_url("003_cracker_box", "google_16k");
    assert_eq!(
        url,
        "https://ycb-benchmarks.s3.amazonaws.com/data/google/003_cracker_box_google_16k.tgz",
    );
}

#[test]
fn ycbust_contract_object_mesh_path_layout() {
    // `render.rs` loads `{object_dir}/google_16k/textured.obj` directly. This
    // helper MUST produce the same relative layout so `models_exist` and the
    // renderer agree.
    let root = Path::new("/tmp/ycb");
    assert_eq!(
        object_mesh_path(root, "003_cracker_box"),
        root.join("003_cracker_box")
            .join("google_16k")
            .join("textured.obj"),
    );
}

#[test]
fn ycbust_contract_object_texture_path_layout() {
    let root = Path::new("/tmp/ycb");
    assert_eq!(
        object_texture_path(root, "003_cracker_box"),
        root.join("003_cracker_box")
            .join("google_16k")
            .join("texture_map.png"),
    );
}

// -----------------------------------------------------------------------------
// DownloadOptions defaults — bevy-sensor's `download_models` assumes these
// defaults (full=false so only google_16k is fetched; delete_archives=true so
// we don't leak .tgz files into the YCB tree).
// -----------------------------------------------------------------------------

#[test]
fn ycbust_contract_download_options_defaults() {
    let opts = DownloadOptions::default();
    assert!(!opts.overwrite, "overwrite default must be false");
    assert!(
        !opts.full,
        "full default must be false — bevy-sensor only consumes google_16k"
    );
    assert!(opts.show_progress, "show_progress default must be true");
    assert!(opts.delete_archives, "delete_archives default must be true");
    // v0.4.0 fields — behavior-preserving defaults.
    assert_eq!(
        opts.concurrency, 1,
        "concurrency default must be 1 (serial) for compatibility"
    );
    assert!(
        opts.verify_integrity,
        "verify_integrity default must be true — correctness over speed"
    );
}

#[test]
fn ycbust_contract_download_options_is_non_exhaustive() {
    // `#[non_exhaustive]` on DownloadOptions means struct-literal construction
    // from outside the crate fails to compile. The compile-time signal is
    // what protects us — this runtime test just pins the construction path
    // we rely on (Default + field override).
    let mut opts = DownloadOptions::default();
    opts.concurrency = 4;
    opts.verify_integrity = false;
    assert_eq!(opts.concurrency, 4);
    assert!(!opts.verify_integrity);
}

// -----------------------------------------------------------------------------
// validate_objects — bevy-sensor plans to delegate `models_exist` to this
// helper. Pin the shape so a field rename is caught here first.
// -----------------------------------------------------------------------------

#[test]
fn ycbust_contract_validate_objects_on_empty_dir() {
    let dir = tempfile::tempdir().expect("tempdir");
    let results = validate_objects(dir.path(), REPRESENTATIVE_OBJECTS);
    assert_eq!(results.len(), REPRESENTATIVE_OBJECTS.len());
    for r in &results {
        assert!(!r.mesh_present);
        assert!(!r.texture_present);
        assert!(!r.is_complete());
    }
}

#[test]
fn ycbust_contract_validate_objects_detects_complete_object() {
    let dir = tempfile::tempdir().expect("tempdir");
    // Materialize both mesh and texture for one representative object.
    let mesh = object_mesh_path(dir.path(), "003_cracker_box");
    let tex = object_texture_path(dir.path(), "003_cracker_box");
    std::fs::create_dir_all(mesh.parent().unwrap()).unwrap();
    std::fs::write(&mesh, b"").unwrap();
    std::fs::write(&tex, b"").unwrap();

    let results = validate_objects(dir.path(), &["003_cracker_box"]);
    assert_eq!(results.len(), 1);
    assert!(results[0].is_complete());
    assert_eq!(results[0].name, "003_cracker_box");
}

#[test]
fn ycbust_contract_object_validation_public_fields() {
    // Explicit type assertion that the public shape we code against still holds.
    let v = ObjectValidation {
        name: "003_cracker_box".to_string(),
        mesh_present: true,
        texture_present: true,
    };
    assert!(v.is_complete());
    let partial = ObjectValidation {
        name: "004_sugar_box".to_string(),
        mesh_present: true,
        texture_present: false,
    };
    assert!(!partial.is_complete());
}

// -----------------------------------------------------------------------------
// Signature pins — compile-only. These evaluate the ycbust functions that
// bevy-sensor wraps, so a breaking signature change fails here (and at the
// real call sites in `src/lib.rs`).
// -----------------------------------------------------------------------------

#[test]
fn ycbust_contract_download_ycb_is_callable() {
    // Build the future but never poll it — network call never happens, but
    // the signature is type-checked by construction.
    let fut = ycbust::download_ycb(
        Subset::Representative,
        Path::new("/nonexistent-path-for-type-check-only"),
        DownloadOptions::default(),
    );
    drop(fut);
}

#[test]
fn ycbust_contract_extract_tgz_is_callable() {
    // Call on a path we know doesn't exist; we only care that it returns an
    // error rather than failing to compile.
    let dir = tempfile::tempdir().expect("tempdir");
    let err = ycbust::extract_tgz(
        &dir.path().join("does_not_exist.tgz"),
        dir.path(),
        false,
    );
    assert!(err.is_err());
}

#[test]
fn ycbust_contract_download_objects_is_callable() {
    // Upstream `download_objects` (shipped in v0.3.3) — bevy-sensor's
    // `ycb::download_objects` wrapper is a thin delegate over this.
    let fut = ycbust::download_objects(
        &["003_cracker_box"],
        Path::new("/nonexistent-path-for-type-check-only"),
        DownloadOptions::default(),
    );
    drop(fut);
}

// -----------------------------------------------------------------------------
// GOOGLE_16K path consts (v0.3.3) — `render.rs` joins these with per-object
// directories in place of hardcoded strings. Literal values are part of the
// contract: if upstream switches to `google_16k\\textured.obj` on any platform
// we need to know immediately, not at asset-load time.
// -----------------------------------------------------------------------------

#[test]
fn ycbust_contract_google_16k_mesh_relative_literal() {
    assert_eq!(GOOGLE_16K_MESH_RELATIVE, "google_16k/textured.obj");
}

#[test]
fn ycbust_contract_google_16k_texture_relative_literal() {
    assert_eq!(GOOGLE_16K_TEXTURE_RELATIVE, "google_16k/texture_map.png");
}

#[test]
fn ycbust_contract_google_16k_consts_compose_with_object_dir() {
    // The whole point of the consts: `object_dir.join(MESH_RELATIVE)` must
    // yield the same path `object_mesh_path(root, id)` produces.
    let root = Path::new("/tmp/ycb");
    let per_object = root.join("003_cracker_box");
    assert_eq!(
        per_object.join(GOOGLE_16K_MESH_RELATIVE),
        object_mesh_path(root, "003_cracker_box"),
    );
    assert_eq!(
        per_object.join(GOOGLE_16K_TEXTURE_RELATIVE),
        object_texture_path(root, "003_cracker_box"),
    );
}

// -----------------------------------------------------------------------------
// YcbError (v0.4.0) — bevy-sensor maps these into `RenderError` in follow-up
// work. Pinning the match shape here means a future variant rename or removal
// fails in this suite before it reaches the mapping code.
// -----------------------------------------------------------------------------

#[test]
fn ycbust_contract_error_variants_matchable() {
    // Cover every concrete variant we rely on distinguishing. `YcbError` is
    // `#[non_exhaustive]`, so the wildcard arm is required to absorb variants
    // added upstream without breaking us — but removing one of these named
    // arms *will* fail to compile here.
    fn classify(err: YcbError) -> &'static str {
        match err {
            YcbError::Network(_) => "network",
            YcbError::HttpStatus { .. } => "http_status",
            YcbError::Extraction { .. } => "extraction",
            YcbError::Io(_) => "io",
            YcbError::Integrity { .. } => "integrity",
            YcbError::InvalidResponse(_) => "invalid_response",
            YcbError::UnsafeArchive(_) => "unsafe_archive",
            YcbError::Other(_) => "other",
            _ => "unknown_non_exhaustive",
        }
    }
    let http = YcbError::HttpStatus {
        status: 404,
        url: "https://example.com".into(),
    };
    assert_eq!(classify(http), "http_status");

    let integrity = YcbError::Integrity {
        path: "/tmp/foo.tgz".into(),
        reason: "expected 1024, got 512".into(),
    };
    assert_eq!(classify(integrity), "integrity");
}

#[test]
fn ycbust_contract_error_from_and_into_anyhow() {
    // Bidirectional conversion keeps anyhow-using consumers working.
    let y = YcbError::HttpStatus {
        status: 404,
        url: "https://example.com".into(),
    };
    let a: anyhow::Error = y.into();
    assert!(a.to_string().contains("404"));
}

#[test]
fn ycbust_contract_result_alias_present() {
    // `ycbust::Result<T>` should alias to `Result<T, YcbError>`.
    fn _check(_r: ycbust::Result<()>) {}
}

// -----------------------------------------------------------------------------
// Blocking wrappers (v0.4.0, `blocking` feature) — bevy-sensor's `prerender`
// binary uses these to avoid spinning up a throwaway Tokio runtime. They're
// part of this crate's hard dependency surface now.
// -----------------------------------------------------------------------------

#[test]
fn ycbust_contract_blocking_wrappers_present() {
    // Compile-check: both wrappers exist with the expected signatures.
    let _f1: fn(Subset, &Path, DownloadOptions) -> ycbust::Result<()> =
        ycbust::blocking::download_ycb_blocking;
    let _f2: fn(&[&str], &Path, DownloadOptions) -> ycbust::Result<()> =
        ycbust::blocking::download_objects_blocking;
}

#[test]
fn ycbust_contract_blocking_download_objects_empty_slice_is_noop() {
    // Empty slice should no-op without hitting the network. Also exercises
    // the runtime-construction path in the blocking wrapper.
    let dir = tempfile::tempdir().expect("tempdir");
    let result = ycbust::blocking::download_objects_blocking(
        &[],
        dir.path(),
        DownloadOptions::default(),
    );
    assert!(result.is_ok());
}
