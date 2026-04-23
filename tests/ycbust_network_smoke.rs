//! ycbust network smoke tests.
//!
//! These hit the real YCB S3 endpoint (`https://ycb-benchmarks.s3.amazonaws.com`)
//! and download actual archives. All tests here are `#[ignore]`d so `cargo test`
//! never runs them by default — CI and offline builds stay fast and hermetic.
//!
//! Running:
//!   cargo test --test ycbust_network_smoke -- --ignored --nocapture
//!
//! What this guards:
//! - `ycbust::blocking::download_ycb_blocking` actually completes end-to-end
//!   (the path `prerender` binary now takes after the v0.4.x migration).
//! - `DownloadOptions::concurrency > 1` is faster than serial on a cold cache.
//! - Extracted mesh layout still matches the contract (`GOOGLE_16K_*` consts).
//!
//! Failures here after a ycbust bump mean: coordinate with
//! <https://github.com/Agentic-Insights/ycbust>.

use std::time::Instant;
use ycbust::{
    blocking::download_ycb_blocking, object_mesh_path, object_texture_path, DownloadOptions,
    Subset, REPRESENTATIVE_OBJECTS, TBP_STANDARD_OBJECTS,
};

fn silent_options(concurrency: usize) -> DownloadOptions {
    let mut opts = DownloadOptions::default();
    opts.show_progress = false;
    opts.concurrency = concurrency;
    opts
}

fn assert_subset_present<P: AsRef<std::path::Path>>(dir: P, ids: &[&str]) {
    let dir = dir.as_ref();
    for id in ids {
        let mesh = object_mesh_path(dir, id);
        let tex = object_texture_path(dir, id);
        assert!(mesh.exists(), "{id}: missing mesh at {}", mesh.display());
        assert!(tex.exists(), "{id}: missing texture at {}", tex.display());
    }
}

#[test]
#[ignore = "hits real network; run with --ignored --nocapture"]
fn ycbust_network_smoke_blocking_download_representative() {
    // The exact path `prerender` takes post-migration: blocking wrapper +
    // default options. Cold cache, no-op second call.
    let dir = tempfile::tempdir().expect("tempdir");

    let start = Instant::now();
    download_ycb_blocking(
        Subset::Representative,
        dir.path(),
        silent_options(1),
    )
    .expect("initial download failed");
    let first = start.elapsed();

    assert_subset_present(dir.path(), REPRESENTATIVE_OBJECTS);
    println!("representative cold download (concurrency=1): {first:?}");

    // Second call should short-circuit on extracted-mesh detection — no
    // re-download. This validates the resume path we rely on.
    let start = Instant::now();
    download_ycb_blocking(
        Subset::Representative,
        dir.path(),
        silent_options(1),
    )
    .expect("warm download failed");
    let warm = start.elapsed();
    println!("representative warm resume (concurrency=1): {warm:?}");

    // Warm resume should be dramatically faster. 10x is conservative — a
    // warm run does only the short-circuit checks, no HEAD/GET.
    assert!(
        warm * 10 < first,
        "warm resume ({warm:?}) not meaningfully faster than cold ({first:?}) — resume check may be broken"
    );
}

#[test]
#[ignore = "hits real network; downloads ~10 YCB archives; run with --ignored --nocapture"]
fn ycbust_network_smoke_parallel_beats_serial_tbp_standard() {
    // TBP standard = 10 objects, enough work that parallel concurrency
    // should visibly win over serial. Each subset into its own tempdir
    // so neither run benefits from the other's cache.
    let serial_dir = tempfile::tempdir().expect("tempdir serial");
    let parallel_dir = tempfile::tempdir().expect("tempdir parallel");

    let start = Instant::now();
    download_ycb_blocking(
        Subset::TbpStandard,
        serial_dir.path(),
        silent_options(1),
    )
    .expect("serial download failed");
    let serial = start.elapsed();
    assert_subset_present(serial_dir.path(), TBP_STANDARD_OBJECTS);

    let start = Instant::now();
    download_ycb_blocking(
        Subset::TbpStandard,
        parallel_dir.path(),
        silent_options(4),
    )
    .expect("parallel download failed");
    let parallel = start.elapsed();
    assert_subset_present(parallel_dir.path(), TBP_STANDARD_OBJECTS);

    println!("tbp_standard serial   (concurrency=1): {serial:?}");
    println!("tbp_standard parallel (concurrency=4): {parallel:?}");
    println!(
        "speedup: {:.2}x",
        serial.as_secs_f64() / parallel.as_secs_f64()
    );

    // Soft check — don't fail on pure noise, but warn if parallel isn't at
    // least marginally faster. Network variance can legitimately cause a
    // ~10% regression on any given run.
    if parallel >= serial {
        eprintln!(
            "WARN: parallel ({parallel:?}) not faster than serial ({serial:?}) — \
             network noise or a concurrency regression upstream"
        );
    }
}
