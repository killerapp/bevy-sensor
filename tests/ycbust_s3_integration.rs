//! Compatibility marker for the historical `ycbust-s3` feature.
//!
//! ycbust 0.5 removed its optional S3 streaming API. bevy-sensor keeps this
//! Cargo feature as a no-op so downstream feature selections do not fail at
//! resolution time, but there is no upstream S3 surface left to exercise here.

#![cfg(feature = "ycbust-s3")]

#[test]
fn ycbust_s3_feature_is_retained_as_noop() {
    // Compile-time coverage: `cargo test --features ycbust-s3` should still
    // resolve and run even though ycbust no longer exposes `ycbust::s3`.
}
