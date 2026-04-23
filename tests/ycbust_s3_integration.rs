//! ycbust S3 integration tests.
//!
//! Gated behind the `ycbust-s3` Cargo feature (which enables `ycbust/s3`).
//! These exist so the S3 streaming path upstream stays exercised from
//! bevy-sensor's side — bevy-sensor does not use S3 at runtime, but neocortx
//! deployments may, and we want a canary if the upstream surface drifts.
//!
//! Conventions:
//! - `ycbust_s3_contract_*` — offline compile/parse-level tests. Run by default
//!   when `--features ycbust-s3` is set.
//! - `ycbust_s3_live_*` — live tests hitting a real bucket. `#[ignore]`d so
//!   they never run on CI by accident. Opt in with
//!   `cargo test --features ycbust-s3 -- --ignored ycbust_s3_live`
//!   and set `BEVY_SENSOR_S3_TEST_BUCKET=my-bucket`.
//!
//! Running:
//!   cargo test --features ycbust-s3 ycbust_s3_contract
//!   cargo test --features ycbust-s3 -- --ignored ycbust_s3_live

#![cfg(feature = "ycbust-s3")]

use ycbust::s3::S3Destination;

// -----------------------------------------------------------------------------
// Offline contract — parse / constructor behavior. No network.
// -----------------------------------------------------------------------------

#[test]
fn ycbust_s3_contract_destination_parses_bucket_and_prefix() {
    let dest = S3Destination::from_url("s3://my-bucket/ycb-data/").expect("parse");
    assert_eq!(dest.bucket, "my-bucket");
    assert_eq!(dest.prefix, "ycb-data/");
    assert_eq!(dest.region, "us-east-1");
}

#[test]
fn ycbust_s3_contract_destination_normalizes_missing_trailing_slash() {
    let dest = S3Destination::from_url("s3://my-bucket/ycb-data").expect("parse");
    assert_eq!(dest.prefix, "ycb-data/");
}

#[test]
fn ycbust_s3_contract_destination_accepts_bucket_root() {
    let dest = S3Destination::from_url("s3://my-bucket/").expect("parse");
    assert_eq!(dest.bucket, "my-bucket");
    assert_eq!(dest.prefix, "");
}

#[test]
fn ycbust_s3_contract_destination_rejects_non_s3_scheme() {
    assert!(S3Destination::from_url("https://my-bucket/").is_err());
}

#[test]
fn ycbust_s3_contract_destination_rejects_empty_bucket() {
    assert!(S3Destination::from_url("s3://").is_err());
}

#[test]
fn ycbust_s3_contract_destination_with_region_is_chainable() {
    let dest = S3Destination::from_url("s3://my-bucket/p/")
        .expect("parse")
        .with_region("us-west-2");
    assert_eq!(dest.region, "us-west-2");
}

#[test]
fn ycbust_s3_contract_destination_full_path_joins_prefix() {
    let dest = S3Destination::from_url("s3://b/ycb/").unwrap();
    assert_eq!(dest.full_path("003_cracker_box.tgz"), "ycb/003_cracker_box.tgz");
}

#[test]
fn ycbust_s3_contract_destination_to_url_roundtrips_prefix() {
    let dest = S3Destination::from_url("s3://b/ycb/").unwrap();
    assert_eq!(dest.to_url(), "s3://b/ycb/");
}

// -----------------------------------------------------------------------------
// Live integration — opt-in, requires AWS credentials + a test bucket.
// -----------------------------------------------------------------------------

#[tokio::test]
#[ignore = "requires real S3 bucket + AWS creds; set BEVY_SENSOR_S3_TEST_BUCKET"]
async fn ycbust_s3_live_download_representative_to_bucket() {
    use ycbust::s3::download_ycb_to_s3;
    use ycbust::{DownloadOptions, Subset};

    let bucket = std::env::var("BEVY_SENSOR_S3_TEST_BUCKET")
        .expect("set BEVY_SENSOR_S3_TEST_BUCKET to an AWS bucket you own");
    let dest =
        S3Destination::from_url(&format!("s3://{}/bevy-sensor-ycbust-live/", bucket)).unwrap();

    download_ycb_to_s3(Subset::Representative, dest, DownloadOptions::default(), None)
        .await
        .expect("live S3 upload of representative subset");
}

#[tokio::test]
#[ignore = "requires real AWS creds; verifies ycbust can load them end-to-end"]
async fn ycbust_s3_live_credentials_loadable() {
    let id = ycbust::s3::check_aws_credentials(None)
        .await
        .expect("AWS credentials must resolve via default chain");
    assert!(id.contains("AWS credentials"));
}
