use super::{Command, parse_args};
use crate::serve::{SpeculativeAsset, build_carrier_flags};
use std::path::Path;

#[test]
fn retired_prefill_controls_refuse_before_launch() {
    for args in [
        vec!["serve", "--gdn-prefill-only"],
        vec!["serve", "--native-prefill-router-fast", "on"],
        vec![
            "serve",
            "--speed-preset",
            "--native-prefill-router-fast",
            "on",
        ],
    ] {
        let Command::Serve(opts) = parse_args(args).unwrap().command else {
            panic!("serve");
        };
        assert!(
            build_carrier_flags(
                &opts,
                SpeculativeAsset::MtpSidecar(Path::new("/tmp/mtp.gguf")),
                Path::new("/tmp/state")
            )
            .is_err()
        );
    }
}

#[test]
fn runtime_controls_are_absent_by_default_and_off_is_not_forwarded() {
    for args in [
        vec!["serve"],
        vec![
            "serve",
            "--native-prepared-validation",
            "off",
            "--native-dflash2-draft-policy",
            "always",
            "--worker-burst-pacing",
            "off",
        ],
    ] {
        let Command::Serve(opts) = parse_args(args).unwrap().command else {
            panic!("serve");
        };
        let flags = build_carrier_flags(
            &opts,
            SpeculativeAsset::MtpSidecar(Path::new("/tmp/mtp.gguf")),
            Path::new("/tmp/state"),
        )
        .unwrap();
        for flag in [
            "--native-prepared-validation",
            "--native-verify-cost-profile",
            "--native-dflash2-draft-policy",
            "--native-verify-m8-splits",
            "--worker-burst-pacing",
        ] {
            assert!(!flags.iter().any(|f| f == flag));
        }
    }
}

#[test]
fn worker_burst_pacing_is_forwarded_only_when_enabled() {
    let Command::Serve(opts) = parse_args(["serve", "--worker-burst-pacing", "on"])
        .unwrap()
        .command
    else {
        panic!("serve");
    };
    let flags = build_carrier_flags(
        &opts,
        SpeculativeAsset::MtpSidecar(Path::new("/tmp/mtp.gguf")),
        Path::new("/tmp/state"),
    )
    .unwrap();
    assert!(
        flags
            .windows(2)
            .any(|pair| pair == ["--worker-burst-pacing", "on"])
    );
    assert!(parse_args(["serve", "--worker-burst-pacing", "auto"]).is_err());
}

#[test]
fn prepared_validation_is_forwarded_only_when_enabled() {
    let Command::Serve(opts) = parse_args(["serve", "--native-prepared-validation", "on"])
        .unwrap()
        .command
    else {
        panic!("serve");
    };
    let flags = build_carrier_flags(
        &opts,
        SpeculativeAsset::MtpSidecar(Path::new("/tmp/mtp.gguf")),
        Path::new("/tmp/state"),
    )
    .unwrap();
    assert!(
        flags
            .windows(2)
            .any(|pair| pair == ["--native-prepared-validation", "on"])
    );
    assert!(parse_args(["serve", "--native-prepared-validation", "auto"]).is_err());
}

#[test]
fn drafting_and_numerical_controls_refuse_missing_prerequisites() {
    for args in [
        vec!["serve", "--native-dflash2-draft-policy", "auto"],
        vec!["serve", "--native-verify-m8-splits", "/tmp/splits.json"],
    ] {
        let Command::Serve(opts) = parse_args(args).unwrap().command else {
            panic!("serve");
        };
        assert!(
            build_carrier_flags(
                &opts,
                SpeculativeAsset::MtpSidecar(Path::new("/tmp/mtp.gguf")),
                Path::new("/tmp/state")
            )
            .is_err()
        );
    }
}

#[test]
fn controller_dflash2_explicit_off_survives_preset_expansion() {
    let Command::Serve(opts) = parse_args([
        "serve",
        "--speed-preset",
        "--drafter",
        "dflash2",
        "--dflash2-dir",
        "/tmp/dflash2",
        "--no-verify-nax",
        "--native-verify-hc-gdn-fused",
        "off",
    ])
    .unwrap()
    .command
    else {
        panic!("serve");
    };
    let flags = build_carrier_flags(
        &opts,
        SpeculativeAsset::DFlash2Dir(Path::new("/tmp/dflash2")),
        Path::new("/tmp/state"),
    )
    .unwrap();
    assert!(
        flags
            .windows(2)
            .any(|p| p == ["--native-verify-nax", "off"])
    );
    assert!(!flags.iter().any(|f| f == "--native-verify-nax-m8"));
    assert!(
        flags
            .windows(2)
            .any(|p| p == ["--native-prefill-dense", "nax"])
    );
}
