#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::fs;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::os::unix::fs::{PermissionsExt, symlink};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use tokspur_cli::state::StateLock;
use tokspur_cli::verify::{
    NATIVE_SOURCE_SCOPE, audit_macho_dependencies, collect_native_sources, collect_rust_sources,
    inspect_and_hash_nofollow, is_native_build_input,
};
use tokspur_json::{Json, parse_json};
use tokspur_sys::process::kill;
use tokspur_sys::{ESRCH, SIGINT, SIGTERM};
use tokspur_text::sha256::{Sha256, hex};

/// The packaged CLI can inspect config metadata on a machine with no Tokspur
/// state, target shards, draft sidecar, or built worker.
#[test]
fn check_model_config_before_adoption() {
    let dir = TempDir::new("model-config");
    let path = dir.0.join("config.json");
    let layers: Vec<_> = (0..48)
        .map(|i| {
            if i % 4 == 3 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect();
    let text = format!(
        r#"{{"model_type":"qwen4_exp_text","hidden_size":2560,"num_hidden_layers":48,
        "vocab_size":248320,"num_attention_heads":24,"num_key_value_heads":2,"head_dim":256,
        "num_experts":512,"num_experts_per_tok":10,"moe_intermediate_size":640,
        "shared_expert_intermediate_size":640,"full_attention_interval":4,"linear_conv_kernel_dim":4,
        "linear_num_key_heads":16,"linear_num_value_heads":48,"linear_key_head_dim":128,
        "linear_value_head_dim":128,"hc_count":4,"hc_lowrank":320,"indexer_n_heads":4,
        "indexer_kv_heads":1,"indexer_head_dim":128,"indexer_budget":2048,"indexer_compress_ratio":4,
        "ngram_size":3,"heads_per_ngram":8,"ple_conv_kernel_size":4,"ple_embed_dim":2560,
        "rms_norm_eps":1e-6,"partial_rotary_factor":0.25,"rope_parameters":{{"rope_theta":10000000,"partial_rotary_factor":0.25,"mrope_section":[11,11,10]}},
        "max_position_embeddings":262144,"layer_types":{layers:?},"ple_layer_ids":[2]}}"#
    );
    let check = |config: &str| {
        fs::write(&path, config).unwrap();
        Command::new(env!("CARGO_BIN_EXE_tokspur"))
            .current_dir(&dir.0)
            .env("TOKSPUR_STATE_ROOT", dir.0.join("unused-state"))
            .arg("check-model-config")
            .arg(&path)
            .output()
            .unwrap()
    };
    let nested = format!(
        r#"{{"model_type":"qwen4_exp","vision_config":{{"hidden_size":64}},"text_config":{text}}}"#
    );
    let duplicate = nested.replace(r#""vision_config":"#, r#""hidden_size":2560,"rms_norm_eps":0.000001,"rope_scaling":{"rope_theta":1e7},"vision_config":"#);
    let valid_bounds = nested.replace("1e-6", "0.01").replace("10000000", "1.01");
    for config in [&text, &nested, &duplicate, &valid_bounds] {
        let output = check(config);
        assert!(output.status.success(), "{output:?}");
        assert!(
            String::from_utf8(output.stdout)
                .unwrap()
                .contains("checked fields match")
        );
    }
    for invalid_rope in [
        nested.replace("[11,11,10]", "[12,10,10]"),
        nested.replacen("0.25", "0.5", 1),
        nested.replace(
            "\"vision_config\":",
            "\"partial_rotary_factor\":0.5,\"vision_config\":",
        ),
    ] {
        let output = check(&invalid_rope);
        assert_eq!(output.status.code(), Some(1));
        let report = String::from_utf8(output.stdout).unwrap();
        assert!(
            report.contains("partial_rotary_factor expected")
                || report.contains("mrope_section expected")
        );
    }
    for (field, replacement, expected) in [
        (
            r#""indexer_kv_heads":1"#,
            r#""indexer_kv_heads":2"#,
            "indexer_kv_heads expected 1",
        ),
        (
            r#""rms_norm_eps":1e-6"#,
            r#""rms_norm_eps":0"#,
            "rms_norm_eps expected",
        ),
        (
            r#""rms_norm_eps":1e-6"#,
            r#""rms_norm_eps":0.011"#,
            "rms_norm_eps expected",
        ),
        (
            r#""rope_theta":10000000"#,
            r#""rope_theta":1"#,
            "rope_theta expected",
        ),
        (
            r#""rope_theta":10000000"#,
            r#""rope_theta":"10000000""#,
            "rope_theta expected",
        ),
        (
            r#""vision_config":"#,
            r#""hidden_size":1536,"vision_config":"#,
            "root.hidden_size conflicts",
        ),
        (
            r#""vision_config":"#,
            r#""max_position_embeddings":131072,"vision_config":"#,
            "root.max_position_embeddings conflicts",
        ),
        (
            r#""vision_config":"#,
            r#""rope_theta":1000,"vision_config":"#,
            "conflicts with the other rope_theta",
        ),
        (
            r#""vision_config":"#,
            r#""rms_norm_eps":0.001,"vision_config":"#,
            "conflicts with the other rms_norm_eps",
        ),
    ] {
        let output = check(&nested.replace(field, replacement));
        assert_eq!(output.status.code(), Some(1), "{field}");
        assert!(
            String::from_utf8(output.stdout).unwrap().contains(expected),
            "{expected}"
        );
    }
    for (declaration, expected) in [
        (r#""indexer_kv_heads":1,"#, "indexer_kv_heads"),
        (r#""rms_norm_eps":1e-6,"#, "rms_norm_eps"),
        (r#""rope_theta":10000000,"#, "rope_theta"),
    ] {
        let output = check(&nested.replace(declaration, ""));
        assert_eq!(output.status.code(), Some(1));
        assert!(
            String::from_utf8(output.stdout)
                .unwrap()
                .contains(&format!("Unverified fields: {expected}"))
        );
    }
    let changed = nested
        .replace("\"num_experts_per_tok\":10", "\"num_experts_per_tok\":4")
        .replace("262144", "8192")
        .replace("linear_attention", "full_attention");
    let output = check(&changed);
    assert_eq!(output.status.code(), Some(1));
    let text = String::from_utf8(output.stdout).unwrap();
    assert!(text.contains("num_experts_per_tok expected 10, found 4"));
    assert!(text.contains("max_position_embeddings requires at least 65536"));
    assert!(text.contains("layer_types expected"));
    let missing = check(r#"{"model_type":"future_flash","hidden_size":2560}"#);
    assert_eq!(missing.status.code(), Some(1));
    assert!(
        String::from_utf8(missing.stdout)
            .unwrap()
            .contains("insufficient evidence")
    );
    for invalid in [
        "[]",
        "{}",
        r#"{"text_config":null,"num_hidden_layers":48}"#,
        r#"{"hidden_size":true}"#,
        r#"{"hidden_size":2560,"hidden_size":1}"#,
        "not json",
    ] {
        assert_eq!(check(invalid).status.code(), Some(1), "{invalid}");
    }
    assert!(!dir.0.join("unused-state").exists());
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static SERVE_TEST_MUTEX: Mutex<()> = Mutex::new(());

struct TempDir(PathBuf);

impl TempDir {
    fn new(prefix: &str) -> Self {
        let base = fs::canonicalize(std::env::temp_dir()).unwrap();
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        let dir = base.join(format!(
            "tokspur-cli-test-{prefix}-{}-{id}",
            std::process::id()
        ));
        let _ignored = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o700)).unwrap();
        Self(dir)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ignored = fs::remove_dir_all(&self.0);
    }
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex(&h.finalize())
}

fn free_loopback_port() -> u16 {
    let listener = TcpListener::bind(("127.0.0.1", 0)).unwrap();
    listener.local_addr().unwrap().port()
}

fn write_mode(path: &Path, data: &[u8], mode: u32) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    fs::write(path, data).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(mode)).unwrap();
}

fn dump_json(value: &Json) -> Vec<u8> {
    let mut out = value.dump().unwrap();
    out.push(b'\n');
    out
}

const FAKE_SERVER_C: &str = r#"
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static volatile sig_atomic_t got_stop = 0;

static void on_stop(int sig) {
    (void)sig;
    got_stop = 1;
}

static void write_pid_file(const char *env_key, pid_t pid) {
    const char *path = getenv(env_key);
    if (!path || !path[0]) {
        return;
    }
    FILE *f = fopen(path, "w");
    if (f) {
        fprintf(f, "%d\n", (int)pid);
        fclose(f);
    }
}

// A model-free stand-in for the verified native pack protocol. Exercise
// argv conversion here, including absolute paths with spaces and all four shards.
static int pack_command(int argc, char **argv) {
    const char *name = strrchr(argv[0], '/');
    if (!name || strcmp(name + 1, "tokspur-native") != 0) return 5;
    const char *out = NULL, *mtp = NULL, *reserve = NULL, *chunk = NULL;
    int shards = 0;
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) return 6;
        const char *value = argv[i + 1];
        if (!strcmp(argv[i], "--output")) out = value;
        else if (!strcmp(argv[i], "--mtp")) mtp = value;
        else if (!strcmp(argv[i], "--reserve-bytes")) reserve = value;
        else if (!strcmp(argv[i], "--chunk-bytes")) chunk = value;
        else if (!strcmp(argv[i], "--gguf") && value[0] == '/' && access(value, R_OK) == 0) ++shards;
        else return 6;
    }
    if (!out || out[0] != '/' || !mtp || mtp[0] != '/' || access(mtp, R_OK) != 0 || shards != 4 || !reserve)
        return 6;
    if (strstr(out, "cancel-pending")) {
        signal(SIGTERM, on_stop);
        FILE *f = fopen(out, "w");
        if (!f) return 9;
        fputs("streaming", f);
        fclose(f);
        while (!got_stop) usleep(1000);
        unlink(out);
        return 10;
    }
    if (strstr(out, "native-refusal")) {
        fputs("native fixture refused output\n", stderr);
        return 7;
    }
    if (!strcmp(argv[1], "pack-v3-plan")) {
        if (!chunk || strcmp(chunk, "16777216")) return 8;
        printf("{\"recipe\":\"tokspur-pack-v3\",\"version\":3,\"tensor_count\":1254,\"payload_bytes\":4096,"
               "\"file_bytes\":8192,\"available_bytes\":999999,\"reserve_bytes\":%s,"
               "\"disk_ok\":%s,\"streaming_buffer_bytes\":4096,\"streaming_ceiling_bytes\":8192,\"ram_ok\":%s}\n",
               reserve, strstr(out, "disk-fail") ? "false" : "true", strstr(out, "ram-fail") ? "false" : "true");
    } else {
        if (!chunk || strcmp(chunk, "2097152") || strcmp(reserve, "1610612736")) return 8;
        printf("{\"status\":\"verified-existing\",\"recipe\":\"tokspur-pack-v3\",\"version\":3,"
               "\"tensor_count\":1254,\"payload_bytes\":4096,\"file_bytes\":8192,"
               "\"manifest_sha256\":\"0000000000000000000000000000000000000000000000000000000000000000\"}\n");
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && (!strcmp(argv[1], "pack-v3-plan") || !strcmp(argv[1], "pack-v3-build")))
        return pack_command(argc, argv);
    const char *mode = getenv("FAKE_SERVER_MODE");
    if (!mode) {
        mode = "sleep_sigterm";
    }

    if (strcmp(mode, "exit_code") == 0) {
        write_pid_file("FAKE_SERVER_PID_FILE", getpid());
        const char *code_s = getenv("FAKE_EXIT_CODE");
        return code_s ? atoi(code_s) : 0;
    }

    if (strcmp(mode, "ignore_sigterm") == 0) {
        signal(SIGTERM, SIG_IGN);
        signal(SIGINT, SIG_IGN);
        write_pid_file("FAKE_SERVER_PID_FILE", getpid());
        for (int i = 0; i < 300; ++i) {
            usleep(100000);
        }
        return 0;
    }

    if (strcmp(mode, "spawn_detached") == 0) {
        pid_t child = fork();
        if (child == 0) {
            setsid();
            signal(SIGTERM, on_stop);
            signal(SIGINT, on_stop);
            write_pid_file("FAKE_DESCENDANT_PID_FILE", getpid());
            while (!got_stop) {
                usleep(20000);
            }
            _exit(0);
        }
    }

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = on_stop;
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGINT, &sa, NULL);

    write_pid_file("FAKE_SERVER_PID_FILE", getpid());
    while (!got_stop) {
        usleep(20000);
    }
    return 0;
}
"#;

fn compile_fake_macho(tmp: &Path, out_bin: &Path) {
    let src = tmp.join("fake_server.c");
    fs::write(&src, FAKE_SERVER_C).unwrap();
    if let Some(parent) = out_bin.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    let status = Command::new("/usr/bin/xcrun")
        .args(["clang", "-O2", "-Wall", "-Werror"])
        .arg(&src)
        .arg("-o")
        .arg(out_bin)
        .status()
        .unwrap();
    assert!(
        status.success(),
        "xcrun clang failed compiling fake_server.c"
    );
    fs::set_permissions(out_bin, fs::Permissions::from_mode(0o755)).unwrap();
}

fn clang_version_output() -> String {
    let out = Command::new("/usr/bin/xcrun")
        .args(["clang++", "--version"])
        .output()
        .unwrap();
    assert!(out.status.success());
    String::from_utf8(out.stdout).unwrap()
}

struct SyntheticCheckout {
    _tmp: TempDir,
    root: PathBuf,
    target_dir: PathBuf,
    draft_dir: PathBuf,
}

impl SyntheticCheckout {
    fn create(prefix: &str) -> Self {
        let tmp = TempDir::new(prefix);
        let root = tmp.path().join("repo");
        fs::create_dir_all(&root).unwrap();
        fs::set_permissions(&root, fs::Permissions::from_mode(0o700)).unwrap();

        let target_dir = tmp.path().join("weights/target");
        let draft_dir = tmp.path().join("weights/draft");
        fs::create_dir_all(&target_dir).unwrap();
        fs::create_dir_all(draft_dir.join("MTP")).unwrap();

        let shard_names = [
            "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
            "Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf",
            "Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf",
            "Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf",
        ];
        let mut target_files = Vec::new();
        for (i, name) in shard_names.iter().enumerate() {
            let content = format!("synthetic-gguf-shard-{i}\n");
            let sha = sha256_bytes(content.as_bytes());
            fs::write(target_dir.join(name), content.as_bytes()).unwrap();
            target_files.push(Json::new_object([
                ("name", Json::from(*name)),
                ("size", Json::integer(content.len() as u64)),
                ("sha256", Json::from(sha)),
            ]));
        }

        let mtp_rel = "MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf";
        let mtp_bytes = b"synthetic-mtp-sidecar\n";
        let mtp_sha = sha256_bytes(mtp_bytes);
        fs::write(draft_dir.join(mtp_rel), mtp_bytes).unwrap();

        let legacy_cfg = b"{\"architectures\":[\"DFlash2\"]}\n";
        let legacy_st = b"synthetic-legacy-safetensors\n";
        let legacy_files = Json::Array(vec![
            Json::new_object([
                ("name", Json::from("config.json")),
                ("size", Json::integer(legacy_cfg.len() as u64)),
                ("sha256", Json::from(sha256_bytes(legacy_cfg))),
            ]),
            Json::new_object([
                ("name", Json::from("model.safetensors")),
                ("size", Json::integer(legacy_st.len() as u64)),
                ("sha256", Json::from(sha256_bytes(legacy_st))),
            ]),
        ]);

        let assets_json = Json::new_object([
            ("schema", Json::integer(1)),
            (
                "target",
                Json::new_object([
                    ("repo", Json::from("unsloth/Qwen3.8-Flash-Next-GGUF")),
                    (
                        "revision",
                        Json::from("766911a6b7369840a91dbcd95f9f997acaab6cd6"),
                    ),
                    ("variant", Json::from("UD-Q4_K_XL")),
                    ("files", Json::Array(target_files)),
                ]),
            ),
            (
                "draft",
                Json::new_object([
                    ("repo", Json::from("unsloth/Qwen3.8-Flash-Next-GGUF")),
                    (
                        "revision",
                        Json::from("766911a6b7369840a91dbcd95f9f997acaab6cd6"),
                    ),
                    (
                        "files",
                        Json::Array(vec![Json::new_object([
                            ("name", Json::from(mtp_rel)),
                            ("size", Json::integer(mtp_bytes.len() as u64)),
                            ("sha256", Json::from(mtp_sha)),
                        ])]),
                    ),
                ]),
            ),
            (
                "legacy_dflash2_draft",
                Json::new_object([
                    ("repo", Json::from("incoai/Qwen3.8-Flash-Next-DFlash2")),
                    (
                        "revision",
                        Json::from("a22a1744e85c17364dcc54a6a420af7135d4c7f6"),
                    ),
                    ("files", legacy_files),
                ]),
            ),
        ]);
        write_mode(
            &root.join("profiles/flash-next-assets.json"),
            &dump_json(&assets_json),
            0o644,
        );

        let hw_json = Json::new_object([
            ("schema", Json::integer(1)),
            ("name", Json::from("m5-max-128-baseline")),
            (
                "hardware",
                Json::new_object([
                    ("chip", Json::from("Apple M5 Max")),
                    ("gpu_cores", Json::integer(40)),
                    ("unified_memory_gib", Json::integer(1)),
                ]),
            ),
            ("context_tokens", Json::integer(65536)),
            ("native_memory_gib", Json::integer(96)),
            ("api_memory_ceiling_gib", Json::integer(108)),
            ("kv_format", Json::from("bf16")),
            ("state_format", Json::from("fp32")),
            ("verify_rows", Json::from("all")),
            ("ple_memory_mib", Json::integer(512)),
            ("resident_experts", Json::Bool(true)),
            ("ane_prefill", Json::Bool(false)),
            ("temporary_ssd_cache_gib", Json::integer(2)),
            ("persistent_cache", Json::Bool(false)),
            ("text_only", Json::Bool(true)),
            ("queue_size", Json::integer(1)),
            ("serialize_requests", Json::Bool(true)),
            ("host", Json::from("127.0.0.1")),
            ("port", Json::integer(18081)),
            ("public_model", Json::from("tokspur")),
        ]);
        write_mode(
            &root.join("profiles/m5-max-128.json"),
            &dump_json(&hw_json),
            0o644,
        );
        write_mode(
            &root.join("profiles/owned-target-experimental.json"),
            b"{\"schema\": 1}\n",
            0o644,
        );

        // Minimal source trees for `verify-build`. The Markdown note is documentation, so the recorded `native_sources` leaves it out, and the licence text is a pinned build input.
        write_mode(&root.join("native/Makefile"), b"# synthetic\n", 0o644);
        write_mode(
            &root.join("native/README.md"),
            b"# synthetic notes\n",
            0o644,
        );
        write_mode(
            &root.join("native/data/UNICODE-LICENSE.txt"),
            b"synthetic licence\n",
            0o644,
        );
        write_mode(
            &root.join("rust/Cargo.toml"),
            b"[workspace]\nmembers = []\n",
            0o644,
        );
        write_mode(&root.join("rust/Cargo.lock"), b"version = 4\n", 0o644);
        write_mode(
            &root.join("rust/rust-toolchain.toml"),
            b"[toolchain]\nchannel = \"1.99.1\"\n",
            0o644,
        );
        write_mode(
            &root.join("rust/.cargo/config.toml"),
            b"[build]\nrustflags = []\n",
            0o644,
        );
        write_mode(&root.join("rust/crates/stub.rs"), b"// stub\n", 0o644);

        let worker_bin = root.join("build/owned-native/tokspur-native");
        let server_bin = root.join("build/rust-fast-math/release/tokspur-server");
        compile_fake_macho(tmp.path(), &worker_bin);
        compile_fake_macho(tmp.path(), &server_bin);
        let cli_src = Path::new(env!("CARGO_BIN_EXE_tokspur"));
        let cli_dst = root.join("build/rust-fast-math/release/tokspur");
        fs::copy(cli_src, &cli_dst).unwrap();
        fs::set_permissions(&cli_dst, fs::Permissions::from_mode(0o755)).unwrap();

        let metallib = root.join("build/owned-native/tokspur-owned.metallib");
        write_mode(&metallib, b"MTLB\x01\x00\x02\x00synthetic-metallib", 0o644);

        let mut this = Self {
            _tmp: tmp,
            root,
            target_dir,
            draft_dir,
        };
        this.refresh_build_manifest();
        this
    }

    fn refresh_build_manifest(&mut self) {
        let mut rel_artifacts = vec![
            ("build/owned-native/tokspur-native", false),
            ("build/owned-native/tokspur-owned.metallib", true),
            ("build/rust-fast-math/release/tokspur-server", false),
            ("build/rust-fast-math/release/tokspur", false),
        ];
        if self
            .root
            .join("build/owned-native/scalar-channel-target")
            .is_file()
        {
            rel_artifacts.push(("build/owned-native/scalar-channel-target", false));
        }
        let mut artifacts_obj = Vec::new();
        let mut sys_deps_obj = Vec::new();
        for (rel, is_metallib) in rel_artifacts {
            let full = self.root.join(rel);
            let (st, sha, _m) = inspect_and_hash_nofollow(&full, 512 << 20, rel).unwrap();
            artifacts_obj.push((
                rel.to_owned(),
                Json::new_object([
                    ("sha256", Json::from(sha)),
                    ("size", Json::integer(st.size)),
                    ("mode", Json::from(format!("0o{:o}", st.permissions()))),
                ]),
            ));
            if !is_metallib {
                let deps = audit_macho_dependencies(&full).unwrap();
                sys_deps_obj.push((
                    rel.to_owned(),
                    Json::Array(deps.into_iter().map(Json::from).collect()),
                ));
            }
        }

        let mut prof_obj = Vec::new();
        for rel in [
            "profiles/flash-next-assets.json",
            "profiles/m5-max-128.json",
            "profiles/owned-target-experimental.json",
        ] {
            let (_st, sha, _m) =
                inspect_and_hash_nofollow(&self.root.join(rel), 16 << 20, rel).unwrap();
            prof_obj.push((rel.to_owned(), Json::from(sha)));
        }

        let native_sources = collect_native_sources(&self.root).unwrap();
        let rust_sources = collect_rust_sources(&self.root).unwrap();
        let map_to_json = |m: &BTreeMap<String, String>| {
            Json::new_object(m.iter().map(|(k, v)| (k.clone(), Json::from(v.clone()))))
        };

        let manifest = Json::new_object([
            ("schema", Json::integer(2)),
            ("native_source_scope", Json::from(NATIVE_SOURCE_SCOPE)),
            (
                "git_head",
                Json::from("45563e6000000000000000000000000000000000"),
            ),
            ("git_dirty", Json::Bool(false)),
            ("enable_metal", Json::Bool(true)),
            ("frozen", Json::Bool(true)),
            ("metallib_function_count", Json::integer(12)),
            ("target_triple", Json::from("aarch64-apple-darwin")),
            ("profile", Json::from("release")),
            ("rustflags", Json::from("")),
            ("clang_version", Json::from(clang_version_output())),
            ("metal_version", Json::from("Apple metal version 32023")),
            ("rustc_verbose", Json::from("rustc 1.99.1")),
            ("cargo_version", Json::from("cargo 1.99.1")),
            (
                "rust_toolchain_sha256",
                Json::from(rust_sources["rust/rust-toolchain.toml"].clone()),
            ),
            (
                "cargo_lock_sha256",
                Json::from(rust_sources["rust/Cargo.lock"].clone()),
            ),
            (
                "cargo_config_sha256",
                Json::from(rust_sources["rust/.cargo/config.toml"].clone()),
            ),
            (
                "native_config",
                Json::new_object([
                    ("NATIVE_CONFIG", Json::from("carrier")),
                    ("ENABLE_METAL", Json::integer(1)),
                ]),
            ),
            (
                "tracked_cflags",
                Json::Array(vec![
                    Json::from("carrier"),
                    Json::from(":"),
                    Json::from("-O3"),
                ]),
            ),
            ("tracked_metalflags", Json::Array(vec![Json::from("-O2")])),
            (
                "rust_targets",
                Json::new_object([
                    ("tokspur-server", Json::Array(vec![Json::from("fast-math")])),
                    ("tokspur", Json::Array(Vec::new())),
                ]),
            ),
            ("profile_digests", Json::new_object(prof_obj)),
            ("native_sources", map_to_json(&native_sources)),
            ("rust_sources", map_to_json(&rust_sources)),
            ("artifacts", Json::new_object(artifacts_obj)),
            ("system_dependencies", Json::new_object(sys_deps_obj)),
        ]);

        let receipt_dir = self.root.join(".tokspur/receipts");
        fs::create_dir_all(&receipt_dir).unwrap();
        fs::set_permissions(
            self.root.join(".tokspur"),
            fs::Permissions::from_mode(0o700),
        )
        .unwrap();
        fs::set_permissions(&receipt_dir, fs::Permissions::from_mode(0o700)).unwrap();
        write_mode(
            &receipt_dir.join("build-manifest.json"),
            &dump_json(&manifest),
            0o600,
        );
    }

    fn cli_bin(&self) -> PathBuf {
        self.root.join("build/rust-fast-math/release/tokspur")
    }
}

fn wait_for_pid_file(path: &Path, timeout: Duration) -> i32 {
    let deadline = Instant::now() + timeout;
    loop {
        if let Ok(text) = fs::read_to_string(path)
            && let Ok(pid) = text.trim().parse::<i32>()
            && pid > 0
        {
            return pid;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for pid file {}",
            path.display()
        );
        thread::sleep(Duration::from_millis(20));
    }
}

fn spawn_serve_with_pid_file(
    mut make_cmd: impl FnMut() -> Command,
    pid_file: &Path,
    timeout: Duration,
) -> (Child, i32) {
    let deadline = Instant::now() + timeout;
    let stderr_path = pid_file.with_extension("stderr");
    let mut last_stderr = String::new();
    loop {
        let _ignored = fs::remove_file(pid_file);
        let stderr_file = fs::File::create(&stderr_path).unwrap();
        let mut cmd = make_cmd();
        cmd.stderr(Stdio::from(stderr_file));
        let mut child = cmd.spawn().unwrap();
        loop {
            if let Ok(text) = fs::read_to_string(pid_file)
                && let Ok(pid) = text.trim().parse::<i32>()
                && pid > 0
            {
                return (child, pid);
            }
            if let Some(status) = child.try_wait().unwrap() {
                let stderr = fs::read_to_string(&stderr_path).unwrap_or_default();
                if !stderr.is_empty() {
                    last_stderr = stderr.clone();
                }
                if stderr.contains("Other inference workers found:") && Instant::now() < deadline {
                    thread::sleep(Duration::from_millis(150));
                    break;
                }
                panic!(
                    "tokspur serve exited early ({status:?}) before writing {}: {stderr}",
                    pid_file.display()
                );
            }
            if Instant::now() >= deadline {
                let cur_stderr = fs::read_to_string(&stderr_path).unwrap_or_default();
                let diag = if cur_stderr.is_empty() {
                    last_stderr
                } else {
                    cur_stderr
                };
                panic!(
                    "timed out waiting for pid file {} (stderr: {diag})",
                    pid_file.display()
                );
            }
            thread::sleep(Duration::from_millis(20));
        }
    }
}

fn output_serve_retrying_transient_workers(
    mut make_cmd: impl FnMut() -> Command,
    timeout: Duration,
) -> Output {
    let deadline = Instant::now() + timeout;
    loop {
        let out = make_cmd().output().unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr);
        if !out.status.success()
            && stderr.contains("Other inference workers found:")
            && Instant::now() < deadline
        {
            thread::sleep(Duration::from_millis(150));
            continue;
        }
        return out;
    }
}

fn pid_is_alive(pid: i32) -> bool {
    match kill(pid, 0) {
        Ok(()) => true,
        Err(err) => err.raw_os_error() != Some(ESRCH),
    }
}

fn wait_until_pid_gone(pid: i32, timeout: Duration) {
    let deadline = Instant::now() + timeout;
    while pid_is_alive(pid) {
        assert!(
            Instant::now() < deadline,
            "pid {pid} remained alive past deadline"
        );
        thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn adopted_assets_survive_device_renumbering_only_on_the_verified_volume() {
    let _serial = SERVE_TEST_MUTEX
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fixture = SyntheticCheckout::create("adoption-remount");
    let adopted = Command::new(fixture.cli_bin())
        .args(["adopt", "--target"])
        .arg(&fixture.target_dir)
        .arg("--draft")
        .arg(&fixture.draft_dir)
        .output()
        .unwrap();
    assert!(adopted.status.success());
    let path = fixture.root.join(".tokspur/adoption.json");
    let baseline = parse_json(&fs::read(&path).unwrap()).unwrap();
    let first = &baseline
        .find("target")
        .unwrap()
        .unwrap()
        .find("assets")
        .unwrap()
        .unwrap()
        .array()
        .unwrap()[0];
    let changed_device = first
        .find("device")
        .unwrap()
        .unwrap()
        .unsigned_integer()
        .unwrap()
        + 1;
    let edit = |input: &Json, key: &[u8], value: Option<Json>| {
        let mut output = input.clone();
        let Json::Object(root) = &mut output else {
            panic!("adoption object")
        };
        let Json::Object(target) = &mut root.iter_mut().find(|(k, _)| k == b"target").unwrap().1
        else {
            panic!("target object")
        };
        let Json::Array(assets) = &mut target.iter_mut().find(|(k, _)| k == b"assets").unwrap().1
        else {
            panic!("assets array")
        };
        let Json::Object(fields) = &mut assets[0] else {
            panic!("asset object")
        };
        fields.retain(|(k, _)| k != key);
        if let Some(value) = value {
            fields.push((key.to_vec(), value));
        }
        output
    };
    let check = |value: &Json| {
        write_mode(&path, &dump_json(value), 0o600);
        Command::new(fixture.cli_bin())
            .args(["adopt", "--check-only"])
            .output()
            .unwrap()
            .status
            .success()
    };
    let remounted = edit(&baseline, b"device", Some(Json::integer(changed_device)));
    assert!(
        check(&remounted),
        "device renumbering on the verified volume must remain usable"
    );
    assert_eq!(
        first
            .find("volume_uuid")
            .unwrap()
            .unwrap()
            .string()
            .unwrap()
            .len(),
        32
    );
    assert!(!check(&edit(
        &remounted,
        b"volume_uuid",
        Some(Json::from("0".repeat(32)))
    )));
    for key in ["inode", "size", "mtime_ns", "ctime_ns"] {
        let value = first
            .find(key)
            .unwrap()
            .unwrap()
            .unsigned_integer()
            .unwrap()
            + 1;
        assert!(
            !check(&edit(
                &remounted,
                key.as_bytes(),
                Some(Json::integer(value))
            )),
            "changed {key} must refuse"
        );
    }
    assert!(!check(&edit(
        &remounted,
        b"sha256",
        Some(Json::from("0".repeat(64)))
    )));
    for value in [
        parse_json(b"-1").unwrap(),
        Json::Bool(true),
        Json::integer(0x8000_0000),
    ] {
        assert!(!check(&edit(&remounted, b"device", Some(value))));
    }
    for value in [
        Json::Null,
        Json::integer(0),
        Json::from("bad"),
        Json::from("z".repeat(32)),
    ] {
        assert!(!check(&edit(&baseline, b"volume_uuid", Some(value))));
    }
    let legacy = edit(&baseline, b"volume_uuid", None);
    assert!(check(&legacy));
    assert!(!check(&edit(
        &legacy,
        b"device",
        Some(Json::integer(changed_device))
    )));
}

#[test]
fn checkout_doctor_plan_adopt_legacy_compat_and_configure_succeed() {
    let _serial = SERVE_TEST_MUTEX
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fixture = SyntheticCheckout::create("checkout-cmds");
    let unrelated_cwd = TempDir::new("unrelated-cwd");

    let doc_out = Command::new(fixture.cli_bin())
        .current_dir(unrelated_cwd.path())
        .arg("doctor")
        .output()
        .unwrap();
    assert!(
        doc_out.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&doc_out.stderr)
    );
    let doc = parse_json(&doc_out.stdout).unwrap();
    assert_eq!(
        doc.find("cli_implementation")
            .unwrap()
            .unwrap()
            .string()
            .unwrap(),
        b"rust"
    );
    assert_eq!(
        doc.find("layout").unwrap().unwrap().string().unwrap(),
        b"checkout"
    );

    let plan_out = Command::new(fixture.cli_bin())
        .current_dir(unrelated_cwd.path())
        .arg("plan")
        .output()
        .unwrap();
    assert!(plan_out.status.success());

    let adopt_out = Command::new(fixture.cli_bin())
        .current_dir(unrelated_cwd.path())
        .args(["adopt", "--target"])
        .arg(&fixture.target_dir)
        .arg("--draft")
        .arg(&fixture.draft_dir)
        .output()
        .unwrap();
    assert!(
        adopt_out.status.success(),
        "adopt stderr={}",
        String::from_utf8_lossy(&adopt_out.stderr)
    );

    let check_out = Command::new(fixture.cli_bin())
        .current_dir(unrelated_cwd.path())
        .args(["adopt", "--check-only"])
        .output()
        .unwrap();
    assert!(check_out.status.success());

    let port = free_loopback_port();
    let cfg_out = Command::new(fixture.cli_bin())
        .current_dir(unrelated_cwd.path())
        .args(["configure", "--port", &port.to_string()])
        .output()
        .unwrap();
    assert!(
        cfg_out.status.success(),
        "configure stderr={}",
        String::from_utf8_lossy(&cfg_out.stderr)
    );

    // Verify legacy_dflash2_draft 2-file adoption receipt compatibility
    let legacy_dir = fixture._tmp.path().join("weights/legacy-draft");
    fs::create_dir_all(&legacy_dir).unwrap();
    let cfg_bytes = b"{\"architectures\":[\"DFlash2\"]}\n";
    let st_bytes = b"synthetic-legacy-safetensors\n";
    fs::write(legacy_dir.join("config.json"), cfg_bytes).unwrap();
    fs::write(legacy_dir.join("model.safetensors"), st_bytes).unwrap();

    let adoption_path = fixture.root.join(".tokspur/adoption.json");
    let current_adoption = parse_json(&fs::read(&adoption_path).unwrap()).unwrap();
    let target_group = current_adoption.find("target").unwrap().unwrap().clone();

    let legacy_entry = |p: &Path, bytes: &[u8]| -> Json {
        let (st, _sha, _m) = inspect_and_hash_nofollow(p, 16 << 20, "legacy").unwrap();
        Json::new_object([
            ("path", Json::from(p.to_string_lossy().into_owned())),
            ("sha256", Json::from(sha256_bytes(bytes))),
            ("device", Json::integer(u64::try_from(st.dev).unwrap())),
            ("inode", Json::integer(st.ino)),
            ("size", Json::integer(st.size)),
            (
                "mtime_ns",
                Json::integer(u64::try_from(st.mtime_ns).unwrap()),
            ),
            (
                "ctime_ns",
                Json::integer(u64::try_from(st.ctime_ns).unwrap()),
            ),
        ])
    };
    let legacy_adoption = Json::new_object([
        ("schema", Json::integer(1)),
        ("verification", Json::from("full-sha256")),
        ("target", target_group),
        (
            "draft",
            Json::new_object([
                (
                    "directory",
                    Json::from(legacy_dir.to_string_lossy().into_owned()),
                ),
                (
                    "assets",
                    Json::Array(vec![
                        legacy_entry(&legacy_dir.join("config.json"), cfg_bytes),
                        legacy_entry(&legacy_dir.join("model.safetensors"), st_bytes),
                    ]),
                ),
            ]),
        ),
    ]);
    write_mode(&adoption_path, &dump_json(&legacy_adoption), 0o600);

    let legacy_check = Command::new(fixture.cli_bin())
        .current_dir(unrelated_cwd.path())
        .args(["adopt", "--check-only"])
        .output()
        .unwrap();
    assert!(
        legacy_check.status.success(),
        "legacy_dflash2_draft check failed: {}",
        String::from_utf8_lossy(&legacy_check.stderr)
    );
    let missing_mtp = Command::new(fixture.cli_bin())
        .args(["pack", "plan", "--format", "v3"])
        .output()
        .unwrap();
    assert!(!missing_mtp.status.success());
    assert!(
        String::from_utf8_lossy(&missing_mtp.stderr).contains("adopted draft missing MTP sidecar")
    );
}

#[test]
fn pack_format_controls_verified_native_dispatch_and_budget_failures() {
    let mut fixture = SyntheticCheckout::create("pack-format");
    let cli = fixture.cli_bin();
    let run = |args: &[&str]| Command::new(&cli).args(args).output().unwrap();
    let native = fixture.root.join("build/owned-native/tokspur-native");
    fs::copy(
        &native,
        fixture
            .root
            .join("build/owned-native/scalar-channel-target"),
    )
    .unwrap();
    fixture.refresh_build_manifest();

    // V2's static plan remains available before adoption; both spellings enter the same implementation.
    for args in [
        vec!["pack", "plan", "--json"],
        vec!["pack", "plan", "--format", "v2", "--json"],
    ] {
        let output = run(&args);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let value = parse_json(&output.stdout).unwrap();
        assert_eq!(
            value.find("recipe").unwrap().unwrap().string().unwrap(),
            b"tokspur-pack-v2"
        );
    }
    assert!(!run(&["pack", "plan", "--format", "v3"]).status.success());
    assert!(
        Command::new(&cli)
            .args(["adopt", "--target"])
            .arg(&fixture.target_dir)
            .arg("--draft")
            .arg(&fixture.draft_dir)
            .output()
            .unwrap()
            .status
            .success()
    );
    let before = fixture.root.join(".tokspur/derived");
    assert!(!before.exists());
    let default_plan = run(&["pack", "plan", "--format", "v3", "--check"]);
    assert!(
        default_plan.status.success(),
        "{}",
        String::from_utf8_lossy(&default_plan.stderr)
    );
    assert!(String::from_utf8_lossy(&default_plan.stdout).contains("RAM budget: PASS"));
    assert!(!before.exists());

    for name in ["disk-fail", "ram-fail"] {
        let dir = fixture.root.join(name);
        fs::create_dir(&dir).unwrap();
        for check in [false, true] {
            let mut cmd = Command::new(&cli);
            cmd.args([
                "pack",
                "plan",
                "--format",
                "v3",
                "--json",
                "--reserve-gb",
                "1.5",
                "--target-dir",
            ])
            .arg(&dir);
            if check {
                cmd.arg("--check");
            }
            let output = cmd.output().unwrap();
            assert_eq!(
                output.status.code(),
                Some(i32::from(check)),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let plan = parse_json(&output.stdout).unwrap();
            assert_eq!(
                plan.find("reserve_bytes")
                    .unwrap()
                    .unwrap()
                    .unsigned_integer()
                    .unwrap(),
                1_610_612_736
            );
        }
    }
    let options = [
        "pack",
        "build",
        "--format",
        "v3",
        "--chunk-mib",
        "2",
        "--reserve-gb",
        "1.5",
    ];
    let explicit = fixture.root.join("custom pack.gguf");
    for path in [None, Some(&explicit)] {
        let mut cmd = Command::new(&cli);
        cmd.args(options);
        if let Some(path) = path {
            cmd.arg("--output").arg(path);
        }
        let output = cmd.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let receipt = parse_json(&output.stdout).unwrap();
        assert_eq!(
            receipt.find("status").unwrap().unwrap().string().unwrap(),
            b"verified-existing"
        );
        let expected = path
            .cloned()
            .unwrap_or_else(|| before.join("tokspur-pack-v3.asa.gguf"));
        assert_eq!(
            receipt.find("output").unwrap().unwrap().string().unwrap(),
            expected.to_str().unwrap().as_bytes()
        );
    }
    let refusal = Command::new(&cli)
        .args(options)
        .arg("--output")
        .arg(fixture.root.join("native-refusal.gguf"))
        .output()
        .unwrap();
    assert!(!refusal.status.success());
    assert!(refusal.stdout.is_empty());
    assert!(String::from_utf8_lossy(&refusal.stderr).contains("native fixture refused output"));
    let pending = fixture.root.join("cancel-pending.gguf");
    let mut building = Command::new(&cli)
        .args(options)
        .arg("--output")
        .arg(&pending)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let timeout = Instant::now() + Duration::from_secs(10);
    while !pending.is_file() {
        assert!(building.try_wait().unwrap().is_none() && Instant::now() < timeout);
        thread::sleep(Duration::from_millis(10));
    }
    assert!(StateLock::acquire(&fixture.root.join(".tokspur")).is_err());
    kill(building.id() as i32, SIGTERM).unwrap();
    assert!(!building.wait().unwrap().success());
    assert!(!pending.exists());
    assert!(StateLock::acquire(&fixture.root.join(".tokspur")).is_ok());
    // The selected scalar worker is still valid; this auxiliary executable is not.
    write_mode(&native, b"modified pack tool", 0o755);
    let tampered = run(&["pack", "plan", "--format", "v3"]);
    assert!(!tampered.status.success());
    assert!(String::from_utf8_lossy(&tampered.stderr).contains("artifact changed"));
}

#[test]
fn cross_language_serve_lock_excludes_python_and_rust() {
    let _serial = SERVE_TEST_MUTEX
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fixture = SyntheticCheckout::create("cross-lock");
    let state_dir = fixture.root.join(".tokspur/state");
    fs::create_dir_all(&state_dir).unwrap();
    fs::set_permissions(
        fixture.root.join(".tokspur"),
        fs::Permissions::from_mode(0o700),
    )
    .unwrap();
    fs::set_permissions(&state_dir, fs::Permissions::from_mode(0o700)).unwrap();
    let lock_file = state_dir.join("serve.lock");

    // 1. Python holds `serve.lock` -> Rust CLI refuses with exact message.
    let mut py_holder = Command::new("/usr/bin/python3")
        .arg("-c")
        .arg(
            r#"import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
sys.stdout.write("LOCKED\n")
sys.stdout.flush()
sys.stdin.read()
"#,
        )
        .arg(&lock_file)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();

    let mut line = [0u8; 7];
    py_holder
        .stdout
        .as_mut()
        .unwrap()
        .read_exact(&mut line)
        .unwrap();
    assert_eq!(&line, b"LOCKED\n");

    let port = free_loopback_port();
    let rust_attempt = Command::new(fixture.cli_bin())
        .args(["configure", "--port", &port.to_string()])
        .output()
        .unwrap();
    assert_eq!(rust_attempt.status.code(), Some(1));
    assert!(
        String::from_utf8_lossy(&rust_attempt.stderr)
            .contains("Another tokspur operation owns this state"),
        "stderr={}",
        String::from_utf8_lossy(&rust_attempt.stderr)
    );

    drop(py_holder.stdin.take());
    let _ = py_holder.wait().unwrap();

    // 2. Rust `StateLock` holds `serve.lock` -> Python `flock(LOCK_EX | LOCK_NB)` fails with BlockingIOError.
    let guard = StateLock::acquire(&fixture.root.join(".tokspur")).unwrap();
    let py_probe = Command::new("/usr/bin/python3")
        .arg("-c")
        .arg(
            r#"import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit(42)
sys.exit(0)
"#,
        )
        .arg(&lock_file)
        .status()
        .unwrap();
    assert_eq!(py_probe.code(), Some(42));
    drop(guard);
}

#[test]
fn verify_build_refuses_unrecorded_scalar_worker_and_bad_metallib() {
    let _serial = SERVE_TEST_MUTEX
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut fixture = SyntheticCheckout::create("verify-build");

    let ok_out = Command::new(fixture.cli_bin())
        .arg("verify-build")
        .output()
        .unwrap();
    assert!(
        ok_out.status.success(),
        "verify-build failed: {}",
        String::from_utf8_lossy(&ok_out.stderr)
    );

    // Drop an unrecorded `scalar-channel-target` into `build/owned-native/`
    let scalar = fixture
        .root
        .join("build/owned-native/scalar-channel-target");
    fs::copy(
        fixture.root.join("build/owned-native/tokspur-native"),
        &scalar,
    )
    .unwrap();
    fs::set_permissions(&scalar, fs::Permissions::from_mode(0o755)).unwrap();

    let bad_scalar = Command::new(fixture.cli_bin())
        .arg("verify-build")
        .output()
        .unwrap();
    assert!(!bad_scalar.status.success());
    assert!(
        String::from_utf8_lossy(&bad_scalar.stderr)
            .contains("Built artifact changed or unrecorded: scalar-channel-target"),
        "stderr={}",
        String::from_utf8_lossy(&bad_scalar.stderr)
    );
    fs::remove_file(&scalar).unwrap();

    // Corrupt `.metallib` header so it lacks `MTLB`
    write_mode(
        &fixture
            .root
            .join("build/owned-native/tokspur-owned.metallib"),
        b"BAD!\x01\x00\x02\x00synthetic-metallib",
        0o644,
    );
    fixture.refresh_build_manifest();
    let bad_mtlb = Command::new(fixture.cli_bin())
        .arg("verify-build")
        .output()
        .unwrap();
    assert!(!bad_mtlb.status.success());
    assert!(
        String::from_utf8_lossy(&bad_mtlb.stderr)
            .contains("tokspur-owned.metallib lacks valid MTLB header magic"),
        "stderr={}",
        String::from_utf8_lossy(&bad_mtlb.stderr)
    );
}

/// Shared with `tests/test_native_build_inputs.py`, which builds the same tree and holds the Python collectors to the same inventory.
const NATIVE_BUILD_INPUT_CORPUS: &str =
    include_str!("../../../../tests/fixtures/native-build-inputs.txt");

fn native_build_input_corpus() -> Vec<(bool, &'static str)> {
    NATIVE_BUILD_INPUT_CORPUS
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let (verdict, path) = line.split_once(' ').unwrap();
            match verdict {
                "pinned" => (true, path),
                "unpinned" => (false, path),
                other => panic!("unknown verdict {other:?} in the build-input corpus"),
            }
        })
        .collect()
}

#[test]
fn native_build_input_rule_and_inventory_match_the_shared_corpus() {
    let cases = native_build_input_corpus();
    assert!(cases.iter().any(|(pinned, _)| *pinned));
    assert!(cases.iter().any(|(pinned, _)| !*pinned));

    let tmp = TempDir::new("native-corpus");
    let mut expected = BTreeMap::new();
    for (pinned, path) in &cases {
        let name = path.rsplit('/').next().unwrap();
        assert_eq!(is_native_build_input(name), *pinned, "{path}");
        let body = format!("{path}\n");
        write_mode(&tmp.path().join(path), body.as_bytes(), 0o644);
        if *pinned {
            expected.insert((*path).to_owned(), sha256_bytes(body.as_bytes()));
        }
    }
    assert_eq!(collect_native_sources(tmp.path()).unwrap(), expected);
}

#[test]
fn verify_build_pins_native_build_inputs_but_not_documentation() {
    let _serial = SERVE_TEST_MUTEX
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fixture = SyntheticCheckout::create("native-scope");
    let root = &fixture.root;
    let verify = || {
        Command::new(fixture.cli_bin())
            .arg("verify-build")
            .output()
            .unwrap()
    };
    let assert_accepted = |step: &str| {
        let out = verify();
        assert!(
            out.status.success(),
            "{step}: verify-build refused: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    };
    let assert_drift = |step: &str| {
        let out = verify();
        assert!(!out.status.success(), "{step}: verify-build accepted");
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("Source drift in native/"),
            "{step}: stderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
    };

    let manifest =
        parse_json(&fs::read(root.join(".tokspur/receipts/build-manifest.json")).unwrap()).unwrap();
    assert_eq!(
        manifest
            .find("native_source_scope")
            .unwrap()
            .unwrap()
            .string()
            .unwrap(),
        NATIVE_SOURCE_SCOPE.as_bytes()
    );
    let recorded = manifest.find("native_sources").unwrap().unwrap();
    assert!(recorded.find("native/Makefile").unwrap().is_some());
    assert!(
        recorded
            .find("native/data/UNICODE-LICENSE.txt")
            .unwrap()
            .is_some()
    );
    assert!(recorded.find("native/README.md").unwrap().is_none());
    assert_accepted("as recorded");

    // Notes edited, added in a new directory, replaced by a leftover and removed after the manifest was recorded.
    write_mode(&root.join("native/README.md"), b"# edited notes\n", 0o644);
    write_mode(
        &root.join("native/PROVENANCE.md"),
        b"# added notes\n",
        0o644,
    );
    write_mode(
        &root.join("native/notes/NOTES.md"),
        b"# nested notes\n",
        0o644,
    );
    write_mode(&root.join("native/Makefile.orig"), b"# leftover\n", 0o644);
    assert_accepted("documentation edited and added");
    fs::remove_file(root.join("native/README.md")).unwrap();
    assert_accepted("documentation removed");

    // Build inputs and licence text stay pinned.
    let makefile = root.join("native/Makefile");
    write_mode(&makefile, b"# edited\n", 0o644);
    assert_drift("Makefile edited");
    write_mode(&makefile, b"# synthetic\n", 0o644);
    assert_accepted("Makefile restored");
    let licence = root.join("native/data/UNICODE-LICENSE.txt");
    write_mode(&licence, b"edited licence\n", 0o644);
    assert_drift("licence edited");
    write_mode(&licence, b"synthetic licence\n", 0o644);
    assert_accepted("licence restored");
    let added = root.join("native/src/added.cpp");
    write_mode(&added, b"// added\n", 0o644);
    assert_drift("build input added");
    fs::remove_file(&added).unwrap();
    assert_accepted("build input removed again");
    // `native/Makefile` also marks the checkout root, so delete the licence text to show that a recorded build input going missing is drift.
    fs::remove_file(&licence).unwrap();
    assert_drift("build input removed");
}

#[test]
fn verify_build_refuses_a_manifest_from_before_the_native_source_scope() {
    let _serial = SERVE_TEST_MUTEX
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fixture = SyntheticCheckout::create("native-scope-old");
    let manifest_path = fixture.root.join(".tokspur/receipts/build-manifest.json");
    let recorded = fs::read_to_string(&manifest_path).unwrap();
    assert!(recorded.contains("\"native_source_scope\""));

    let attempts = [
        (
            "scope absent",
            recorded.replace("\"native_source_scope\"", "\"native_source_scope_unread\""),
        ),
        (
            "scope from another rule",
            recorded.replace(NATIVE_SOURCE_SCOPE, "everything-v0"),
        ),
    ];
    for (label, edited) in attempts {
        write_mode(&manifest_path, edited.as_bytes(), 0o600);
        let out = Command::new(fixture.cli_bin())
            .arg("verify-build")
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success(), "{label}: manifest accepted");
        assert!(stderr.contains("native_source_scope"), "{label}: {stderr}");
        assert!(stderr.contains(NATIVE_SOURCE_SCOPE), "{label}: {stderr}");
        assert!(
            stderr.contains("run scripts/bootstrap.py"),
            "{label}: {stderr}"
        );
        assert!(!stderr.contains("Source drift"), "{label}: {stderr}");
    }
}

#[test]
fn serve_subprocess_lifecycle_signals_escalation_receipt_failure_descendants_and_exit_code() {
    let _serial = SERVE_TEST_MUTEX
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fixture = SyntheticCheckout::create("serve-lifecycle");
    let adopt = Command::new(fixture.cli_bin())
        .args(["adopt", "--target"])
        .arg(&fixture.target_dir)
        .arg("--draft")
        .arg(&fixture.draft_dir)
        .status()
        .unwrap();
    assert!(adopt.success());

    let port = free_loopback_port();
    let cfg = Command::new(fixture.cli_bin())
        .args(["configure", "--port", &port.to_string()])
        .status()
        .unwrap();
    assert!(cfg.success());

    let process_receipt = fixture.root.join(".tokspur/state/process.json");

    // 1. Early SIGTERM & SIGINT cancellation -> exit 0, process.json removed, zero orphans
    for sig in [SIGTERM, SIGINT] {
        let server_pid_file = fixture._tmp.path().join(format!("srv-sig-{sig}.pid"));
        let (mut child, srv_pid) = spawn_serve_with_pid_file(
            || {
                let mut cmd = Command::new(fixture.cli_bin());
                cmd.args([
                    "serve",
                    "--raw-gguf",
                    "--drain-timeouts-ms",
                    "400,400,400,400",
                ])
                .env("FAKE_SERVER_MODE", "sleep_sigterm")
                .env("FAKE_SERVER_PID_FILE", &server_pid_file);
                cmd
            },
            &server_pid_file,
            Duration::from_secs(15),
        );
        // Wait briefly for `process.json` to appear
        let deadline = Instant::now() + Duration::from_secs(2);
        while !process_receipt.exists() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(15));
        }
        assert!(process_receipt.exists());

        kill(child.id() as i32, sig).unwrap();
        let status = child.wait().unwrap();
        assert_eq!(status.code(), Some(0));
        assert!(!process_receipt.exists());
        wait_until_pid_gone(srv_pid, Duration::from_secs(2));
    }

    // 2. Repeated signal burst during drain -> bounded clean exit 0
    {
        let server_pid_file = fixture._tmp.path().join("srv-burst.pid");
        let (mut child, srv_pid) = spawn_serve_with_pid_file(
            || {
                let mut cmd = Command::new(fixture.cli_bin());
                cmd.args([
                    "serve",
                    "--raw-gguf",
                    "--drain-timeouts-ms",
                    "400,400,400,400",
                ])
                .env("FAKE_SERVER_MODE", "sleep_sigterm")
                .env("FAKE_SERVER_PID_FILE", &server_pid_file);
                cmd
            },
            &server_pid_file,
            Duration::from_secs(15),
        );
        for sig in [SIGTERM, SIGINT, SIGTERM, SIGINT, SIGTERM] {
            let _ignored = kill(child.id() as i32, sig);
            thread::sleep(Duration::from_millis(5));
        }
        let status = child.wait().unwrap();
        assert_eq!(status.code(), Some(0));
        assert!(!process_receipt.exists());
        wait_until_pid_gone(srv_pid, Duration::from_secs(2));
    }

    // 3. Child ignoring SIGTERM -> escalated to SIGKILL -> exit 0, process.json removed
    {
        let server_pid_file = fixture._tmp.path().join("srv-ignore.pid");
        let (mut child, srv_pid) = spawn_serve_with_pid_file(
            || {
                let mut cmd = Command::new(fixture.cli_bin());
                cmd.args([
                    "serve",
                    "--raw-gguf",
                    "--drain-timeouts-ms",
                    "250,500,250,250",
                ])
                .env("FAKE_SERVER_MODE", "ignore_sigterm")
                .env("FAKE_SERVER_PID_FILE", &server_pid_file);
                cmd
            },
            &server_pid_file,
            Duration::from_secs(15),
        );
        kill(child.id() as i32, SIGTERM).unwrap();
        let status = child.wait().unwrap();
        assert_eq!(status.code(), Some(0));
        assert!(!process_receipt.exists());
        wait_until_pid_gone(srv_pid, Duration::from_secs(2));
    }

    // 4. Failed `process.json` write (pre-existing symlink) -> SupervisorGuard::drop kills spawned child
    {
        let server_pid_file = fixture._tmp.path().join("srv-bad-receipt.pid");
        symlink(fixture._tmp.path().join("nowhere"), &process_receipt).unwrap();
        let out = output_serve_retrying_transient_workers(
            || {
                let mut cmd = Command::new(fixture.cli_bin());
                cmd.args([
                    "serve",
                    "--raw-gguf",
                    "--drain-timeouts-ms",
                    "300,300,300,300",
                ])
                .env("FAKE_SERVER_MODE", "sleep_sigterm")
                .env("FAKE_SERVER_PID_FILE", &server_pid_file);
                cmd
            },
            Duration::from_secs(15),
        );
        assert_eq!(out.status.code(), Some(1));
        assert!(
            String::from_utf8_lossy(&out.stderr).contains("Refusing to overwrite a symlink"),
            "stderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
        if let Ok(pid_text) = fs::read_to_string(&server_pid_file)
            && let Ok(srv_pid) = pid_text.trim().parse::<i32>()
        {
            wait_until_pid_gone(srv_pid, Duration::from_secs(2));
        }
        let _ignored = fs::remove_file(&process_receipt);
    }

    // 5. Detached `setsid()` non-system descendant tracked and killed by `drain_owned`
    {
        let server_pid_file = fixture._tmp.path().join("srv-parent.pid");
        let desc_pid_file = fixture._tmp.path().join("srv-descendant.pid");
        let (mut child, srv_pid) = spawn_serve_with_pid_file(
            || {
                let mut cmd = Command::new(fixture.cli_bin());
                cmd.args([
                    "serve",
                    "--raw-gguf",
                    "--drain-timeouts-ms",
                    "300,300,300,300",
                ])
                .env("FAKE_SERVER_MODE", "spawn_detached")
                .env("FAKE_SERVER_PID_FILE", &server_pid_file)
                .env("FAKE_DESCENDANT_PID_FILE", &desc_pid_file);
                cmd
            },
            &server_pid_file,
            Duration::from_secs(15),
        );
        let desc_pid = wait_for_pid_file(&desc_pid_file, Duration::from_secs(5));
        // Let the supervisor's poll loop snapshot the live descendant
        thread::sleep(Duration::from_millis(260));
        kill(child.id() as i32, SIGTERM).unwrap();
        let status = child.wait().unwrap();
        assert_eq!(status.code(), Some(0));
        assert!(!process_receipt.exists());
        wait_until_pid_gone(srv_pid, Duration::from_secs(2));
        wait_until_pid_gone(desc_pid, Duration::from_secs(2));
    }

    // 6. Natural child exit code preserved (`7`)
    {
        let server_pid_file = fixture._tmp.path().join("srv-exit7.pid");
        let out = output_serve_retrying_transient_workers(
            || {
                let mut cmd = Command::new(fixture.cli_bin());
                cmd.args([
                    "serve",
                    "--raw-gguf",
                    "--drain-timeouts-ms",
                    "300,300,300,300",
                ])
                .env("FAKE_SERVER_MODE", "exit_code")
                .env("FAKE_EXIT_CODE", "7")
                .env("FAKE_SERVER_PID_FILE", &server_pid_file);
                cmd
            },
            Duration::from_secs(15),
        );
        assert_eq!(out.status.code(), Some(7));
        assert!(!process_receipt.exists());
    }

    // 7. `--drafter dflash2 --dflash2-dir <dir>` launches without requiring `blk.48` MTP sidecar
    {
        let dflash2_dir = fixture._tmp.path().join("weights/dflash2-draft");
        fs::create_dir_all(&dflash2_dir).unwrap();
        let cfg_bytes = b"{\"architectures\":[\"DFlash2\"]}\n";
        let st_bytes = b"synthetic-legacy-safetensors\n";
        fs::write(dflash2_dir.join("config.json"), cfg_bytes).unwrap();
        fs::write(dflash2_dir.join("model.safetensors"), st_bytes).unwrap();

        let adoption_path = fixture.root.join(".tokspur/adoption.json");
        let current_adoption = parse_json(&fs::read(&adoption_path).unwrap()).unwrap();
        let target_group = current_adoption.find("target").unwrap().unwrap().clone();
        let asset_entry = |p: &Path, bytes: &[u8]| -> Json {
            let (st, _sha, _m) = inspect_and_hash_nofollow(p, 16 << 20, "dflash2").unwrap();
            Json::new_object([
                ("path", Json::from(p.to_string_lossy().into_owned())),
                ("sha256", Json::from(sha256_bytes(bytes))),
                ("device", Json::integer(u64::try_from(st.dev).unwrap())),
                ("inode", Json::integer(st.ino)),
                ("size", Json::integer(st.size)),
                (
                    "mtime_ns",
                    Json::integer(u64::try_from(st.mtime_ns).unwrap()),
                ),
                (
                    "ctime_ns",
                    Json::integer(u64::try_from(st.ctime_ns).unwrap()),
                ),
            ])
        };
        let dflash2_adoption = Json::new_object([
            ("schema", Json::integer(1)),
            ("verification", Json::from("full-sha256")),
            ("target", target_group),
            (
                "draft",
                Json::new_object([
                    (
                        "directory",
                        Json::from(dflash2_dir.to_string_lossy().into_owned()),
                    ),
                    (
                        "assets",
                        Json::Array(vec![
                            asset_entry(&dflash2_dir.join("config.json"), cfg_bytes),
                            asset_entry(&dflash2_dir.join("model.safetensors"), st_bytes),
                        ]),
                    ),
                ]),
            ),
        ]);
        write_mode(&adoption_path, &dump_json(&dflash2_adoption), 0o600);

        let mtp_refused = output_serve_retrying_transient_workers(
            || {
                let mut cmd = Command::new(fixture.cli_bin());
                cmd.args(["serve", "--raw-gguf"]);
                cmd
            },
            Duration::from_secs(15),
        );
        assert_eq!(mtp_refused.status.code(), Some(1));
        assert!(
            String::from_utf8_lossy(&mtp_refused.stderr).contains("MTP sidecar"),
            "expected missing MTP sidecar refusal, got: {}",
            String::from_utf8_lossy(&mtp_refused.stderr)
        );

        let server_pid_file = fixture._tmp.path().join("srv-dflash2.pid");
        let dflash2_ok = output_serve_retrying_transient_workers(
            || {
                let mut cmd = Command::new(fixture.cli_bin());
                cmd.args([
                    "serve",
                    "--raw-gguf",
                    "--drafter",
                    "dflash2",
                    "--dflash2-dir",
                    dflash2_dir.to_str().unwrap(),
                    "--verify-nax",
                    "--prefill-dense",
                    "nax",
                    "--prefill-grouped",
                    "nax",
                    "--drain-timeouts-ms",
                    "300,300,300,300",
                ])
                .env("FAKE_SERVER_MODE", "exit_code")
                .env("FAKE_EXIT_CODE", "0")
                .env("FAKE_SERVER_PID_FILE", &server_pid_file);
                cmd
            },
            Duration::from_secs(15),
        );
        assert_eq!(
            dflash2_ok.status.code(),
            Some(0),
            "dflash2 serve failed: {}",
            String::from_utf8_lossy(&dflash2_ok.stderr)
        );
        assert!(!process_receipt.exists());
    }
}

#[test]
fn staged_layout_invoked_from_unrelated_working_directory() {
    let _serial = SERVE_TEST_MUTEX
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let fixture = SyntheticCheckout::create("staged-src");
    let stage_tmp = TempDir::new("staged-pkg");
    let pkg_dir = stage_tmp.path().join("pkg");
    let ext_cwd = stage_tmp.path().join("unrelated-cwd");
    let ext_state = stage_tmp.path().join("ext-state");
    fs::create_dir_all(&pkg_dir).unwrap();
    fs::set_permissions(&pkg_dir, fs::Permissions::from_mode(0o700)).unwrap();
    fs::create_dir_all(&ext_cwd).unwrap();
    fs::create_dir_all(&ext_state).unwrap();
    fs::set_permissions(&ext_state, fs::Permissions::from_mode(0o700)).unwrap();

    let copy_staged = |src: &Path, name: &str, mode: u32| {
        let dst = pkg_dir.join(name);
        fs::copy(src, &dst).unwrap();
        fs::set_permissions(&dst, fs::Permissions::from_mode(mode)).unwrap();
    };
    copy_staged(&fixture.cli_bin(), "tokspur", 0o700);
    copy_staged(
        &fixture
            .root
            .join("build/rust-fast-math/release/tokspur-server"),
        "tokspur-server",
        0o700,
    );
    copy_staged(
        &fixture.root.join("build/owned-native/tokspur-native"),
        "tokspur-native",
        0o700,
    );
    copy_staged(
        &fixture
            .root
            .join("build/owned-native/tokspur-owned.metallib"),
        "tokspur-owned.metallib",
        0o600,
    );
    copy_staged(
        &fixture.root.join("profiles/owned-target-experimental.json"),
        "owned-target-experimental.json",
        0o600,
    );
    copy_staged(
        &fixture.root.join("profiles/m5-max-128.json"),
        "m5-max-128.json",
        0o600,
    );
    copy_staged(
        &fixture.root.join("profiles/flash-next-assets.json"),
        "flash-next-assets.json",
        0o600,
    );

    let mut entries = Vec::new();
    let mut sys_deps = Vec::new();
    for (name, is_exec, is_metallib) in [
        ("tokspur", true, false),
        ("tokspur-server", true, false),
        ("tokspur-native", true, false),
        ("tokspur-owned.metallib", false, true),
        ("owned-target-experimental.json", false, false),
        ("m5-max-128.json", false, false),
        ("flash-next-assets.json", false, false),
    ] {
        let p = pkg_dir.join(name);
        let (st, sha, _m) = inspect_and_hash_nofollow(&p, 512 << 20, name).unwrap();
        entries.push((
            name.to_owned(),
            Json::new_object([
                ("sha256", Json::from(sha)),
                ("size", Json::integer(st.size)),
                ("mode", Json::from(format!("0{:o}", st.permissions()))),
            ]),
        ));
        if is_exec && !is_metallib {
            let deps = audit_macho_dependencies(&p).unwrap();
            sys_deps.push((
                name.to_owned(),
                Json::Array(deps.into_iter().map(Json::from).collect()),
            ));
        }
    }

    let prov = Json::new_object([
        ("schema", Json::integer(2)),
        ("control_plane", Json::from("rust")),
        ("entries", Json::new_object(entries)),
        ("system_dependencies", Json::new_object(sys_deps)),
    ]);
    let mut prov_file = fs::File::create(pkg_dir.join("provenance.json")).unwrap();
    prov_file.write_all(&dump_json(&prov)).unwrap();
    fs::set_permissions(
        pkg_dir.join("provenance.json"),
        fs::Permissions::from_mode(0o600),
    )
    .unwrap();

    let staged_cli = pkg_dir.join("tokspur");

    // 1. Missing `--state-root` / `TOKSPUR_STATE_ROOT` fails on stateful commands
    let missing_state = Command::new(&staged_cli)
        .current_dir(&ext_cwd)
        .env_remove("TOKSPUR_STATE_ROOT")
        .env_remove("TOKSPUR_ROOT")
        .args(["adopt", "--check-only"])
        .output()
        .unwrap();
    assert!(!missing_state.status.success());
    assert!(
        String::from_utf8_lossy(&missing_state.stderr).contains("TOKSPUR_STATE_ROOT"),
        "stderr={}",
        String::from_utf8_lossy(&missing_state.stderr)
    );

    // 2. Overlapping `--state-root` inside `pkg_dir` is refused
    let overlap_state = Command::new(&staged_cli)
        .current_dir(&ext_cwd)
        .arg("--state-root")
        .arg(pkg_dir.join("nested-state"))
        .args(["adopt", "--check-only"])
        .output()
        .unwrap();
    assert!(!overlap_state.status.success());
    assert!(
        String::from_utf8_lossy(&overlap_state.stderr)
            .contains("separate from the immutable package directory")
    );

    // 3. Undeclared file inside `pkg_dir` is refused by `verify-build`
    write_mode(&pkg_dir.join("stray.txt"), b"stray", 0o600);
    let stray_check = Command::new(&staged_cli)
        .current_dir(&ext_cwd)
        .env("TOKSPUR_STATE_ROOT", &ext_state)
        .arg("verify-build")
        .output()
        .unwrap();
    assert!(!stray_check.status.success());
    assert!(
        String::from_utf8_lossy(&stray_check.stderr)
            .contains("Staged package contains undeclared or missing files")
    );
    fs::remove_file(pkg_dir.join("stray.txt")).unwrap();

    // 4. `doctor`, `plan`, `verify-build`, `adopt`, `configure` and `serve` from `ext_cwd` all pass!
    let doc_out = Command::new(&staged_cli)
        .current_dir(&ext_cwd)
        .env("TOKSPUR_STATE_ROOT", &ext_state)
        .arg("doctor")
        .output()
        .unwrap();
    assert!(doc_out.status.success());
    let doc = parse_json(&doc_out.stdout).unwrap();
    assert_eq!(
        doc.find("layout").unwrap().unwrap().string().unwrap(),
        b"staged"
    );

    assert!(
        Command::new(&staged_cli)
            .current_dir(&ext_cwd)
            .env("TOKSPUR_STATE_ROOT", &ext_state)
            .arg("plan")
            .status()
            .unwrap()
            .success()
    );

    let vb_out = Command::new(&staged_cli)
        .current_dir(&ext_cwd)
        .env("TOKSPUR_STATE_ROOT", &ext_state)
        .arg("verify-build")
        .output()
        .unwrap();
    assert!(
        vb_out.status.success(),
        "staged verify-build stderr={}",
        String::from_utf8_lossy(&vb_out.stderr)
    );

    assert!(
        Command::new(&staged_cli)
            .current_dir(&ext_cwd)
            .env("TOKSPUR_STATE_ROOT", &ext_state)
            .args(["adopt", "--target"])
            .arg(&fixture.target_dir)
            .arg("--draft")
            .arg(&fixture.draft_dir)
            .status()
            .unwrap()
            .success()
    );

    let staged_pack = Command::new(&staged_cli)
        .current_dir(&ext_cwd)
        .env("TOKSPUR_STATE_ROOT", &ext_state)
        .args(["pack", "plan", "--format", "v3", "--json", "--check"])
        .output()
        .unwrap();
    assert!(
        staged_pack.status.success(),
        "{}",
        String::from_utf8_lossy(&staged_pack.stderr)
    );
    assert_eq!(
        parse_json(&staged_pack.stdout)
            .unwrap()
            .find("version")
            .unwrap()
            .unwrap()
            .unsigned_integer()
            .unwrap(),
        3
    );
    let staged_build = Command::new(&staged_cli)
        .current_dir(&ext_cwd)
        .env("TOKSPUR_STATE_ROOT", &ext_state)
        .args([
            "pack",
            "build",
            "--format",
            "v3",
            "--chunk-mib",
            "2",
            "--reserve-gb",
            "1.5",
        ])
        .output()
        .unwrap();
    assert!(
        staged_build.status.success(),
        "{}",
        String::from_utf8_lossy(&staged_build.stderr)
    );
    let check_pack_output = |output_dir: &Path, format: &str| {
        Command::new(&staged_cli)
            .current_dir(&ext_cwd)
            .env("TOKSPUR_STATE_ROOT", &ext_state)
            .args([
                "pack",
                "build",
                "--format",
                format,
                "--chunk-mib",
                "2",
                "--reserve-gb",
                "1.5",
                "--output",
            ])
            .arg(output_dir.join("custom.gguf"))
            .output()
            .unwrap()
    };
    assert!(check_pack_output(&ext_cwd, "v3").status.success());
    let future = stage_tmp.path().join("future-state");
    let future_plan = Command::new(&staged_cli)
        .arg("--state-root")
        .arg(&future)
        .arg("plan")
        .output()
        .unwrap();
    assert!(future_plan.status.success());
    assert!(!future.exists());
    let case_alias = stage_tmp.path().join("PKG");
    for internal in [&pkg_dir, &case_alias].into_iter().filter(|p| p.is_dir()) {
        for format in ["v2", "v3"] {
            let rejected = check_pack_output(internal, format);
            assert!(!rejected.status.success());
            assert!(
                String::from_utf8_lossy(&rejected.stderr).contains("immutable package directory")
            );
        }
    }
    if case_alias.is_dir() {
        let parent = stage_tmp.path();
        let parent_alias = parent.with_file_name(parent.file_name().unwrap().to_ascii_uppercase());
        for state in [&case_alias, &case_alias.join("nested-state"), &parent_alias] {
            let rejected = Command::new(&staged_cli)
                .arg("--state-root")
                .arg(state)
                .args(["pack", "build", "--format", "v3"])
                .output()
                .unwrap();
            assert!(!rejected.status.success());
            assert!(
                String::from_utf8_lossy(&rejected.stderr).contains("immutable package directory")
            );
            assert!(!state.join(".tokspur").exists());
        }
        assert!(!pkg_dir.join("nested-state").exists());
    }
    let link = stage_tmp.path().join("link-to-package");
    symlink(&pkg_dir, &link).unwrap();
    let rejected = check_pack_output(&link, "v3");
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("symlink"));
    let port = free_loopback_port();
    assert!(
        Command::new(&staged_cli)
            .current_dir(&ext_cwd)
            .env("TOKSPUR_STATE_ROOT", &ext_state)
            .args(["configure", "--port", &port.to_string()])
            .status()
            .unwrap()
            .success()
    );

    let staged_pid_file = stage_tmp.path().join("staged-srv.pid");
    let (mut srv, srv_pid) = spawn_serve_with_pid_file(
        || {
            let mut cmd = Command::new(&staged_cli);
            cmd.current_dir(&ext_cwd)
                .env("TOKSPUR_STATE_ROOT", &ext_state)
                .env("FAKE_SERVER_MODE", "sleep_sigterm")
                .env("FAKE_SERVER_PID_FILE", &staged_pid_file)
                .args([
                    "serve",
                    "--raw-gguf",
                    "--drain-timeouts-ms",
                    "300,300,300,300",
                ]);
            cmd
        },
        &staged_pid_file,
        Duration::from_secs(15),
    );
    kill(srv.id() as i32, SIGTERM).unwrap();
    assert_eq!(srv.wait().unwrap().code(), Some(0));
    wait_until_pid_gone(srv_pid, Duration::from_secs(2));
}
