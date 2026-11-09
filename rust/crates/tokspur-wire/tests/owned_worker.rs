//! Port of `native/tests/test_owned_worker.cpp`.
//!
//! As in C++, this binary is also the fake target: the worker launches a hard link to it named `tokspur-native`, and `main` dispatches on `argv[1]`. The C++ forks a controller for the parent-death cases; here the controller is this binary re-executed in `controller` mode, which owns the worker the same way.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::print_stdout,
    clippy::print_stderr
)]

use std::io::{BufRead, Read, Write};
use std::os::unix::fs::DirBuilderExt;
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use tokspur_json::{Json, parse_json};
use tokspur_sys::process;
use tokspur_wire::worker::{CallbackFailed, OwnedWorker, WorkerError, WorkerOptions};

static CHECKS: AtomicUsize = AtomicUsize::new(0);

fn check(value: bool, why: &str) {
    CHECKS.fetch_add(1, Ordering::Relaxed);
    assert!(value, "{why}");
}

fn refuses<T>(result: Result<T, WorkerError>) {
    CHECKS.fetch_add(1, Ordering::Relaxed);
    match result {
        Err(error) if error.is_refused() => {}
        Err(other) => panic!("expected WorkerError, got {other}"),
        Ok(_) => panic!("unsafe worker operation accepted"),
    }
}

fn cancellation<T>(result: Result<T, WorkerError>) {
    CHECKS.fetch_add(1, Ordering::Relaxed);
    match result {
        Err(WorkerError::Cancelled(_)) => {}
        Err(other) => panic!("expected WorkerCancelled, got {other}"),
        Ok(_) => panic!("cancelled worker request did not report cancellation"),
    }
}

fn pause_ms(milliseconds: u64) {
    std::thread::sleep(Duration::from_millis(milliseconds));
}

fn pause_forever() -> ! {
    loop {
        std::thread::sleep(Duration::from_secs(3600));
    }
}

fn write_out(bytes: &[u8]) {
    let mut out = std::io::stdout().lock();
    if out.write_all(bytes).and_then(|()| out.flush()).is_err() {
        std::process::exit(90);
    }
}

fn int(value: u64) -> Json {
    Json::integer(value)
}

fn broker_storage(value: Json) -> Json {
    let mut fields = value.object().unwrap().clone();
    for (name, n) in [
        ("checkpoint_slots", 2),
        ("checkpoint_combined_budget_bytes", 1 << 20),
        ("checkpoint_host_bytes", 512),
        ("checkpoint_store_host_bytes", 256),
        ("checkpoint_gpu_bytes", 1024),
    ] {
        fields.push((name.as_bytes().to_vec(), int(n)));
    }
    Json::Object(fields)
}

fn ready_fixture(execution: &str, broker: bool) -> Json {
    let mut ready = vec![
        (b"event".to_vec(), Json::from("ready")),
        (b"pid".to_vec(), int(u64::from(std::process::id()))),
        (b"guardian_pid".to_vec(), int(process::parent_pid() as u64)),
        (b"group".to_vec(), int(process::group_of(0) as u64)),
        (b"session".to_vec(), int(process::session_of(0) as u64)),
        (b"small_gemv_column_splits".to_vec(), int(16)),
        (b"block_gemv_formats".to_vec(), int(3)),
        (b"stack_execution".to_vec(), Json::from(execution)),
        (b"maximum_generated_tokens".to_vec(), int(4096)),
    ];
    if execution == "recurrent-replay" {
        ready.push((b"replay_prepared".to_vec(), Json::Bool(true)));
        ready.push((b"recurrent_replay_captures".to_vec(), int(24)));
        ready.push((b"mapped_weight_region_limit_bytes".to_vec(), int(4 << 30)));
    }
    if broker {
        ready.push((b"checkpoint_broker".to_vec(), Json::Bool(true)));
        ready.push((b"broker_epoch".to_vec(), int(77)));
        return broker_storage(Json::Object(ready));
    }
    Json::Object(ready)
}

fn emit(value: &Json, fragment: bool) {
    let mut line = value.dump().unwrap();
    line.push(b'\n');
    if !fragment {
        write_out(&line);
    } else {
        for piece in line.chunks(3) {
            write_out(piece);
            pause_ms(1);
        }
    }
}

fn append_line(path: &Path, text: &str) {
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap();
    file.write_all(text.as_bytes()).unwrap();
}

fn fake_worker(args: &[String]) -> i32 {
    let mut policy = PathBuf::new();
    let mut execution = String::new();
    let mut selections = 0;
    let mut broker = false;
    let mut i = 2;
    while i + 1 < args.len() {
        if args[i] == "--policy" {
            policy = PathBuf::from(&args[i + 1]);
        }
        if args[i] == "--stack-execution" {
            execution = args[i + 1].clone();
            selections += 1;
        }
        if args[i] == "--worker-broker" {
            broker = args[i + 1] == "true";
        }
        i += 1;
    }
    if policy.as_os_str().is_empty()
        || selections != 1
        || (execution != "direct" && execution != "recurrent-replay")
    {
        return 91;
    }
    let mode = policy.file_name().unwrap().to_string_lossy().into_owned();
    let fragment = mode == "fragment";
    let directory = policy.parent().unwrap().to_path_buf();
    std::fs::write(
        directory.join("worker.pid"),
        format!("{}\n{}\n", std::process::id(), process::parent_pid()),
    )
    .unwrap();
    match mode.as_str() {
        "stall-ready" => pause_forever(),
        "eof-ready" => return 0,
        "bad-ready" => {
            write_out(b"{invalid}\n");
            return 0;
        }
        "huge-ready" => {
            write_out(&vec![b'x'; 1024 * 1024 + 1]);
            pause_forever();
        }
        _ => {}
    }
    let mut ready = ready_fixture(&execution, broker).object().unwrap().clone();
    for (flag, field) in [
        ("--native-host-logit-views", "native_host_logit_views"),
        (
            "--native-dflash2-defer-context",
            "native_dflash2_defer_context",
        ),
    ] {
        if let Some(pair) = args.windows(2).find(|pair| pair[0] == flag) {
            if mode == "sync-missing" {
                continue;
            }
            let enabled = pair[1] == "on";
            let value = match mode.as_str() {
                "sync-mismatch" => Json::Bool(!enabled),
                "sync-malformed" => Json::integer(1),
                _ => Json::Bool(enabled),
            };
            ready.push((field.as_bytes().to_vec(), value));
        }
    }
    emit(&Json::Object(ready), fragment);
    let mut prefix: u64 = 0;
    let mut seed: u64 = 0;
    let mut sampling = Json::new_object([
        ("temperature", int(0)),
        ("top_k", int(0)),
        ("top_p", int(1)),
        ("min_p", int(0)),
    ]);
    let stdin = std::io::stdin();
    let mut input = stdin.lock();
    loop {
        let mut line = Vec::new();
        if input.read_until(b'\n', &mut line).unwrap() == 0 {
            break;
        }
        if line.last() == Some(&b'\n') {
            line.pop();
        }
        append_line(&directory.join("commands"), "1\n");
        match mode.as_str() {
            "eof-command" => return 0,
            "partial-eof" => {
                write_out(b"{\"event\":\"done\"");
                return 0;
            }
            "bad-command" => {
                write_out(b"{invalid}\n");
                continue;
            }
            _ => {}
        }
        if mode == "stderr-stall" {
            for _ in 0..64 {
                if std::io::stderr().write_all(&[b'e'; 4096]).is_err() {
                    std::process::exit(90);
                }
            }
        }
        if mode == "stall" || mode == "stderr-stall" {
            pause_forever();
        }
        let body = parse_json(&line).unwrap();
        let sampled = |sampling: &Json, prefix: u64, seed: u64| {
            let mut fields = sampling.object().unwrap().clone();
            fields.push((b"valid".to_vec(), Json::Bool(true)));
            fields.push((b"entropy_prefix".to_vec(), int(prefix)));
            fields.push((b"seed".to_vec(), int(seed)));
            Json::Object(fields)
        };
        let stamp = |value: Json| {
            if !broker {
                return value;
            }
            let mut fields = value.object().unwrap().clone();
            for name in ["broker_epoch", "command_id", "lease_id"] {
                let mut n = body
                    .find(name)
                    .unwrap()
                    .unwrap()
                    .unsigned_integer()
                    .unwrap();
                if (mode == "broker-stale-epoch" && name == "broker_epoch")
                    || (mode == "broker-stale-lease" && name == "lease_id")
                {
                    n += 1;
                }
                fields.push((name.as_bytes().to_vec(), int(n)));
            }
            broker_storage(Json::Object(fields))
        };
        if let Some(op) = body.find("op").unwrap() {
            let begin = op.string().unwrap() == b"branch_begin";
            if begin {
                prefix = 0;
                sampling = body.find("sampling").unwrap().unwrap().clone();
                seed = body
                    .find("seed")
                    .unwrap()
                    .unwrap()
                    .unsigned_integer()
                    .unwrap();
            }
            if mode == "broker-slow-control" {
                pause_ms(80);
            }
            if mode == "broker-control-token" {
                emit(
                    &stamp(Json::new_object([
                        ("event", Json::from("token")),
                        ("id", int(42)),
                        ("committed_prefix", int(prefix + 1)),
                    ])),
                    false,
                );
            }
            emit(
                &stamp(Json::new_object([
                    ("event", Json::from("control_done")),
                    ("op", op.clone()),
                    (
                        "outcome",
                        Json::from(if begin { "cold_miss" } else { "captured" }),
                    ),
                    ("model_steps", int(0)),
                    ("generated_ids", Json::Array(Vec::new())),
                    ("common_prefix", int(prefix)),
                    ("committed_prefix", int(prefix)),
                    ("sampling", sampled(&sampling, prefix, seed)),
                    ("stop_id", Json::Null),
                    ("logits_available", Json::Bool(prefix != 0)),
                    ("continuation_allowed", Json::Bool(prefix != 0)),
                    ("checkpoint_retained", int(if begin { 0 } else { 1 })),
                ])),
                false,
            );
            continue;
        }
        if body.find("reset").unwrap().unwrap().boolean().unwrap() {
            prefix = 0;
        }
        if broker && body.find("sampling").unwrap().is_some() {
            sampling = body.find("sampling").unwrap().unwrap().clone();
            seed = body
                .find("seed")
                .unwrap()
                .unwrap()
                .unsigned_integer()
                .unwrap();
        }
        prefix += body.find("tokens").unwrap().unwrap().array().unwrap().len() as u64;
        let mut generated = Vec::new();
        if body
            .find("generate")
            .unwrap()
            .unwrap()
            .unsigned_integer()
            .unwrap()
            != 0
        {
            prefix += 1;
            generated.push(int(42));
            emit(
                &stamp(Json::new_object([
                    ("event", Json::from("token")),
                    ("id", int(42)),
                    ("committed_prefix", int(prefix)),
                    ("text", Json::from("café 🦉")),
                ])),
                fragment,
            );
            if mode == "slow-terminal" {
                pause_ms(80);
            }
        }
        let mut fields = vec![
            (b"event".to_vec(), Json::from("done")),
            (b"generated_ids".to_vec(), Json::Array(generated)),
            (b"committed_prefix".to_vec(), int(prefix)),
            (b"stack_execution".to_vec(), Json::from(execution.as_str())),
            (
                b"sampling".to_vec(),
                if broker {
                    sampled(&sampling, prefix, seed)
                } else {
                    Json::new_object([("valid", Json::Bool(true)), ("entropy_prefix", int(prefix))])
                },
            ),
        ];
        if broker {
            fields.push((b"checkpoint_retained".to_vec(), int(0)));
        }
        let done = stamp(Json::Object(fields));
        if mode == "extra" {
            let mut both = done.dump().unwrap();
            both.push(b'\n');
            both.extend(ready_fixture(&execution, false).dump().unwrap());
            both.push(b'\n');
            write_out(&both);
        } else {
            emit(&done, fragment);
        }
    }
    0
}

struct Fixture {
    directory: PathBuf,
    mode: String,
}

static FIXTURES: AtomicUsize = AtomicUsize::new(0);

impl Fixture {
    fn new(executable: &Path, mode: &str) -> Self {
        let directory = std::env::temp_dir().join(format!(
            "tokspur-owned-worker-rs-{}-{}",
            std::process::id(),
            FIXTURES.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::DirBuilder::new()
            .mode(0o700)
            .create(&directory)
            .unwrap();
        let target = directory.join("tokspur-native");
        if std::fs::hard_link(executable, &target).is_err() {
            std::fs::copy(executable, &target).unwrap();
        }
        for name in [mode, "owned.metallib"] {
            std::fs::write(directory.join(name), "owned CPU fixture\n").unwrap();
        }
        for i in 0..4 {
            std::fs::write(directory.join(format!("shard-{i}")), i.to_string()).unwrap();
        }
        Self {
            directory,
            mode: mode.to_owned(),
        }
    }

    fn reopen(directory: &Path, mode: &str) -> Self {
        Self {
            directory: directory.to_path_buf(),
            mode: mode.to_owned(),
        }
    }

    fn options(&self) -> WorkerOptions {
        WorkerOptions {
            executable: self.directory.join("tokspur-native"),
            policy: self.directory.join(&self.mode),
            library: self.directory.join("owned.metallib"),
            shards: (0..4)
                .map(|i| self.directory.join(format!("shard-{i}")))
                .collect(),
            startup_timeout: Duration::from_secs(8),
            step_timeout: Duration::from_secs(3),
            drain_timeout: Duration::from_millis(300),
            ..WorkerOptions::default()
        }
    }

    fn commands(&self) -> usize {
        std::fs::read_to_string(self.directory.join("commands"))
            .map_or(0, |text| text.lines().count())
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        if std::fs::remove_dir_all(&self.directory).is_err() {
            // Best-effort cleanup of a temporary fixture, as in C++.
        }
    }
}

fn request(reset: bool, generate: bool) -> Json {
    Json::new_object([
        (
            "tokens",
            Json::Array(if generate { Vec::new() } else { vec![int(1)] }),
        ),
        ("generate", int(u64::from(generate))),
        ("reset", Json::Bool(reset)),
    ])
}

fn soon() -> Instant {
    Instant::now() + Duration::from_secs(3)
}

fn never() -> impl FnMut() -> Result<bool, CallbackFailed> {
    || Ok(false)
}

fn ordinary(executable: &Path, mode: &str, execution: &str) {
    let fixture = Fixture::new(executable, mode);
    let mut options = fixture.options();
    options.stack_execution = execution.to_owned();
    let mut worker = OwnedWorker::new(options).unwrap();
    let ready = worker.ready().clone();
    check(
        ready
            .find("stack_execution")
            .unwrap()
            .unwrap()
            .string()
            .unwrap()
            == execution.as_bytes(),
        "fixed startup execution choice was not passed to target argv",
    );
    let field = |name: &str| {
        ready
            .find(name)
            .unwrap()
            .unwrap()
            .unsigned_integer()
            .unwrap()
    };
    check(
        field("pid") == field("group"),
        "target did not own its process group before ready",
    );
    check(
        field("guardian_pid") == field("session"),
        "guardian was not an armed isolated session before target ready",
    );
    check(worker.alive(), "ready worker not available");
    let one = worker
        .request(&request(true, false), soon(), Some(&mut never()), None)
        .unwrap();
    check(
        one.find("stack_execution")
            .unwrap()
            .unwrap()
            .string()
            .unwrap()
            == execution.as_bytes(),
        "persistent child changed execution choice",
    );
    check(
        one.find("committed_prefix")
            .unwrap()
            .unwrap()
            .unsigned_integer()
            .unwrap()
            == 1,
        "persistent input prefix differs",
    );
    let mut tokens = 0;
    let mut text_ok = true;
    let two = worker
        .request(
            &request(false, true),
            soon(),
            Some(&mut never()),
            Some(&mut |token: &Json| {
                tokens += 1;
                text_ok &=
                    token.find("text").unwrap().unwrap().string().unwrap() == "café 🦉".as_bytes();
                Ok(())
            }),
        )
        .unwrap();
    check(text_ok, "fragmented UTF-8 worker token changed");
    check(
        tokens == 1
            && two
                .find("committed_prefix")
                .unwrap()
                .unwrap()
                .unsigned_integer()
                .unwrap()
                == 2
            && fixture.commands() == 2,
        "persistent worker command/token accounting differs",
    );
    worker.close().unwrap();
    check(!worker.alive(), "closed worker remained admissible");
    worker.close().unwrap();
}

fn refusal_cases(executable: &Path) {
    for mode in ["eof-command", "partial-eof", "bad-command", "extra"] {
        let fixture = Fixture::new(executable, mode);
        let mut worker = OwnedWorker::new(fixture.options()).unwrap();
        refuses(worker.request(&request(true, false), soon(), Some(&mut never()), None));
        check(
            !worker.alive() && fixture.commands() == 1,
            "protocol failure retained admission or replayed a command",
        );
        refuses(worker.request(&request(true, false), soon(), Some(&mut never()), None));
        check(
            fixture.commands() == 1,
            "invalidated worker implicitly retried",
        );
        worker.close().unwrap();
    }
    for mode in ["bad-ready", "eof-ready", "huge-ready"] {
        let fixture = Fixture::new(executable, mode);
        refuses(OwnedWorker::new(fixture.options()));
        check(
            fixture.commands() == 0,
            "unsafe startup submitted inference commands",
        );
    }
    for mode in ["stall", "stderr-stall"] {
        let fixture = Fixture::new(executable, mode);
        let mut worker = OwnedWorker::new(fixture.options()).unwrap();
        let started = Instant::now();
        refuses(worker.request(
            &request(true, false),
            Instant::now() + Duration::from_millis(80),
            Some(&mut never()),
            None,
        ));
        check(
            started.elapsed() < Duration::from_secs(5)
                && !worker.alive()
                && fixture.commands() == 1,
            "stalled worker exceeded cleanup bound or retried",
        );
        worker.close().unwrap();
    }
}

fn native_sync_selector_cases(executable: &Path) {
    for host_view in [true, false] {
        for enabled in [true, false] {
            let fixture = Fixture::new(executable, "normal");
            let mut options = fixture.options();
            if host_view {
                options.host_logit_views = Some(enabled);
            } else {
                options.dflash2_defer_context = Some(enabled);
            }
            let worker = OwnedWorker::new(options).unwrap();
            let field = if host_view {
                "native_host_logit_views"
            } else {
                "native_dflash2_defer_context"
            };
            check(
                worker
                    .ready()
                    .find(field)
                    .unwrap()
                    .unwrap()
                    .boolean()
                    .unwrap()
                    == enabled,
                "selected synchronisation option did not reach worker argv",
            );
            drop(worker);
            for mode in ["sync-missing", "sync-mismatch", "sync-malformed"] {
                let fixture = Fixture::new(executable, mode);
                let mut options = fixture.options();
                if host_view {
                    options.host_logit_views = Some(enabled);
                } else {
                    options.dflash2_defer_context = Some(enabled);
                }
                check(
                    OwnedWorker::new(options).is_err(),
                    "unconfirmed synchronisation selector was admitted",
                );
                check(
                    fixture.commands() == 0,
                    "unconfirmed selector issued a request",
                );
            }
        }
    }
}

fn cancellation_cases(executable: &Path) {
    let fixture = Fixture::new(executable, "slow-terminal");
    let mut worker = OwnedWorker::new(fixture.options()).unwrap();
    cancellation(worker.request(&request(true, false), soon(), Some(&mut || Ok(true)), None));
    check(
        fixture.commands() == 0 && worker.alive(),
        "pre-submission cancellation sent command bytes",
    );
    worker
        .request(&request(true, false), soon(), Some(&mut never()), None)
        .unwrap();
    let stopped = AtomicBool::new(false);
    cancellation(worker.request(
        &request(false, true),
        soon(),
        Some(&mut || Ok(stopped.load(Ordering::Relaxed))),
        Some(&mut |_: &Json| {
            stopped.store(true, Ordering::Relaxed);
            Ok(())
        }),
    ));
    check(
        worker.alive() && fixture.commands() == 2,
        "cancelled single step did not drain or was replayed",
    );
    refuses(worker.request(&request(false, false), soon(), Some(&mut never()), None));
    check(
        fixture.commands() == 2,
        "cancelled context appended without explicit reset",
    );
    worker
        .request(&request(true, false), soon(), Some(&mut never()), None)
        .unwrap();
    refuses(worker.request(
        &request(false, true),
        soon(),
        Some(&mut never()),
        Some(&mut |_: &Json| Err(CallbackFailed)),
    ));
    check(
        !worker.alive() && fixture.commands() == 4,
        "failed token callback was hidden as a reusable cancellation",
    );
    worker.close().unwrap();
}

fn bounded_admission(executable: &Path) {
    let fixture = Fixture::new(executable, "normal");
    let mut worker = OwnedWorker::new(fixture.options()).unwrap();
    let two_inputs = Json::new_object([
        ("tokens", Json::Array(vec![int(1), int(2)])),
        ("generate", int(0)),
        ("reset", Json::Bool(true)),
    ]);
    refuses(worker.request(&two_inputs, soon(), Some(&mut never()), None));
    let input_and_output = Json::new_object([
        ("tokens", Json::Array(vec![int(1)])),
        ("generate", int(1)),
        ("reset", Json::Bool(true)),
    ]);
    refuses(worker.request(&input_and_output, soon(), Some(&mut never()), None));
    check(
        fixture.commands() == 0 && worker.alive(),
        "multi-step admission wrote bytes or damaged persistent child",
    );
    worker
        .request(&request(true, false), soon(), Some(&mut never()), None)
        .unwrap();
    worker.close().unwrap();
    let mut bad = fixture.options();
    bad.shards[3] = bad.shards[0].clone();
    refuses(OwnedWorker::new(bad));
    let mut bad = fixture.options();
    bad.executable = PathBuf::from("/bin/sh");
    refuses(OwnedWorker::new(bad));
    for execution in ["", "capture", "indirect-compatible", "unknown"] {
        let invalid_mode = Fixture::new(executable, "normal");
        let mut options = invalid_mode.options();
        options.stack_execution = execution.to_owned();
        refuses(OwnedWorker::new(options));
        check(
            !invalid_mode.directory.join("worker.pid").exists() && invalid_mode.commands() == 0,
            "unsupported execution choice launched a child or submitted work",
        );
    }
    let loading = Fixture::new(executable, "stall-ready");
    let started = Instant::now();
    let mut options = loading.options();
    options.startup_cancelled = Some(Box::new(move || {
        started.elapsed() > Duration::from_millis(80)
    }));
    refuses(OwnedWorker::new(options));
    check(
        started.elapsed() < Duration::from_secs(5) && loading.commands() == 0,
        "startup cancellation did not bound readiness/cleanup",
    );
}

/// `controller` mode: own one worker, report readiness, then wait to die.
fn controller(directory: &Path, mode: &str) -> i32 {
    let fixture = Fixture::reopen(directory, mode);
    let options = fixture.options();
    std::mem::forget(fixture); // The parent test owns the directory.
    match OwnedWorker::new(options) {
        Ok(worker) => {
            write_out(b"R");
            let _keep = worker;
            pause_forever();
        }
        Err(_) => 80,
    }
}

fn parent_death(executable: &Path, before_ready: bool) {
    let fixture = Fixture::new(
        executable,
        if before_ready {
            "stall-ready"
        } else {
            "normal"
        },
    );
    // C++ forks the controller over a pipe; here the controller is spawned with a piped stdout. The check points correspond one to one.
    let spawned = Command::new(std::env::current_exe().unwrap())
        .arg("controller")
        .arg(&fixture.directory)
        .arg(&fixture.mode)
        .stdout(Stdio::piped())
        .spawn();
    check(
        spawned.is_ok(),
        "cannot fork parent-death fixture controller",
    );
    let mut child = spawned.unwrap();
    let status_pipe = child.stdout.take();
    check(
        status_pipe.is_some(),
        "cannot create parent-death fixture channel",
    );
    let mut status_pipe = status_pipe.unwrap();
    let deadline = Instant::now() + Duration::from_secs(8);
    if !before_ready {
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut ready = [0u8; 1];
            let read = status_pipe.read(&mut ready).map(|n| (n, ready[0]));
            sender.send(read.ok()).ok();
        });
        let ready = receiver.recv_timeout(Duration::from_secs(8));
        check(ready.is_ok(), "controller did not reach ready");
        check(
            ready.ok().flatten() == Some((1, b'R')),
            "controller ready framing differs",
        );
    }
    let mut target: i32 = -1;
    while Instant::now() < deadline && target <= 1 {
        if let Ok(text) = std::fs::read_to_string(fixture.directory.join("worker.pid")) {
            target = text
                .lines()
                .next()
                .and_then(|line| line.parse().ok())
                .unwrap_or(-1);
        }
        if target <= 1 {
            pause_ms(10);
        }
    }
    check(target > 1, "worker never crossed the pre-exec armed gate");
    check(
        child.kill().is_ok(),
        "cannot terminate exact fixture controller",
    );
    let status = child.wait();
    check(
        status.is_ok_and(|status| status.signal().is_some()),
        "controller death was not verified",
    );
    let mut gone = false;
    for _ in 0..300 {
        if process::process_absent(target) {
            gone = true;
            break;
        }
        pause_ms(10);
    }
    check(
        gone,
        "armed guardian did not reap target after parent death",
    );
}

fn begin_control(restore: bool) -> Json {
    Json::new_object([
        ("op", Json::from("branch_begin")),
        ("prompt_ids", Json::Array(vec![int(1), int(2)])),
        ("generation_requested", Json::Bool(true)),
        ("allow_restore", Json::Bool(restore)),
        ("seed", int(7)),
        (
            "sampling",
            Json::new_object([
                ("temperature", int(0)),
                ("top_k", int(0)),
                ("top_p", int(1)),
                ("min_p", int(0)),
            ]),
        ),
    ])
}

fn prepared(fixture: &Fixture) -> WorkerOptions {
    let mut options = fixture.options();
    options.checkpoint_slots = 2;
    options.checkpoint_budget_bytes = 1 << 20;
    options
}

fn broker_cases(executable: &Path) {
    let capture = Json::new_object([("op", Json::from("branch_capture"))]);
    {
        let fixture = Fixture::new(executable, "broker-normal");
        let mut worker = OwnedWorker::new(prepared(&fixture)).unwrap();
        refuses(worker.request(&begin_control(true), soon(), Some(&mut never()), None));
        for body in [
            r#"{"tokens":[1,2],"generate":0,"reset":true}"#,
            r#"{"tokens":[1],"generate":1,"reset":true}"#,
            r#"{"op":"branch_capture","tokens":[1]}"#,
        ] {
            refuses(worker.request(
                &parse_json(body.as_bytes()).unwrap(),
                soon(),
                Some(&mut never()),
                None,
            ));
        }
        check(
            fixture.commands() == 0 && worker.alive(),
            "unleased reuse submitted control bytes",
        );
        let began = worker
            .request(&begin_control(false), soon(), Some(&mut never()), None)
            .unwrap();
        check(
            began.find("event").unwrap().unwrap().string().unwrap() == b"control_done"
                && began
                    .find("committed_prefix")
                    .unwrap()
                    .unwrap()
                    .unsigned_integer()
                    .unwrap()
                    == 0
                && began
                    .find("generated_ids")
                    .unwrap()
                    .unwrap()
                    .array()
                    .unwrap()
                    .is_empty(),
            "zero-step cold begin performed model work",
        );
        worker
            .request(&request(false, false), soon(), Some(&mut never()), None)
            .unwrap();
        worker
            .request(&capture, soon(), Some(&mut never()), None)
            .unwrap();
        let before = fixture.commands();
        cancellation(worker.request(&begin_control(false), soon(), Some(&mut || Ok(true)), None));
        refuses(worker.request(&capture, soon(), Some(&mut never()), None));
        refuses(worker.request(&begin_control(true), soon(), Some(&mut never()), None));
        check(
            fixture.commands() == before && worker.alive(),
            "pre-control cancellation bypassed cold-reset requirement",
        );
        worker
            .request(&begin_control(false), soon(), Some(&mut never()), None)
            .unwrap();
        worker.close().unwrap();
    }
    {
        let fixture = Fixture::new(executable, "broker-slow-control");
        let mut worker = OwnedWorker::new(prepared(&fixture)).unwrap();
        let start = Instant::now();
        cancellation(worker.request(
            &begin_control(false),
            soon(),
            Some(&mut || Ok(start.elapsed() > Duration::from_millis(10))),
            None,
        ));
        let count = fixture.commands();
        refuses(worker.request(&request(false, false), soon(), Some(&mut never()), None));
        check(
            fixture.commands() == count && worker.alive(),
            "drained control cancellation left an append lease reusable",
        );
        let reset = worker
            .request(&begin_control(false), soon(), Some(&mut never()), None)
            .unwrap();
        check(
            reset
                .find("lease_id")
                .unwrap()
                .unwrap()
                .unsigned_integer()
                .unwrap()
                == 2,
            "control recovery reused the cancelled lease",
        );
        worker.close().unwrap();
    }
    for mode in [
        "broker-stale-epoch",
        "broker-stale-lease",
        "broker-control-token",
    ] {
        let fixture = Fixture::new(executable, mode);
        let mut worker = OwnedWorker::new(prepared(&fixture)).unwrap();
        refuses(worker.request(&begin_control(false), soon(), Some(&mut never()), None));
        check(
            !worker.alive(),
            "ambiguous control reply retained worker admission",
        );
        worker.close().unwrap();
    }
}

fn mtp_abort_cases(executable: &Path) {
    let fixture = Fixture::new(executable, "normal");
    let mut worker = OwnedWorker::new(fixture.options()).unwrap();

    // Case 1: abort_mtp_generation on idle worker (no last_response) -> Ok(())
    worker.abort_mtp_generation(soon()).unwrap();
    check(
        worker.alive(),
        "idle abort_mtp_generation retains worker alive",
    );

    // Case 2: normal request
    let _first = worker
        .request(&request(true, false), soon(), Some(&mut never()), None)
        .unwrap();
    check(worker.alive(), "first request ok");

    // Case 3: abort_mtp_generation when last_response has no native_mtp_generation -> Ok(())
    worker.abort_mtp_generation(soon()).unwrap();
    check(worker.alive(), "no native_mtp_generation abort ok");

    // Case 4: direct submission of mtp_generation abort command in scalar mode
    let abort_job = Json::new_object([
        ("tokens", Json::Array(Vec::new())),
        ("generate", int(0)),
        ("reset", Json::Bool(false)),
        (
            "mtp_generation",
            Json::new_object([("id", int(42)), ("abort", Json::Bool(true))]),
        ),
    ]);
    let done = worker
        .request(&abort_job, soon(), Some(&mut never()), None)
        .unwrap();
    check(
        done.find("event").unwrap().unwrap().string().unwrap() == b"done",
        "abort command processed",
    );

    // Case 5: closed worker abort_mtp_generation is a no-op Ok(())
    worker.close().unwrap();
    worker.abort_mtp_generation(soon()).unwrap();

    // Case 6: MTP abort succeeds even when worker.needs_reset = true
    let fixture2 = Fixture::new(executable, "normal");
    let mut worker2 = OwnedWorker::new(fixture2.options()).unwrap();
    let _ = worker2
        .request(&request(true, false), soon(), Some(&mut never()), None)
        .unwrap();
    let cancel = std::sync::atomic::AtomicBool::new(true);
    let mut probe = || Ok(cancel.load(Ordering::Relaxed));
    let cancel_res = worker2.request(&request(false, false), soon(), Some(&mut probe), None);
    check(
        matches!(cancel_res, Err(WorkerError::Cancelled(_))),
        "cancelled before submission",
    );
    let ordinary_res = worker2.request(&request(false, false), soon(), Some(&mut never()), None);
    check(
        matches!(ordinary_res, Err(WorkerError::Refused(_))),
        "ordinary request refused when needs_reset",
    );
    let done_reset = worker2
        .request(&abort_job, soon(), Some(&mut never()), None)
        .unwrap();
    check(
        done_reset.find("event").unwrap().unwrap().string().unwrap() == b"done",
        "abort succeeds when needs_reset is true",
    );
    worker2.close().unwrap();

    // Case 7: Invalid MTP abort frames are refused
    let fixture3 = Fixture::new(executable, "normal");
    let mut worker3 = OwnedWorker::new(fixture3.options()).unwrap();
    let bad_abort_in = Json::new_object([
        ("tokens", Json::Array(vec![int(42)])),
        ("generate", int(0)),
        ("reset", Json::Bool(false)),
        (
            "mtp_generation",
            Json::new_object([("id", int(42)), ("abort", Json::Bool(true))]),
        ),
    ]);
    check(
        matches!(
            worker3.request(&bad_abort_in, soon(), Some(&mut never()), None),
            Err(WorkerError::Refused(_))
        ),
        "abort with inputs refused",
    );
    let bad_abort_gen = Json::new_object([
        ("tokens", Json::Array(Vec::new())),
        ("generate", int(1)),
        ("reset", Json::Bool(false)),
        (
            "mtp_generation",
            Json::new_object([("id", int(42)), ("abort", Json::Bool(true))]),
        ),
    ]);
    check(
        matches!(
            worker3.request(&bad_abort_gen, soon(), Some(&mut never()), None),
            Err(WorkerError::Refused(_))
        ),
        "abort with generate refused",
    );
    let bad_abort_rst = Json::new_object([
        ("tokens", Json::Array(Vec::new())),
        ("generate", int(0)),
        ("reset", Json::Bool(true)),
        (
            "mtp_generation",
            Json::new_object([("id", int(42)), ("abort", Json::Bool(true))]),
        ),
    ]);
    check(
        matches!(
            worker3.request(&bad_abort_rst, soon(), Some(&mut never()), None),
            Err(WorkerError::Refused(_))
        ),
        "abort with reset refused",
    );
    worker3.close().unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() > 1 && args[1] == "target" {
        std::process::exit(fake_worker(&args));
    }
    if args.len() > 3 && args[1] == "controller" {
        std::process::exit(controller(Path::new(&args[2]), &args[3]));
    }
    if args.iter().any(|arg| arg == "--list") {
        // `cargo test -- --list` probes harnessless tests; report one case.
        println!("owned_worker: test");
        return;
    }
    // TOKSPUR_FAKE_TARGET swaps in another fake target, e.g. the C++ `test-owned_worker` binary, to cross-check the wire with the same cases.
    let fake = std::env::var_os("TOKSPUR_FAKE_TARGET").unwrap_or_else(|| args[0].clone().into());
    let executable = std::fs::canonicalize(fake).unwrap();
    ordinary(&executable, "normal", "direct");
    ordinary(&executable, "fragment", "direct");
    ordinary(&executable, "normal", "recurrent-replay");
    ordinary(&executable, "fragment", "recurrent-replay");
    refusal_cases(&executable);
    native_sync_selector_cases(&std::env::current_exe().unwrap());
    cancellation_cases(&executable);
    bounded_admission(&executable);
    parent_death(&executable, false);
    parent_death(&executable, true);
    broker_cases(&executable);
    mtp_abort_cases(&executable);
    println!(
        "Owned worker CPU fixture checks passed: {}",
        CHECKS.load(Ordering::Relaxed)
    );
}
