//! Compiles the C++ oracle for the differential fuzz targets: cpp/oracle.cpp (the narrow `extern "C"` shim) with the unchanged native/src parsers it calls, using the flags of the differential scripts in rust/tools/diff.
//!
//! Under `--cfg fuzzing` (set by `native/rust.mk rust-fuzz`) the C++ is also built with libFuzzer's sanitizer-coverage instrumentation, so coverage of the C++ guides the fuzzer as coverage of the Rust does, and with UBSan in trap mode plus libc++'s extensive hardening, so undefined behaviour or an out-of-bounds access in the C++ oracle stops the run as a crash. Neither needs a sanitizer runtime. Nothing here reads from the network.

#![forbid(unsafe_code)]

use std::error::Error;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// The native/src parsers the shim calls. json.cpp and response.cpp use unicode.cpp's UTF-8 checks; gguf.cpp defines what gguf.hpp declares.
const NATIVE_SOURCES: [&str; 4] = ["json.cpp", "unicode.cpp", "gguf.cpp", "response.cpp"];
const NATIVE_HEADERS: [&str; 4] = ["json.hpp", "unicode.hpp", "gguf.hpp", "response.hpp"];

fn rerun(out: &mut impl Write, path: &Path) -> io::Result<()> {
    writeln!(out, "cargo::rerun-if-changed={}", path.display())
}

fn main() -> Result<(), Box<dyn Error>> {
    let manifest = PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR").ok_or("CARGO_MANIFEST_DIR is not set")?,
    );
    let native = manifest.join("../../../native");
    let shim = manifest.join("cpp/oracle.cpp");
    let mut out = io::stdout().lock();
    rerun(&mut out, Path::new("build.rs"))?;
    if std::env::var_os("CARGO_FEATURE_CPP_ORACLE").is_none() {
        return Ok(());
    }
    rerun(&mut out, &shim)?;
    rerun(&mut out, &native.join("data/unicode9.inc"))?;
    for header in NATIVE_HEADERS {
        rerun(&mut out, &native.join("include/tokspur").join(header))?;
    }

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++20")
        .opt_level(2)
        .warnings(true)
        .extra_warnings(true)
        .warnings_into_errors(true)
        .flag("-Wpedantic")
        .include(native.join("include"))
        .file(&shim);
    for source in NATIVE_SOURCES {
        let path = native.join("src").join(source);
        rerun(&mut out, &path)?;
        build.file(path);
    }
    if std::env::var_os("CARGO_CFG_FUZZING").is_some() {
        build
            .flag("-fsanitize-coverage=inline-8bit-counters,pc-table,trace-cmp")
            .flag("-fsanitize=undefined")
            .flag("-fsanitize-trap=undefined")
            .define("_LIBCPP_HARDENING_MODE", "_LIBCPP_HARDENING_MODE_EXTENSIVE");
    }
    out.flush()?;
    drop(out);
    build.try_compile("tokspur_fuzz_oracle")?;
    Ok(())
}
