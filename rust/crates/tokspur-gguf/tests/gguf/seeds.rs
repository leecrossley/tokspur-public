//! `test_gguf --write-fuzz-seeds DIR`: the generated fuzz corpus.

use tokspur_gguf::Gguf;

use crate::support::fuzz_seeds;

/// Where the ignored writer puts the corpus (the C++ takes it as an argument).
const SEEDS_DIRECTORY: &str = "TOKSPUR_GGUF_FUZZ_SEEDS";

#[test]
fn fuzz_seeds_are_valid_inputs() {
    for (i, seed) in fuzz_seeds().iter().enumerate() {
        let file = Gguf::from_bytes(seed).unwrap();
        let mut bytes = Vec::new();
        for tensor in file.tensors() {
            file.tensor_bytes(tensor, &mut bytes).unwrap();
        }
        assert!(!file.tensors().is_empty(), "seed-{i}");
    }
}

#[test]
#[ignore = "writes seed-0 to seed-3 into $TOKSPUR_GGUF_FUZZ_SEEDS"]
fn write_fuzz_seeds() {
    let directory = std::env::var_os(SEEDS_DIRECTORY)
        .unwrap_or_else(|| panic!("set {SEEDS_DIRECTORY} to the corpus directory"));
    let directory = std::path::PathBuf::from(directory);
    std::fs::create_dir_all(&directory).unwrap();
    for (i, seed) in fuzz_seeds().iter().enumerate() {
        std::fs::write(directory.join(format!("seed-{i}")), seed)
            .unwrap_or_else(|error| panic!("cannot write generated fuzz seed: {error}"));
    }
}
