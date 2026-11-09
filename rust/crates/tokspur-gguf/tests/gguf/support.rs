//! The C++ test's byte builders (`number`, `replace`, `string`, `integer`, `text`, `Key`, `Data`, `fixture`, `split`, `Temporary`), its fuzz seed corpus, the `fuzz_gguf` mutation sequence, and the hashes the differential summaries print.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use tokspur_gguf::GgufError;
use tokspur_testkit::Mt19937;

pub type Bytes = Vec<u8>;

/// `rejected`: the action must fail. Returns the message, which the C++ test does not check and the Rust port does.
pub fn refusal<T>(result: Result<T, GgufError>) -> String {
    match result {
        Ok(_) => panic!("malformed input was accepted"),
        Err(error) => error.to_string(),
    }
}

pub fn number(bytes: &mut Bytes, value: u64, width: usize) {
    for i in 0..width {
        bytes.push(((value >> (i * 8)) & 255) as u8);
    }
}

/// `replace`. Indexing panics where the C++ `at()` throws.
pub fn replace(bytes: &mut Bytes, offset: usize, value: u64, width: usize) {
    for i in 0..width {
        bytes[offset + i] = ((value >> (i * 8)) & 255) as u8;
    }
}

pub fn string(bytes: &mut Bytes, value: &[u8]) {
    number(bytes, value.len() as u64, 8);
    bytes.extend_from_slice(value);
}

pub fn integer(value: u64, width: usize) -> Bytes {
    let mut bytes = Bytes::new();
    number(&mut bytes, value, width);
    bytes
}

pub fn text(value: impl AsRef<[u8]>) -> Bytes {
    let mut bytes = Bytes::new();
    string(&mut bytes, value.as_ref());
    bytes
}

#[derive(Clone)]
pub struct Key {
    pub name: Bytes,
    pub r#type: u32,
    pub value: Bytes,
}

pub fn key(name: impl AsRef<[u8]>, r#type: u32, value: Bytes) -> Key {
    Key {
        name: name.as_ref().to_vec(),
        r#type,
        value,
    }
}

#[derive(Clone)]
pub struct Data {
    pub name: Bytes,
    pub dimensions: Vec<u64>,
    pub r#type: u32,
    pub offset: u64,
    /// Payload bytes the fixture provides, not the tensor's encoded size.
    pub bytes: u64,
}

impl Default for Data {
    fn default() -> Self {
        data("weight", &[4], 0, 0, 16)
    }
}

pub fn data(
    name: impl AsRef<[u8]>,
    dimensions: &[u64],
    r#type: u32,
    offset: u64,
    bytes: u64,
) -> Data {
    Data {
        name: name.as_ref().to_vec(),
        dimensions: dimensions.to_vec(),
        r#type,
        offset,
        bytes,
    }
}

/// `fixture` without its payload: the aligned header, and the payload size the full fixture appends. Lets large cases become sparse files.
pub fn fixture_header(keys: &[Key], tensors: &[Data], alignment: u64) -> (Bytes, u64) {
    let mut bytes = Bytes::new();
    number(&mut bytes, 0x4655_4747, 4);
    number(&mut bytes, 3, 4);
    number(&mut bytes, tensors.len() as u64, 8);
    number(&mut bytes, keys.len() as u64, 8);
    for key in keys {
        string(&mut bytes, &key.name);
        number(&mut bytes, u64::from(key.r#type), 4);
        bytes.extend_from_slice(&key.value);
    }
    let mut payload = 0;
    for tensor in tensors {
        string(&mut bytes, &tensor.name);
        number(&mut bytes, tensor.dimensions.len() as u64, 4);
        for &dimension in &tensor.dimensions {
            number(&mut bytes, dimension, 8);
        }
        number(&mut bytes, u64::from(tensor.r#type), 4);
        number(&mut bytes, tensor.offset, 8);
        payload = payload.max(tensor.offset + tensor.bytes);
    }
    let length = (bytes.len() as u64).div_ceil(alignment) * alignment;
    bytes.resize(length as usize, 0);
    (bytes, payload)
}

/// `fixture(keys, tensors, alignment)`.
pub fn fixture(keys: &[Key], tensors: &[Data], alignment: u64) -> Bytes {
    let (mut bytes, payload) = fixture_header(keys, tensors, alignment);
    bytes.resize(bytes.len() + payload as usize, 0);
    bytes
}

/// `fixture()`: one 4-element F32 tensor named `weight`.
pub fn default_fixture() -> Bytes {
    fixture(&[], &[Data::default()], 32)
}

/// `fixture(keys)`.
pub fn keys_fixture(keys: &[Key]) -> Bytes {
    fixture(keys, &[Data::default()], 32)
}

/// `split(part, count, total)`.
pub fn split(part: u64, count: u64, total: u64) -> Vec<Key> {
    vec![
        key("general.architecture", 8, text("test")),
        key("split.no", 2, integer(part, 2)),
        key("split.count", 2, integer(count, 2)),
        key("split.tensors.count", 5, integer(total, 4)),
    ]
}

/// `Temporary`: a fresh directory, removed with everything in it.
pub struct Temporary {
    pub root: PathBuf,
}

impl Temporary {
    pub fn new() -> Self {
        Self::under(&std::env::temp_dir())
    }

    pub fn under(parent: &Path) -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        loop {
            let name = format!(
                "tokspur-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            );
            let root = parent.join(name);
            match std::fs::create_dir(&root) {
                Ok(()) => return Self { root },
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
                Err(error) => panic!("cannot create temporary directory: {error}"),
            }
        }
    }

    pub fn write(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.root.join(name);
        if let Err(error) = std::fs::write(&path, bytes) {
            panic!("cannot write fixture: {error}");
        }
        path
    }
}

impl Drop for Temporary {
    fn drop(&mut self) {
        if let Err(error) = std::fs::remove_dir_all(&self.root)
            && !std::thread::panicking()
        {
            panic!("cannot remove {}: {error}", self.root.display());
        }
    }
}

/// The corpus `test_gguf --write-fuzz-seeds` writes, as `seed-0` to `seed-3`.
pub fn fuzz_seeds() -> [Bytes; 4] {
    [
        default_fixture(),
        keys_fixture(&[
            key("general.architecture", 8, text("test")),
            key("a", 7, integer(1, 1)),
        ]),
        fixture(&[], &[data("q4", &[256], 12, 0, 144)], 32),
        fixture(
            &split(0, 2, 2),
            &[data("a", &[4], 0, 0, 16), data("b", &[4], 0, 32, 16)],
            32,
        ),
    ]
}

/// The inputs `fuzz_gguf` (standalone) derives from `seeds`, in order: the same `std::mt19937` seed, draws and mutations.
pub fn mutations(seeds: &[Bytes], rounds: usize) -> Vec<Bytes> {
    let mut random = Mt19937::new(0x746f_6b73);
    let mut inputs = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        let mut input = seeds[random.next_u32() as usize % seeds.len()].clone();
        let count = 1 + random.next_u32() % 5;
        for _ in 0..count {
            let offset = if input.is_empty() {
                0
            } else {
                random.next_u32() as usize % input.len()
            };
            match random.next_u32() % 5 {
                0 => {
                    if !input.is_empty() {
                        input[offset] ^= 1u8 << (random.next_u32() % 8);
                    }
                }
                1 => {
                    if !input.is_empty() {
                        input[offset] = random.next_u32() as u8;
                    }
                }
                2 => input.truncate(offset),
                3 => input.insert(offset, random.next_u32() as u8),
                _ => {
                    for byte in input.iter_mut().skip(offset).take(8) {
                        *byte = 255;
                    }
                }
            }
        }
        inputs.push(input);
    }
    inputs
}

pub const FNV_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a, as the driver computes it.
pub fn fnv(hash: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(hash, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

pub fn fnv_number(hash: u64, value: u64) -> u64 {
    fnv(hash, &value.to_le_bytes())
}

/// FNV-1a over each string's eight-byte length and bytes.
pub fn strings_hash(values: &[&[u8]]) -> u64 {
    values.iter().fold(FNV_BASIS, |hash, value| {
        fnv(fnv_number(hash, value.len() as u64), value)
    })
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

const SHA256_ROUND: [u32; 64] = [
    0x428a_2f98,
    0x7137_4491,
    0xb5c0_fbcf,
    0xe9b5_dba5,
    0x3956_c25b,
    0x59f1_11f1,
    0x923f_82a4,
    0xab1c_5ed5,
    0xd807_aa98,
    0x1283_5b01,
    0x2431_85be,
    0x550c_7dc3,
    0x72be_5d74,
    0x80de_b1fe,
    0x9bdc_06a7,
    0xc19b_f174,
    0xe49b_69c1,
    0xefbe_4786,
    0x0fc1_9dc6,
    0x240c_a1cc,
    0x2de9_2c6f,
    0x4a74_84aa,
    0x5cb0_a9dc,
    0x76f9_88da,
    0x983e_5152,
    0xa831_c66d,
    0xb003_27c8,
    0xbf59_7fc7,
    0xc6e0_0bf3,
    0xd5a7_9147,
    0x06ca_6351,
    0x1429_2967,
    0x27b7_0a85,
    0x2e1b_2138,
    0x4d2c_6dfc,
    0x5338_0d13,
    0x650a_7354,
    0x766a_0abb,
    0x81c2_c92e,
    0x9272_2c85,
    0xa2bf_e8a1,
    0xa81a_664b,
    0xc24b_8b70,
    0xc76c_51a3,
    0xd192_e819,
    0xd699_0624,
    0xf40e_3585,
    0x106a_a070,
    0x19a4_c116,
    0x1e37_6c08,
    0x2748_774c,
    0x34b0_bcb5,
    0x391c_0cb3,
    0x4ed8_aa4a,
    0x5b9c_ca4f,
    0x682e_6ff3,
    0x748f_82ee,
    0x78a5_636f,
    0x84c8_7814,
    0x8cc7_0208,
    0x90be_fffa,
    0xa450_6ceb,
    0xbef9_a3f7,
    0xc671_78f2,
];

/// SHA-256 (FIPS 180-4), test-only, so real-shard fingerprints can be compared with the C++ `CC_SHA256` ones.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    let mut state: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];
    let mut message = data.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&(data.len() as u64).wrapping_mul(8).to_be_bytes());
    for block in message.as_chunks::<64>().0 {
        let mut schedule = [0u32; 64];
        for (word, bytes) in schedule.iter_mut().zip(block.as_chunks::<4>().0) {
            *word = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }
        for i in 16..64 {
            let low = schedule[i - 15];
            let high = schedule[i - 2];
            let s0 = low.rotate_right(7) ^ low.rotate_right(18) ^ (low >> 3);
            let s1 = high.rotate_right(17) ^ high.rotate_right(19) ^ (high >> 10);
            schedule[i] = schedule[i - 16]
                .wrapping_add(s0)
                .wrapping_add(schedule[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for (round, word) in SHA256_ROUND.iter().zip(schedule) {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choice = (e & f) ^ (!e & g);
            let t1 = h
                .wrapping_add(s1)
                .wrapping_add(choice)
                .wrapping_add(*round)
                .wrapping_add(word);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (value, add) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *value = value.wrapping_add(add);
        }
    }
    let mut digest = [0u8; 32];
    for (bytes, word) in digest.as_chunks_mut::<4>().0.iter_mut().zip(state) {
        bytes.copy_from_slice(&word.to_be_bytes());
    }
    digest
}

/// `string_fingerprint` from `native/src/text.mm`: SHA-256 over the count and each string's length (eight bytes, little-endian) and bytes.
pub fn string_fingerprint(values: &[&[u8]]) -> String {
    let mut message = (values.len() as u64).to_le_bytes().to_vec();
    for value in values {
        message.extend_from_slice(&(value.len() as u64).to_le_bytes());
        message.extend_from_slice(value);
    }
    hex(&sha256(&message))
}

#[test]
fn sha256_matches_the_standard_vectors() {
    assert_eq!(
        hex(&sha256(b"")),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        hex(&sha256(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(
        hex(&sha256(
            b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
        )),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}

#[test]
fn fixture_matches_the_cpp_layout() {
    let bytes = default_fixture();
    // 24-byte header, 38-byte tensor info, padding to 64, 16 payload bytes.
    assert_eq!(bytes.len(), 80);
    assert_eq!(&bytes[..4], b"GGUF");
    assert_eq!(&bytes[32..38], b"weight");
    let sizes: Vec<usize> = fuzz_seeds().iter().map(Vec::len).collect();
    assert_eq!(sizes, [80, 144, 208, 272]);
}
