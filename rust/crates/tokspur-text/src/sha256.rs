//! SHA-256 for asset pins and `string_fingerprint`, computed as `native/src/text.mm` computes it: with CommonCrypto's `CC_SHA256` family, wrapped in `tokspur_sys::digest`, which uses the SHA-256 instructions of Apple silicon.

pub use tokspur_sys::digest::{DIGEST_LENGTH, Sha256};

/// One-shot SHA-256, as `CC_SHA256`.
pub fn sha256(bytes: &[u8]) -> [u8; DIGEST_LENGTH] {
    let mut context = Sha256::new();
    context.update(bytes);
    context.finalize()
}

/// Lowercase hexadecimal, as `text.mm` spells digests and pins.
pub fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut result = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        result.push(char::from(DIGITS[usize::from(byte >> 4)]));
        result.push(char::from(DIGITS[usize::from(byte & 15)]));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(bytes: &[u8]) -> String {
        hex(&sha256(bytes))
    }

    #[test]
    fn nist_short_messages() {
        // FIPS 180-4 examples (csrc.nist.gov SHA256.pdf, SHA256_2.pdf).
        assert_eq!(
            digest(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            digest(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(
            digest(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        assert_eq!(
            digest(
                b"abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmno\
                  ijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"
            ),
            "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1"
        );
    }

    #[test]
    fn nist_one_million_a() {
        assert_eq!(
            digest(&vec![b'a'; 1_000_000]),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn padding_boundaries_around_one_and_two_blocks() {
        // Expected values from `shasum -a 256` and Python's hashlib.
        let repeated_a = [
            (
                55,
                "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318",
            ),
            (
                56,
                "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a",
            ),
            (
                63,
                "7d3e74a05d7db15bce4ad9ec0658ea98e3f06eeecf16b4c6fff2da457ddc2f34",
            ),
            (
                64,
                "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb",
            ),
            (
                65,
                "635361c48bb9eab14198e76ea8ab7f1a41685d6ad62aa9146d301d4f17eb0ae0",
            ),
            (
                119,
                "31eba51c313a5c08226adf18d4a359cfdfd8d2e816b13f4af952f7ea6584dcfb",
            ),
            (
                120,
                "2f3d335432c70b580af0e8e1b3674a7c020d683aa5f73aaaedfdc55af904c21c",
            ),
            (
                127,
                "c57e9278af78fa3cab38667bef4ce29d783787a2f731d4e12200270f0c32320a",
            ),
            (
                128,
                "6836cf13bac400e9105071cd6af47084dfacad4e5e302c94bfed24e013afb73e",
            ),
        ];
        for (length, expected) in repeated_a {
            assert_eq!(digest(&vec![b'a'; length]), expected, "{length} x 'a'");
        }
        let counting = [
            (
                55,
                "463eb28e72f82e0a96c0a4cc53690c571281131f672aa229e0d45ae59b598b59",
            ),
            (
                56,
                "da2ae4d6b36748f2a318f23e7ab1dfdf45acdc9d049bd80e59de82a60895f562",
            ),
            (
                63,
                "29af2686fd53374a36b0846694cc342177e428d1647515f078784d69cdb9e488",
            ),
            (
                64,
                "fdeab9acf3710362bd2658cdc9a29e8f9c757fcf9811603a8c447cd1d9151108",
            ),
            (
                65,
                "4bfd2c8b6f1eec7a2afeb48b934ee4b2694182027e6d0fc075074f2fabb31781",
            ),
        ];
        for (length, expected) in counting {
            let bytes: Vec<u8> = (0..length).map(|byte| byte as u8).collect();
            assert_eq!(digest(&bytes), expected, "bytes 0..{length}");
        }
    }

    #[test]
    fn incremental_updates_match_one_shot_at_every_split() {
        let message: Vec<u8> = (0..300u32).map(|i| (i * 7 + 3) as u8).collect();
        for length in [0, 1, 55, 56, 63, 64, 65, 127, 128, 129, 300] {
            let expected = sha256(&message[..length]);
            for split in 0..=length {
                let mut context = Sha256::new();
                context.update(&message[..split]);
                context.update(&message[split..length]);
                assert_eq!(
                    context.finalize(),
                    expected,
                    "length {length}, split {split}"
                );
            }
            let mut bytewise = Sha256::default();
            for byte in &message[..length] {
                bytewise.update(std::slice::from_ref(byte));
            }
            assert_eq!(
                bytewise.finalize(),
                expected,
                "length {length}, one byte at a time"
            );
        }
    }

    #[test]
    fn length_prefixed_fingerprint_equals_hashing_the_concatenation() {
        // `string_fingerprint` in text.mm: little-endian u64 count, then each string as a little-endian u64 length and its bytes.
        let values: [&[u8]; 3] = [b"", b"a b", &[0xff; 70]];
        let mut context = Sha256::new();
        let mut flat = Vec::new();
        context.update(&(values.len() as u64).to_le_bytes());
        flat.extend_from_slice(&(values.len() as u64).to_le_bytes());
        for value in values {
            context.update(&(value.len() as u64).to_le_bytes());
            context.update(value);
            flat.extend_from_slice(&(value.len() as u64).to_le_bytes());
            flat.extend_from_slice(value);
        }
        assert_eq!(context.finalize(), sha256(&flat));
    }

    #[test]
    fn hex_is_lowercase_and_debug_hides_input() {
        assert_eq!(hex(&[0x00, 0x0f, 0xa0, 0xff]), "000fa0ff");
        assert_eq!(hex(&[]), "");
        let mut context = Sha256::new();
        context.update(b"secret prompt");
        assert_eq!(format!("{context:?}"), "Sha256 { .. }");
    }

    #[test]
    #[ignore = "reads the pinned tokenizer asset under the ignored .tokspur directory (never copied into Git)"]
    fn pinned_tokenizer_json_matches_its_pin() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../.tokspur/state/gguf/Qwen3.8-Flash-Next-UD-Q4_K_XL-7b19454d8a33/tokenizer.json"
        );
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(
            hex(&sha256(&bytes)),
            "8e852142d8ab38bb7678f650bd504ba28c29d9c6357e6ed762bebdbd474689eb"
        );
    }
}
