//! SHA-256 through CommonCrypto's `CC_SHA256` family, as `native/src/text.mm` computes it. The CLI and the pack builder hash model shards, the derived pack and build artefacts with it, because CommonCrypto uses the SHA-256 instructions of Apple silicon. The server keeps the portable `tokspur_text::sha256` for its asset pins.

use std::fmt;

use crate::ffi;

/// `CC_SHA256_DIGEST_LENGTH`.
pub const DIGEST_LENGTH: usize = ffi::CC_SHA256_DIGEST_LENGTH;

/// The most bytes one `CC_SHA256_Update` call is given, as its length is a 32-bit `CC_LONG`.
const UPDATE_LIMIT: usize = 1 << 30;

/// Incremental SHA-256 over a `CC_SHA256_CTX`. `Debug` is hand-written and prints no hashed input.
pub struct Sha256 {
    context: ffi::CcSha256Ctx,
}

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Sha256 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Sha256").finish_non_exhaustive()
    }
}

impl Sha256 {
    /// `CC_SHA256_Init`.
    pub fn new() -> Self {
        let mut context = ffi::CcSha256Ctx {
            count: [0; 2],
            hash: [0; 8],
            wbuf: [0; 16],
        };
        // SAFETY: `context` is writable storage matching `CC_SHA256_CTX`, whose layout `tests/sdk_bindings.rs` checks against the SDK.
        unsafe { ffi::CC_SHA256_Init(&mut context) };
        Self { context }
    }

    /// `CC_SHA256_Update`, in pieces whose length fits `CC_LONG`.
    pub fn update(&mut self, bytes: &[u8]) {
        for piece in bytes.chunks(UPDATE_LIMIT) {
            // SAFETY: `piece` is readable for its length, which fits `CC_LONG`, and `CC_SHA256_Init` initialised `context`.
            unsafe {
                ffi::CC_SHA256_Update(
                    &mut self.context,
                    piece.as_ptr().cast(),
                    piece.len() as ffi::CcLong,
                )
            };
        }
    }

    /// `CC_SHA256_Final`.
    pub fn finalize(mut self) -> [u8; DIGEST_LENGTH] {
        let mut digest = [0u8; DIGEST_LENGTH];
        // SAFETY: `digest` is writable for the `CC_SHA256_DIGEST_LENGTH` bytes `CC_SHA256_Final` stores, and `CC_SHA256_Init` initialised `context`.
        unsafe { ffi::CC_SHA256_Final(digest.as_mut_ptr(), &mut self.context) };
        digest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn digest(bytes: &[u8]) -> String {
        let mut context = Sha256::new();
        context.update(bytes);
        hex(&context.finalize())
    }

    #[test]
    fn nist_messages() {
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
            digest(&vec![b'a'; 1_000_000]),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
    }

    #[test]
    fn incremental_updates_match_one_shot_at_every_split() {
        let message: Vec<u8> = (0..300u32).map(|i| (i * 7 + 3) as u8).collect();
        for length in [0, 1, 55, 56, 63, 64, 65, 127, 128, 129, 300] {
            let expected = digest(&message[..length]);
            for split in 0..=length {
                let mut context = Sha256::default();
                context.update(&message[..split]);
                context.update(&message[split..length]);
                assert_eq!(
                    hex(&context.finalize()),
                    expected,
                    "length {length}, split {split}"
                );
            }
        }
    }

    #[test]
    fn debug_hides_input() {
        let mut context = Sha256::new();
        context.update(b"secret prompt");
        assert_eq!(format!("{context:?}"), "Sha256 { .. }");
    }
}
