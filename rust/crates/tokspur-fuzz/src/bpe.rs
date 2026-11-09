//! Byte-level BPE target: `tokspur-text`'s `ByteBpe` with a small fixed vocabulary (the one the tokenizer unit tests use), checked against a naive reference encoder and the decode, byte-alphabet and UTF-8 repair invariants.

#![forbid(unsafe_code)]

use std::convert::Infallible;
use std::sync::OnceLock;

use tokspur_text::tokenizer::{
    AddedToken, BpeData, ByteBpe, TextProcessor, byte_level_decode, byte_level_encode,
    replace_invalid_utf8,
};

use crate::encode::escape;

/// Multi-byte tokens after the 256 byte singletons (IDs 256 to 265).
const WORDS: [&str; 10] = [
    "ab", "aa", "aba", "abab", "aaaa", " a", "<X>", "<X>!", "<Y>", "zz",
];
/// Ranked merges, lowest rank first.
const MERGES: [(&str, &str); 6] = [
    ("a", "b"),
    ("a", "a"),
    ("ab", "a"),
    ("ab", "ab"),
    ("aa", "aa"),
    (" ", "a"),
];
/// Added tokens: ID, content, special.
const ADDED: [(u32, &str, bool); 3] =
    [(262, "<X>", true), (263, "<X>!", true), (264, "<Y>", false)];

/// The vocabulary as bytes, indexed by ID.
pub fn vocabulary() -> Vec<Vec<u8>> {
    let mut vocabulary: Vec<Vec<u8>> = (0..=u8::MAX).map(|byte| vec![byte]).collect();
    vocabulary.extend(WORDS.iter().map(|word| word.as_bytes().to_vec()));
    vocabulary
}

/// The fixed model data.
pub fn data() -> BpeData {
    BpeData {
        vocabulary: vocabulary(),
        merges: MERGES
            .iter()
            .map(|(left, right)| (left.as_bytes().to_vec(), right.as_bytes().to_vec()))
            .collect(),
        added: ADDED
            .iter()
            .map(|&(id, content, special)| AddedToken {
                id,
                content: content.as_bytes().to_vec(),
                special,
            })
            .collect(),
    }
}

/// The fixed model, built once.
pub fn model() -> &'static ByteBpe {
    static MODEL: OnceLock<ByteBpe> = OnceLock::new();
    MODEL.get_or_init(|| match ByteBpe::new(data()) {
        Ok(model) => model,
        Err(error) => panic!("the fixed BPE model is refused: {error}"),
    })
}

/// Hands each segment over unsplit, so `encode` is BPE of the raw bytes between added tokens.
struct Whole;

impl TextProcessor for Whole {
    type Error = Infallible;

    fn pieces(&self, segment: &[u8]) -> Result<Vec<Vec<u8>>, Infallible> {
        Ok(vec![segment.to_vec()])
    }
}

/// The ID of an ordinary (non-added) token with these bytes.
fn ordinary_id(vocabulary: &[Vec<u8>], bytes: &[u8]) -> Option<u32> {
    vocabulary
        .iter()
        .enumerate()
        .find(|&(id, token)| {
            token == bytes && !ADDED.iter().any(|&(added, _, _)| added as usize == id)
        })
        .and_then(|(id, _)| u32::try_from(id).ok())
}

/// Textbook BPE: repeatedly merge the adjacent pair with the lowest rank, the leftmost one on a tie, until no ranked pair remains.
pub fn reference_piece(vocabulary: &[Vec<u8>], piece: &[u8]) -> Vec<u32> {
    let mut symbols: Vec<Vec<u8>> = piece.iter().map(|&byte| vec![byte]).collect();
    loop {
        let best = (0..symbols.len().saturating_sub(1))
            .filter_map(|i| {
                MERGES
                    .iter()
                    .position(|(left, right)| {
                        symbols[i] == left.as_bytes() && symbols[i + 1] == right.as_bytes()
                    })
                    .map(|rank| (rank, i))
            })
            .min();
        let Some((_, i)) = best else {
            break;
        };
        let right = symbols.remove(i + 1);
        symbols[i].extend_from_slice(&right);
    }
    symbols
        .iter()
        .map(|symbol| match ordinary_id(vocabulary, symbol) {
            Some(id) => id,
            None => panic!("reference merge produced an unknown token"),
        })
        .collect()
}

/// Added tokens matched in raw text, longest at the leftmost position, with textbook BPE between them.
pub fn reference_encode(vocabulary: &[Vec<u8>], text: &[u8]) -> Vec<u32> {
    let mut output = Vec::new();
    let (mut plain, mut i) = (0, 0);
    while i < text.len() {
        let found = ADDED
            .iter()
            .filter(|(_, content, _)| text[i..].starts_with(content.as_bytes()))
            .max_by_key(|(_, content, _)| content.len());
        match found {
            Some(&(id, content, _)) => {
                output.extend(reference_piece(vocabulary, &text[plain..i]));
                output.push(id);
                i += content.len();
                plain = i;
            }
            None => i += 1,
        }
    }
    output.extend(reference_piece(vocabulary, &text[plain..]));
    output
}

fn special(id: u32) -> bool {
    ADDED
        .iter()
        .any(|&(added, _, special)| added == id && special)
}

/// `ByteBpe` on arbitrary bytes: `encode_piece` and `encode` agree with the reference encoder, every ID decodes back to the input (special added tokens drop out with `skip_special`), `decode_text` is the UTF-8 repair of `decode_bytes`, which is valid UTF-8, keeps valid UTF-8 unchanged and matches `String::from_utf8_lossy`; the byte alphabet round-trips, and the input read as IDs decodes exactly when every ID is in the vocabulary.
pub fn check(data: &[u8]) {
    let model = model();
    let input = || escape(data, 512);
    let vocabulary = vocabulary();
    let piece = match model.encode_piece(data) {
        Ok(ids) => ids,
        Err(error) => panic!("encode_piece refused ({error}): {}", input()),
    };
    assert_eq!(
        piece,
        reference_piece(&vocabulary, data),
        "encode_piece: {}",
        input()
    );
    let ids = match model.encode(data, &Whole) {
        Ok(ids) => ids,
        Err(error) => panic!("encode refused ({error}): {}", input()),
    };
    assert_eq!(
        ids,
        reference_encode(&vocabulary, data),
        "encode: {}",
        input()
    );
    for (encoded, skip) in [(&piece, false), (&ids, false), (&ids, true)] {
        let expected: Vec<u8> = encoded
            .iter()
            .filter(|&&id| !(skip && special(id)))
            .flat_map(|&id| vocabulary[id as usize].iter().copied())
            .collect();
        assert_eq!(
            model.decode_bytes(encoded, skip).as_deref(),
            Ok(expected.as_slice()),
            "decode_bytes: {}",
            input()
        );
    }
    let repaired = replace_invalid_utf8(data);
    assert!(
        std::str::from_utf8(&repaired).is_ok(),
        "repair is not UTF-8: {}",
        input()
    );
    assert_eq!(
        repaired,
        String::from_utf8_lossy(data).as_bytes(),
        "repair differs from from_utf8_lossy: {}",
        input()
    );
    if std::str::from_utf8(data).is_ok() {
        assert_eq!(repaired, data, "repair changed valid UTF-8: {}", input());
    }
    assert_eq!(
        model.decode_text(&ids, false).as_deref(),
        Ok(repaired.as_slice())
    );

    let spelled = byte_level_encode(data);
    assert!(
        std::str::from_utf8(&spelled).is_ok(),
        "spelling is not UTF-8: {}",
        input()
    );
    assert_eq!(
        byte_level_decode(&spelled).as_deref(),
        Ok(data),
        "byte alphabet round trip: {}",
        input()
    );
    if let Ok(bytes) = byte_level_decode(data) {
        assert_eq!(
            byte_level_encode(&bytes),
            data,
            "alphabet respelling: {}",
            input()
        );
    }

    // Two input bytes per ID; about one ID in twenty is outside the 266.
    let arbitrary: Vec<u32> = data
        .chunks(2)
        .map(|pair| {
            let high = pair.get(1).copied().unwrap_or(0);
            (u32::from(pair[0]) | (u32::from(high) << 8)) % 280
        })
        .collect();
    let decoded = model.decode_bytes(&arbitrary, false);
    if arbitrary.iter().all(|&id| (id as usize) < vocabulary.len()) {
        let expected: Vec<u8> = arbitrary
            .iter()
            .flat_map(|&id| vocabulary[id as usize].iter().copied())
            .collect();
        assert_eq!(
            decoded.as_deref(),
            Ok(expected.as_slice()),
            "decode: {}",
            input()
        );
    } else {
        assert_eq!(
            decoded.map_err(|error| error.to_string()),
            Err("token ID outside vocabulary".to_owned()),
            "decode refusal: {}",
            input()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fixture_builds_and_satisfies_every_property() {
        assert_eq!(model().vocabulary_size(), 266);
        for text in [
            b"".as_slice(),
            b"z",
            b"ba",
            b"abababa aaaaa",
            b"<X>!<X><Y> a<X",
            b"\xe6\x97\xa5\xff\xc0\x80 a",
            b"\xf0\x9f\x99\x82abab<Y>",
        ] {
            check(text);
        }
        assert_eq!(model().encode_piece(b"abab").unwrap(), [259]);
        assert_eq!(reference_encode(&vocabulary(), b"x<X>!"), [120, 263]);
    }
}
