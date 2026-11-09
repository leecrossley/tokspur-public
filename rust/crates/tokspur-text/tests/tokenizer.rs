//! `native/tests/test_tokenizer.cpp`, check for check, then the refusals and limits the C++ suite leaves untested. Where the C++ `rejected` only asks for some `FormatError`, these tests also pin the exact message.

use std::convert::Infallible;
use std::fmt;

use tokspur_testkit::Mt19937;
use tokspur_text::tokenizer::{
    AddedToken, BpeData, ByteBpe, TextProcessor, TokenizerError, byte_level_decode,
    byte_level_encode, replace_invalid_utf8,
};

/// `Identity`: the whole segment is one piece.
struct Identity;

impl TextProcessor for Identity {
    type Error = Infallible;

    fn pieces(&self, segment: &[u8]) -> Result<Vec<Vec<u8>>, Infallible> {
        Ok(vec![segment.to_vec()])
    }
}

fn bytes(text: &str) -> Vec<u8> {
    text.as_bytes().to_vec()
}

fn fixture() -> BpeData {
    let mut data = BpeData::default();
    for i in 0..=u8::MAX {
        data.vocabulary.push(vec![i]);
    }
    for word in [
        "ab", "aa", "aba", "abab", "aaaa", " a", "<X>", "<X>!", "<Y>", "zz",
    ] {
        data.vocabulary.push(bytes(word));
    }
    data.merges = [
        ("a", "b"),
        ("a", "a"),
        ("ab", "a"),
        ("ab", "ab"),
        ("aa", "aa"),
        (" ", "a"),
    ]
    .map(|(left, right)| (bytes(left), bytes(right)))
    .to_vec();
    data.added = [(262, "<X>", true), (263, "<X>!", true), (264, "<Y>", false)]
        .map(|(id, content, special)| AddedToken {
            id,
            content: bytes(content),
            special,
        })
        .to_vec();
    data
}

fn model() -> Result<ByteBpe, TokenizerError> {
    ByteBpe::new(fixture())
}

/// `slow`: recomputes every neighbouring pair at every step. Deliberately separate from the production linked nodes, heap, IDs, versions and trie. `None` is the C++ "invalid oracle fixture".
fn slow(data: &BpeData, input: &[u8]) -> Option<Vec<u32>> {
    let mut words: Vec<Vec<u8>> = input.iter().map(|&c| vec![c]).collect();
    while words.len() > 1 {
        let mut best_rank = data.merges.len();
        let mut best_index = words.len();
        for i in 0..words.len() - 1 {
            // The C++ loop re-reads `best_rank` on every iteration.
            let mut rank = 0;
            while rank < best_rank {
                if data.merges[rank].0 == words[i] && data.merges[rank].1 == words[i + 1] {
                    best_rank = rank;
                    best_index = i;
                }
                rank += 1;
            }
        }
        if best_index == words.len() {
            break;
        }
        let right = words.remove(best_index + 1);
        words[best_index].extend_from_slice(&right);
    }
    words
        .iter()
        .map(|word| {
            let found = data.vocabulary.iter().position(|token| token == word);
            found.map(|index| index as u32)
        })
        .collect()
}

// alphabet()

#[test]
fn every_byte_round_trips_through_the_alphabet() {
    // "byte alphabet round trip"
    for i in 0..=u8::MAX {
        assert_eq!(byte_level_decode(&byte_level_encode(&[i])).unwrap(), [i]);
    }
}

#[test]
fn all_bytes_round_trip_together() {
    // "all-byte alphabet round trip"
    let every: Vec<u8> = (0..=u8::MAX).collect();
    assert_eq!(
        byte_level_decode(&byte_level_encode(&every)).unwrap(),
        every
    );
}

#[test]
fn alphabet_spells_space_newline_and_nul() {
    // "byte alphabet fixed values": U+0120 U+010A U+0100.
    assert_eq!(
        byte_level_encode(b" \n\0"),
        "\u{120}\u{10a}\u{100}".as_bytes()
    );
}

#[test]
fn encoded_space_decodes() {
    // "encoded space"
    assert_eq!(
        byte_level_decode("\u{120}Hello".as_bytes()).unwrap(),
        b" Hello"
    );
}

#[test]
fn invalid_spellings_are_refused() {
    let cases: [(&[u8], &str); 5] = [
        (b" ", "character outside byte alphabet"),
        ("\u{1f989}".as_bytes(), "invalid byte-alphabet spelling"),
        (b"\xc0\x80", "invalid byte-alphabet spelling"),
        (b"\xc4", "invalid byte-alphabet spelling"),
        (b"\xc4\x20", "invalid byte-alphabet continuation"),
    ];
    for (spelling, message) in cases {
        assert_eq!(
            byte_level_decode(spelling),
            Err(TokenizerError::ByteLevelDecode(message))
        );
    }
}

// merging()

#[test]
fn fixture_geometry() {
    // "BPE geometry"
    let model = model().unwrap();
    assert_eq!((model.vocabulary_size(), model.merge_count()), (266, 6));
}

#[test]
fn empty_piece_has_no_ids() {
    // "empty BPE piece"
    assert!(model().unwrap().encode_piece(b"").unwrap().is_empty());
}

#[test]
fn merges_compose_by_rank() {
    // "ranked composition"
    assert_eq!(model().unwrap().encode_piece(b"abab").unwrap(), [259]);
}

#[test]
fn overlapping_merges_take_the_leftmost_pair() {
    // "leftmost overlapping merge"
    assert_eq!(model().unwrap().encode_piece(b"aaaaa").unwrap(), [260, 97]);
}

#[test]
fn vocabulary_words_without_a_merge_stay_split() {
    // "ignore_merges must stay false"
    assert_eq!(model().unwrap().encode_piece(b"zz").unwrap(), [122, 122]);
}

#[test]
fn longest_added_marker_wins() {
    // "longest added marker"
    let ids = model().unwrap().encode(b"ab<X>!aa<Y>", &Identity).unwrap();
    assert_eq!(ids, [256, 263, 257, 264]);
}

#[test]
fn added_tokens_round_trip() {
    // "added token round trip"
    let model = model().unwrap();
    let ids = model.encode(b"ab<X>!aa<Y>", &Identity).unwrap();
    assert_eq!(model.decode_bytes(&ids, false).unwrap(), b"ab<X>!aa<Y>");
}

#[test]
fn decoding_skips_only_special_added_tokens() {
    // "special and ordinary added tokens differ"
    let model = model().unwrap();
    let ids = model.encode(b"ab<X>!aa<Y>", &Identity).unwrap();
    assert_eq!(model.decode_bytes(&ids, true).unwrap(), b"abaa<Y>");
}

#[test]
fn piece_encoding_ignores_added_tokens() {
    // "piece encoding must not recognize added tokens"
    assert_eq!(model().unwrap().encode_piece(b"<X>").unwrap(), [60, 88, 62]);
}

#[test]
fn partial_marker_is_ordinary_text() {
    // "partial marker"
    assert_eq!(model().unwrap().encode(b"<X", &Identity).unwrap(), [60, 88]);
}

#[test]
fn adjacent_markers_both_match() {
    // "adjacent markers"
    assert_eq!(
        model().unwrap().encode(b"<X><X>", &Identity).unwrap(),
        [262, 262]
    );
}

#[test]
fn heap_merging_matches_the_exhaustive_rank_oracle() {
    // "heap BPE differs from exhaustive rank oracle" and "BPE byte round trip", over the C++ test's exact 5,000 inputs.
    let data = fixture();
    let model = ByteBpe::new(data.clone()).unwrap();
    let mut random = Mt19937::new(0x6270_6531);
    for run in 0..5000 {
        let mut input = Vec::new();
        let count = random.next_u32() % 129;
        for _ in 0..count {
            input.push(if run % 2 == 1 {
                b"ab "[(random.next_u32() % 3) as usize]
            } else {
                (random.next_u32() & 255) as u8
            });
        }
        let actual = model.encode_piece(&input).unwrap();
        let expected = slow(&data, &input).expect("invalid oracle fixture");
        assert_eq!(actual, expected, "run {run}");
        assert_eq!(
            model.decode_bytes(&actual, false).unwrap(),
            input,
            "run {run}"
        );
    }
}

#[test]
fn oversized_piece_is_refused() {
    assert_eq!(
        model().unwrap().encode_piece(&vec![b'a'; 1024 * 1024 + 1]),
        Err(TokenizerError::EncodePiece("BPE piece exceeds limit"))
    );
}

#[test]
fn oversized_input_is_refused() {
    assert_eq!(
        model()
            .unwrap()
            .encode(&vec![b'a'; 8 * 1024 * 1024 + 1], &Identity),
        Err(TokenizerError::Encode("tokenizer input exceeds limit"))
    );
}

#[test]
fn token_id_outside_the_vocabulary_is_refused() {
    assert_eq!(
        model().unwrap().decode_bytes(&[266], false),
        Err(TokenizerError::DecodeBytes("token ID outside vocabulary"))
    );
}

// invalid_models()

fn refused(data: BpeData) -> &'static str {
    match ByteBpe::new(data) {
        Err(TokenizerError::New(message)) => message,
        other => panic!("invalid tokenizer input was accepted: {other:?}"),
    }
}

#[test]
fn empty_vocabulary_is_refused() {
    let mut data = fixture();
    data.vocabulary.clear();
    assert_eq!(refused(data), "unsupported BPE vocabulary size");
}

#[test]
fn duplicate_ordinary_token_is_refused() {
    let mut data = fixture();
    data.vocabulary[1] = data.vocabulary[0].clone();
    assert_eq!(refused(data), "duplicate ordinary BPE token");
}

#[test]
fn vocabulary_without_every_byte_singleton_is_refused() {
    let mut data = fixture();
    data.vocabulary[1] = bytes("no-singleton");
    assert_eq!(refused(data), "BPE vocabulary lacks a byte singleton");
}

#[test]
fn duplicate_merge_is_refused() {
    let mut data = fixture();
    data.merges.push(data.merges[0].clone());
    assert_eq!(refused(data), "invalid or duplicate BPE merge");
}

#[test]
fn merge_without_an_output_token_is_refused() {
    let mut data = fixture();
    data.merges.push((bytes("a"), bytes("z")));
    assert_eq!(refused(data), "invalid or duplicate BPE merge");
}

#[test]
fn merge_with_an_empty_part_is_refused() {
    let mut data = fixture();
    data.merges.push((bytes(""), bytes("a")));
    assert_eq!(refused(data), "invalid BPE merge parts");
}

#[test]
fn added_token_whose_vocabulary_entry_differs_is_refused() {
    let mut data = fixture();
    data.added[0].id = 0;
    assert_eq!(refused(data), "invalid added token");
}

#[test]
fn repeated_added_token_id_is_refused() {
    let mut data = fixture();
    data.added.push(data.added[0].clone());
    assert_eq!(refused(data), "invalid added token");
}

#[test]
fn duplicate_added_token_content_is_refused() {
    let mut data = fixture();
    data.vocabulary[263] = bytes("<X>");
    data.added[1].content = bytes("<X>");
    assert_eq!(refused(data), "duplicate added token content");
}

#[test]
fn oversized_vocabulary_token_is_refused() {
    let mut data = fixture();
    data.vocabulary[265].resize(4097, b'x');
    assert_eq!(refused(data), "BPE vocabulary byte limit exceeded");
}

// lossy_utf8()

const REPLACEMENT: &str = "\u{fffd}";

#[test]
fn valid_utf8_is_unchanged() {
    // "valid UTF-8 changed"
    for value in [&b""[..], "caf\u{e9}\u{1f989}".as_bytes(), b"\0abc"] {
        assert_eq!(replace_invalid_utf8(value), value);
    }
}

#[test]
fn overlong_utf8_is_replaced_byte_by_byte() {
    // "overlong UTF-8"
    assert_eq!(
        replace_invalid_utf8(b"\xc0\x80"),
        REPLACEMENT.repeat(2).as_bytes()
    );
}

#[test]
fn partial_sequence_is_one_maximal_subpart() {
    // "partial UTF-8 maximal subpart"
    assert_eq!(
        replace_invalid_utf8(b"\xe1\x80A"),
        format!("{REPLACEMENT}A").as_bytes()
    );
}

#[test]
fn truncated_sequence_is_one_maximal_subpart() {
    // "truncated UTF-8 maximal subpart"
    assert_eq!(
        replace_invalid_utf8(b"\xf1\x80\x80"),
        REPLACEMENT.as_bytes()
    );
}

#[test]
fn surrogate_is_replaced_byte_by_byte() {
    // "UTF-8 surrogate"
    assert_eq!(
        replace_invalid_utf8(b"\xed\xa0\x80"),
        REPLACEMENT.repeat(3).as_bytes()
    );
}

#[test]
fn value_past_the_last_scalar_is_replaced_byte_by_byte() {
    // "UTF-8 out of range"
    assert_eq!(
        replace_invalid_utf8(b"\xf4\x90\x80\x80"),
        REPLACEMENT.repeat(4).as_bytes()
    );
}

// Refusals and limits beyond the C++ suite.

/// A processor that refuses every segment with its own message.
struct Refusing;

#[derive(Debug, PartialEq)]
struct Refusal;

impl fmt::Display for Refusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("pretokenizer piece limit exceeded")
    }
}

impl std::error::Error for Refusal {}

impl TextProcessor for Refusing {
    type Error = Refusal;

    fn pieces(&self, _segment: &[u8]) -> Result<Vec<Vec<u8>>, Refusal> {
        Err(Refusal)
    }
}

/// A processor that returns fixed pieces whatever the segment.
struct Fixed(Vec<Vec<u8>>);

impl TextProcessor for Fixed {
    type Error = Infallible;

    fn pieces(&self, _segment: &[u8]) -> Result<Vec<Vec<u8>>, Infallible> {
        Ok(self.0.clone())
    }
}

#[test]
fn processor_refusal_reaches_the_caller_unchanged() {
    let model = model().unwrap();
    let error = model.encode(b"<X>abc", &Refusing).unwrap_err();
    assert_eq!(error, TokenizerError::Pieces(Refusal));
    assert_eq!(error.to_string(), "pretokenizer piece limit exceeded");
    // Text that is only markers never reaches the processor.
    assert_eq!(model.encode(b"<X><Y>", &Refusing), Ok(vec![262, 264]));
    assert_eq!(model.encode(b"", &Refusing), Ok(vec![]));
}

#[test]
fn processor_pieces_are_encoded_in_order_with_empty_ones_skipped() {
    let pieces = Fixed(vec![bytes(" a"), bytes(""), bytes("aaaa"), bytes("zz")]);
    assert_eq!(
        model().unwrap().encode(b"q", &pieces).unwrap(),
        [261, 260, 122, 122]
    );
}

#[test]
fn normalized_bytes_past_the_input_limit_are_refused_before_encoding() {
    // The byte count is checked before the piece is encoded, so the piece limit is never reached.
    let huge = Fixed(vec![vec![b'a'; 8 * 1024 * 1024 + 1]]);
    assert_eq!(
        model().unwrap().encode(b"x", &huge),
        Err(TokenizerError::Encode("normalized segment exceeds limit"))
    );
    let large = Fixed(vec![vec![b'a'; 1024 * 1024 + 1]]);
    assert_eq!(
        model().unwrap().encode(b"x", &large),
        Err(TokenizerError::EncodePiece("BPE piece exceeds limit"))
    );
}

#[test]
fn piece_output_past_the_token_limit_is_refused() {
    let model = model().unwrap();
    assert_eq!(
        model.encode_piece(&vec![b'z'; 1_000_000]).unwrap().len(),
        1_000_000
    );
    assert_eq!(
        model.encode_piece(&vec![b'z'; 1_000_001]),
        Err(TokenizerError::EncodePiece("BPE output exceeds limit"))
    );
}

#[test]
fn marker_past_the_token_limit_is_refused() {
    let mut text = vec![b'z'; 1_000_000];
    text.extend_from_slice(b"<X>");
    assert_eq!(
        model().unwrap().encode(&text, &Identity),
        Err(TokenizerError::Encode("tokenizer output exceeds limit"))
    );
}

#[test]
fn segments_past_the_token_limit_are_refused() {
    let mut text = vec![b'z'; 600_000];
    text.extend_from_slice(b"<X>");
    text.extend_from_slice(&vec![b'z'; 600_000]);
    assert_eq!(
        model().unwrap().encode(&text, &Identity),
        Err(TokenizerError::Encode("tokenizer output limit exceeded"))
    );
}

#[test]
fn decoding_past_the_token_or_byte_limit_is_refused() {
    let mut data = fixture();
    data.vocabulary.push(vec![b'q'; 4096]);
    let model = ByteBpe::new(data).unwrap();
    assert_eq!(
        model.decode_bytes(&vec![97; 1_000_001], false),
        Err(TokenizerError::DecodeBytes("decoder token limit exceeded"))
    );
    assert_eq!(
        model
            .decode_bytes(&vec![97; 1_000_000], false)
            .unwrap()
            .len(),
        1_000_000
    );
    // 2,048 x 4,096 bytes is exactly the 8 MiB limit; one byte more is refused.
    let mut ids = vec![266; 2048];
    assert_eq!(
        model.decode_bytes(&ids, false).unwrap().len(),
        8 * 1024 * 1024
    );
    ids.push(97);
    assert_eq!(
        model.decode_bytes(&ids, false),
        Err(TokenizerError::DecodeBytes("decoded byte limit exceeded"))
    );
    assert_eq!(
        model.decode_text(&[266, 1_000], false),
        Err(TokenizerError::DecodeBytes("token ID outside vocabulary"))
    );
}

#[test]
fn decode_text_replaces_invalid_utf8_after_skipping() {
    let model = model().unwrap();
    assert_eq!(
        model.decode_text(&[0xc3, 262, 0xa9], false).unwrap(),
        "\u{fffd}<X>\u{fffd}".as_bytes()
    );
    assert_eq!(
        model.decode_text(&[0xc3, 262, 0xa9], true).unwrap(),
        "\u{e9}".as_bytes()
    );
}

#[test]
fn model_size_limits_are_refused() {
    let mut data = fixture();
    data.vocabulary.truncate(255);
    data.added.clear();
    assert_eq!(refused(data), "unsupported BPE vocabulary size");
    let mut data = fixture();
    data.vocabulary.resize(1_000_001, Vec::new());
    assert_eq!(refused(data), "unsupported BPE vocabulary size");
    let mut data = fixture();
    data.merges.resize(1_000_001, (Vec::new(), Vec::new()));
    assert_eq!(refused(data), "unsupported BPE vocabulary size");
    let mut data = fixture();
    let marker = data.added[0].clone();
    data.added.resize(4097, marker);
    assert_eq!(refused(data), "unsupported BPE vocabulary size");
}

#[test]
fn added_token_shape_is_checked_before_its_vocabulary_entry() {
    for (id, content) in [(266, "<X>"), (262, ""), (262, "<Z>")] {
        let mut data = fixture();
        data.added[0] = AddedToken {
            id,
            content: bytes(content),
            special: true,
        };
        assert_eq!(refused(data), "invalid added token");
    }
    let mut data = fixture();
    data.vocabulary[262] = vec![b'<'; 4097];
    data.added[0].content = vec![b'<'; 4097];
    assert_eq!(refused(data), "invalid added token");
}

#[test]
fn added_token_trie_past_its_node_limit_is_refused() {
    // Sixteen 4,096-byte markers with distinct first bytes need 65,536 nodes besides the root; the last one does not fit.
    let mut data = fixture();
    data.added.clear();
    for marker in 0..16u8 {
        let mut content = vec![b'#'; 4096];
        content[0] = b'A' + marker;
        data.added.push(AddedToken {
            id: data.vocabulary.len() as u32,
            content: content.clone(),
            special: false,
        });
        data.vocabulary.push(content);
    }
    assert_eq!(refused(data.clone()), "added token trie exceeds limit");
    data.added.pop();
    data.vocabulary.pop();
    let model = ByteBpe::new(data).unwrap();
    assert_eq!(model.vocabulary_size(), 266 + 15);
}

#[test]
fn vocabulary_past_the_byte_budget_is_refused() {
    // 16,385 distinct 4,096-byte tokens pass 64 MiB in total.
    let mut data = fixture();
    data.added.clear();
    for i in 0..16_385u32 {
        let mut token = vec![b'.'; 4096];
        token[..4].copy_from_slice(&i.to_le_bytes());
        token[4] = 0xff;
        data.vocabulary.push(token);
    }
    assert_eq!(refused(data), "BPE vocabulary byte limit exceeded");
}

#[test]
fn empty_vocabulary_token_is_refused() {
    let mut data = fixture();
    data.vocabulary[265].clear();
    assert_eq!(refused(data), "BPE vocabulary byte limit exceeded");
}

#[test]
fn merge_parts_past_the_token_size_are_refused() {
    let mut data = fixture();
    data.vocabulary.push(vec![b'a'; 2048]);
    data.vocabulary.push(vec![b'b'; 2049]);
    data.merges.push((vec![b'a'; 2048], vec![b'b'; 2049]));
    assert_eq!(refused(data), "invalid BPE merge parts");
}

#[test]
fn added_token_content_is_not_an_ordinary_merge_part() {
    // "<X>" is an added token, so it is absent from the ordinary IDs.
    let mut data = fixture();
    data.merges.push((bytes("<X>"), bytes("!")));
    assert_eq!(refused(data), "invalid BPE merge parts");
}

#[test]
fn added_token_may_repeat_an_ordinary_token() {
    // An ordinary "ab" (256) and an added "ab" (266) coexist; raw text then matches the added token, while pieces still merge to the ordinary one.
    let mut data = fixture();
    data.vocabulary.push(bytes("ab"));
    data.added.push(AddedToken {
        id: 266,
        content: bytes("ab"),
        special: false,
    });
    let model = ByteBpe::new(data).unwrap();
    assert_eq!(model.encode(b"abab", &Identity).unwrap(), [266, 266]);
    assert_eq!(model.encode_piece(b"abab").unwrap(), [259]);
}
