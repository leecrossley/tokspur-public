//! Byte-level BPE from `native/src/tokenizer.cpp`: the byte alphabet, lossy UTF-8 repair and `ByteBpe` with its ranked merges, added-token trie and limits. Text is bytes throughout, as the C++ `std::string` is.

use std::cmp::Reverse;
use std::collections::hash_map::Entry;
use std::collections::{BTreeSet, BinaryHeap, HashMap};
use std::convert::Infallible;
use std::fmt;

const INPUT_LIMIT: usize = 8 * 1024 * 1024;
const PIECE_LIMIT: usize = 1024 * 1024;
const TOKEN_LIMIT: usize = 1_000_000;
const ABSENT: u32 = u32::MAX;

/// The C++ `FormatError`s this module raises. `Display` is the exact C++ message. `E` is a `TextProcessor`'s own error, carried by `Pieces`; it is `Infallible` wherever no processor is involved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenizerError<E = Infallible> {
    /// `byte_level_decode` refused a spelling.
    ByteLevelDecode(&'static str),
    /// `ByteBpe::new` refused the model data.
    New(&'static str),
    /// `ByteBpe::encode_piece` refused the piece or its output.
    EncodePiece(&'static str),
    /// `ByteBpe::encode` refused the text or its output.
    Encode(&'static str),
    /// The `TextProcessor` refused a segment. Transparent: its message is the caller's message, as the C++ exception leaves `encode` untouched.
    Pieces(E),
    /// `ByteBpe::decode_bytes` (and so `decode_text`) refused the IDs.
    DecodeBytes(&'static str),
}

impl TokenizerError {
    /// Carries a processor-free refusal into `encode`'s error type.
    fn widen<E>(self) -> TokenizerError<E> {
        match self {
            Self::ByteLevelDecode(reason) => TokenizerError::ByteLevelDecode(reason),
            Self::New(reason) => TokenizerError::New(reason),
            Self::EncodePiece(reason) => TokenizerError::EncodePiece(reason),
            Self::Encode(reason) => TokenizerError::Encode(reason),
            Self::DecodeBytes(reason) => TokenizerError::DecodeBytes(reason),
            Self::Pieces(never) => match never {},
        }
    }
}

impl<E: fmt::Display> fmt::Display for TokenizerError<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ByteLevelDecode(reason)
            | Self::New(reason)
            | Self::EncodePiece(reason)
            | Self::Encode(reason)
            | Self::DecodeBytes(reason) => formatter.write_str(reason),
            Self::Pieces(error) => error.fmt(formatter),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for TokenizerError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Pieces(error) => Some(error),
            Self::ByteLevelDecode(_)
            | Self::New(_)
            | Self::EncodePiece(_)
            | Self::Encode(_)
            | Self::DecodeBytes(_) => None,
        }
    }
}

/// `alphabet`: the visible code point that spells each byte. Visible Latin-1 bytes spell themselves; the rest take 256, 257, ... in byte order.
const fn alphabet() -> [u16; 256] {
    let mut result = [0u16; 256];
    let mut extension: u16 = 256;
    let mut byte = 0;
    while byte < 256 {
        let visible = matches!(byte, 33..=126 | 161..=172 | 174..);
        result[byte] = if visible {
            byte as u16
        } else {
            extension += 1;
            extension - 1
        };
        byte += 1;
    }
    result
}

const CODEPOINTS: [u16; 256] = alphabet();

/// `reverse_codepoints`: the byte each alphabet code point spells, or -1.
const REVERSE_CODEPOINTS: [i16; 324] = {
    let mut result = [-1i16; 324];
    let mut i = 0;
    while i < CODEPOINTS.len() {
        result[CODEPOINTS[i] as usize] = i as i16;
        i += 1;
    }
    result
};

fn pair_key(left: u32, right: u32) -> u64 {
    (u64::from(left) << 32) | u64::from(right)
}

fn append_ids<E>(output: &mut Vec<u32>, more: &[u32]) -> Result<(), TokenizerError<E>> {
    if more.len() > TOKEN_LIMIT - output.len() {
        return Err(TokenizerError::Encode("tokenizer output limit exceeded"));
    }
    output.extend_from_slice(more);
    Ok(())
}

/// `byte_level_encode`: spells every byte with its alphabet code point.
pub fn byte_level_encode(bytes: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        let point = CODEPOINTS[usize::from(byte)];
        if point < 128 {
            result.push(point as u8);
        } else {
            result.push((0xc0 | (point >> 6)) as u8);
            result.push((0x80 | (point & 63)) as u8);
        }
    }
    result
}

/// `byte_level_decode`: the bytes an alphabet spelling represents.
pub fn byte_level_decode(spelling: &[u8]) -> Result<Vec<u8>, TokenizerError> {
    let mut result = Vec::with_capacity(spelling.len());
    let mut i = 0;
    while i < spelling.len() {
        let mut point = u32::from(spelling[i]);
        i += 1;
        if point >= 128 {
            if !(0xc2..=0xc5).contains(&point) || i == spelling.len() {
                return Err(TokenizerError::ByteLevelDecode(
                    "invalid byte-alphabet spelling",
                ));
            }
            let continuation = spelling[i];
            i += 1;
            if continuation & 0xc0 != 0x80 {
                return Err(TokenizerError::ByteLevelDecode(
                    "invalid byte-alphabet continuation",
                ));
            }
            point = ((point & 31) << 6) | u32::from(continuation & 63);
        }
        match REVERSE_CODEPOINTS.get(point as usize) {
            Some(&byte) if byte >= 0 => result.push(byte as u8),
            _ => {
                return Err(TokenizerError::ByteLevelDecode(
                    "character outside byte alphabet",
                ));
            }
        }
    }
    Ok(result)
}

/// `replace_invalid_utf8`: replaces each maximal invalid subpart with U+FFFD and keeps valid UTF-8 unchanged.
pub fn replace_invalid_utf8(bytes: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let first = bytes[i];
        if first < 128 {
            result.push(first);
            i += 1;
            continue;
        }
        let count = match first {
            0xc2..=0xdf => 2,
            0xe0..=0xef => 3,
            0xf0..=0xf4 => 4,
            _ => 0,
        };
        let mut valid_prefix = 1;
        if count != 0 {
            while valid_prefix < count && i + valid_prefix < bytes.len() {
                let next = bytes[i + valid_prefix];
                let continuation = (0x80..=0xbf).contains(&next);
                let boundary = valid_prefix != 1
                    || !((first == 0xe0 && next < 0xa0)
                        || (first == 0xed && next > 0x9f)
                        || (first == 0xf0 && next < 0x90)
                        || (first == 0xf4 && next > 0x8f));
                if !continuation || !boundary {
                    break;
                }
                valid_prefix += 1;
            }
        }
        if count != 0 && valid_prefix == count {
            result.extend_from_slice(&bytes[i..i + count]);
        } else {
            result.extend_from_slice("\u{fffd}".as_bytes());
        }
        i += valid_prefix;
    }
    result
}

/// `AddedToken`: a marker matched in raw text before any processing. `Debug` is hand-written and prints neither the ID nor the content.
#[derive(Clone, PartialEq, Eq)]
pub struct AddedToken {
    pub id: u32,
    pub content: Vec<u8>,
    pub special: bool,
}

impl fmt::Debug for AddedToken {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AddedToken")
            .field("content", &format_args!("{} bytes", self.content.len()))
            .field("special", &self.special)
            .finish_non_exhaustive()
    }
}

/// `BpeData`. `Debug` is hand-written and prints only counts.
#[derive(Clone, Default, PartialEq, Eq)]
pub struct BpeData {
    /// Indexed by ID, already decoded to bytes.
    pub vocabulary: Vec<Vec<u8>>,
    /// Ordered by rank.
    pub merges: Vec<(Vec<u8>, Vec<u8>)>,
    pub added: Vec<AddedToken>,
}

impl fmt::Debug for BpeData {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BpeData")
            .field(
                "vocabulary",
                &format_args!("{} tokens", self.vocabulary.len()),
            )
            .field("merges", &format_args!("{} pairs", self.merges.len()))
            .field("added", &format_args!("{} tokens", self.added.len()))
            .finish()
    }
}

/// `TextProcessor`.
pub trait TextProcessor {
    /// The processor's own refusal. `ByteBpe::encode` returns it unchanged as `TokenizerError::Pieces`, so its message reaches the caller exactly.
    type Error: std::error::Error + 'static;

    /// Normalize and split one segment that contains no recognized added token.
    fn pieces(&self, segment: &[u8]) -> Result<Vec<Vec<u8>>, Self::Error>;
}

#[derive(Clone, Copy)]
struct Merge {
    rank: u32,
    output: u32,
}

/// `MarkerNode`: one added-token trie node; `id` is the C++ `-1` as `None`.
#[derive(Default)]
struct MarkerNode {
    next: HashMap<u8, u32>,
    id: Option<u32>,
}

/// `encode_piece`'s doubly linked symbol.
#[derive(Clone, Copy)]
struct Node {
    id: u32,
    previous: u32,
    next: u32,
    version: u32,
    alive: bool,
}

/// `encode_piece`'s queued merge. The derived order compares `rank`, then `left`, as the C++ `Later` comparator does, so `BinaryHeap<Reverse<_>>` pops the lowest rank and then the leftmost pair first. The later fields only order entries that `Later` leaves tied, and those share `left`: at most one of them can still be current (a current entry's `right`, versions and `output` follow from `left`'s state), so the tie order is unobservable.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Candidate {
    rank: u32,
    left: u32,
    right: u32,
    left_version: u32,
    right_version: u32,
    output: u32,
}

/// `ByteBpe`. `Debug` is hand-written and prints only sizes.
pub struct ByteBpe {
    vocabulary: Vec<Vec<u8>>,
    special: Vec<bool>,
    single: [u32; 256],
    merges: HashMap<u64, Merge>,
    markers: Vec<MarkerNode>,
}

impl fmt::Debug for ByteBpe {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ByteBpe")
            .field(
                "vocabulary",
                &format_args!("{} tokens", self.vocabulary.len()),
            )
            .field("merges", &format_args!("{} pairs", self.merges.len()))
            .field("markers", &format_args!("{} nodes", self.markers.len()))
            .finish()
    }
}

impl ByteBpe {
    /// `ByteBpe(BpeData)`: validates the model and builds the merge table and the added-token trie. Checks run in the C++ order, so the first failing check names the refusal.
    pub fn new(data: BpeData) -> Result<Self, TokenizerError> {
        let BpeData {
            vocabulary,
            merges: ranked,
            added,
        } = data;
        let mut special = vec![false; vocabulary.len()];
        let mut markers = vec![MarkerNode::default()];
        if vocabulary.len() < 256
            || vocabulary.len() > TOKEN_LIMIT
            || ranked.len() > TOKEN_LIMIT
            || added.len() > 4096
        {
            return Err(TokenizerError::New("unsupported BPE vocabulary size"));
        }
        let mut added_ids = BTreeSet::new();
        for marker in &added {
            let index = marker.id as usize;
            if index >= vocabulary.len()
                || marker.content.is_empty()
                || marker.content.len() > 4096
                || vocabulary[index] != marker.content
                || !added_ids.insert(marker.id)
            {
                return Err(TokenizerError::New("invalid added token"));
            }
            special[index] = marker.special;
            let mut node = 0;
            for &c in &marker.content {
                node = match markers[node].next.get(&c) {
                    Some(&found) => found as usize,
                    None => {
                        if markers.len() >= 65536 {
                            return Err(TokenizerError::New("added token trie exceeds limit"));
                        }
                        let next = markers.len() as u32;
                        markers[node].next.insert(c, next);
                        markers.push(MarkerNode::default());
                        next as usize
                    }
                };
            }
            if markers[node].id.is_some() {
                return Err(TokenizerError::New("duplicate added token content"));
            }
            markers[node].id = Some(marker.id);
        }
        let mut ids: HashMap<&[u8], u32> = HashMap::with_capacity(vocabulary.len());
        let mut total_bytes: u64 = 0;
        for (i, token) in vocabulary.iter().enumerate() {
            // `checked_add` from gguf.cpp, with its message; unreachable here because the total stays at most 64 MiB plus one token.
            total_bytes = total_bytes
                .checked_add(token.len() as u64)
                .ok_or(TokenizerError::New("GGUF offset overflow"))?;
            if token.is_empty() || token.len() > 4096 || total_bytes > 64 * 1024 * 1024 {
                return Err(TokenizerError::New("BPE vocabulary byte limit exceeded"));
            }
            if !added_ids.contains(&(i as u32)) {
                match ids.entry(token.as_slice()) {
                    Entry::Occupied(_) => {
                        return Err(TokenizerError::New("duplicate ordinary BPE token"));
                    }
                    Entry::Vacant(slot) => {
                        slot.insert(i as u32);
                    }
                }
            }
        }
        let mut single = [0u32; 256];
        for (byte, slot) in (0..=u8::MAX).zip(single.iter_mut()) {
            let Some(&found) = ids.get([byte].as_slice()) else {
                return Err(TokenizerError::New("BPE vocabulary lacks a byte singleton"));
            };
            *slot = found;
        }
        let mut merges = HashMap::with_capacity(ranked.len());
        for (rank, (left, right)) in ranked.iter().enumerate() {
            let (l, r) = match (ids.get(left.as_slice()), ids.get(right.as_slice())) {
                (Some(&l), Some(&r))
                    if !left.is_empty()
                        && !right.is_empty()
                        && left.len() + right.len() <= 4096 =>
                {
                    (l, r)
                }
                _ => return Err(TokenizerError::New("invalid BPE merge parts")),
            };
            let mut joined = Vec::with_capacity(left.len() + right.len());
            joined.extend_from_slice(left);
            joined.extend_from_slice(right);
            let Some(&output) = ids.get(joined.as_slice()) else {
                return Err(TokenizerError::New("invalid or duplicate BPE merge"));
            };
            match merges.entry(pair_key(l, r)) {
                Entry::Occupied(_) => {
                    return Err(TokenizerError::New("invalid or duplicate BPE merge"));
                }
                Entry::Vacant(slot) => {
                    slot.insert(Merge {
                        rank: rank as u32,
                        output,
                    });
                }
            }
        }
        Ok(Self {
            vocabulary,
            special,
            single,
            merges,
            markers,
        })
    }

    /// Queues the merge of `left` with its right neighbour, if one is ranked.
    fn consider(&self, nodes: &[Node], queue: &mut BinaryHeap<Reverse<Candidate>>, left: u32) {
        if left == ABSENT || nodes[left as usize].next == ABSENT {
            return;
        }
        let right = nodes[left as usize].next;
        let Some(found) = self
            .merges
            .get(&pair_key(nodes[left as usize].id, nodes[right as usize].id))
        else {
            return;
        };
        queue.push(Reverse(Candidate {
            rank: found.rank,
            left,
            right,
            left_version: nodes[left as usize].version,
            right_version: nodes[right as usize].version,
            output: found.output,
        }));
    }

    /// `encode_piece`: byte-level BPE of one piece. Added tokens are not recognized here.
    pub fn encode_piece(&self, bytes: &[u8]) -> Result<Vec<u32>, TokenizerError> {
        if bytes.len() > PIECE_LIMIT {
            return Err(TokenizerError::EncodePiece("BPE piece exceeds limit"));
        }
        if bytes.is_empty() {
            return Ok(Vec::new());
        }
        let mut nodes = Vec::with_capacity(bytes.len());
        for (i, &byte) in bytes.iter().enumerate() {
            nodes.push(Node {
                id: self.single[usize::from(byte)],
                previous: if i != 0 { (i - 1) as u32 } else { ABSENT },
                next: if i + 1 < bytes.len() {
                    (i + 1) as u32
                } else {
                    ABSENT
                },
                version: 0,
                alive: true,
            });
        }
        let mut queue = BinaryHeap::new();
        for i in 0..nodes.len() - 1 {
            self.consider(&nodes, &mut queue, i as u32);
        }
        while let Some(Reverse(candidate)) = queue.pop() {
            // `left` precedes `right`: `next` links only ever point forward.
            let (before, from_right) = nodes.split_at_mut(candidate.right as usize);
            let (left, right) = (&mut before[candidate.left as usize], &mut from_right[0]);
            if !left.alive
                || !right.alive
                || left.next != candidate.right
                || left.version != candidate.left_version
                || right.version != candidate.right_version
            {
                continue;
            }
            left.id = candidate.output;
            left.next = right.next;
            left.version = left.version.wrapping_add(1);
            right.alive = false;
            right.version = right.version.wrapping_add(1);
            let (next, previous) = (left.next, left.previous);
            if next != ABSENT {
                nodes[next as usize].previous = candidate.left;
            }
            self.consider(&nodes, &mut queue, previous);
            self.consider(&nodes, &mut queue, candidate.left);
        }
        let mut output = Vec::new();
        let mut i = 0;
        while i != ABSENT {
            if output.len() >= TOKEN_LIMIT {
                return Err(TokenizerError::EncodePiece("BPE output exceeds limit"));
            }
            output.push(nodes[i as usize].id);
            i = nodes[i as usize].next;
        }
        Ok(output)
    }

    /// `encode`: matches added tokens in raw text (longest match at the leftmost position) and hands the text between them to `processor`.
    pub fn encode<P>(
        &self,
        text: &[u8],
        processor: &P,
    ) -> Result<Vec<u32>, TokenizerError<P::Error>>
    where
        P: TextProcessor + ?Sized,
    {
        self.encode_with_literals(text, processor, &[])
    }

    /// Disables added-token recognition within validated renderer-owned byte ranges.
    pub fn encode_with_literals<P>(
        &self,
        text: &[u8],
        processor: &P,
        literal_ranges: &[std::ops::Range<usize>],
    ) -> Result<Vec<u32>, TokenizerError<P::Error>>
    where
        P: TextProcessor + ?Sized,
    {
        if text.len() > INPUT_LIMIT {
            return Err(TokenizerError::Encode("tokenizer input exceeds limit"));
        }
        let mut last = 0;
        for range in literal_ranges {
            if range.start < last || range.end <= range.start || range.end > text.len() {
                return Err(TokenizerError::Encode("invalid tokenizer literal spans"));
            }
            last = range.end;
        }
        let mut output = Vec::new();
        let ordinary =
            |output: &mut Vec<u32>, segment: &[u8]| -> Result<(), TokenizerError<P::Error>> {
                if segment.is_empty() {
                    return Ok(());
                }
                let mut normalized_bytes = 0usize;
                for piece in processor.pieces(segment).map_err(TokenizerError::Pieces)? {
                    normalized_bytes += piece.len();
                    if normalized_bytes > INPUT_LIMIT {
                        return Err(TokenizerError::Encode("normalized segment exceeds limit"));
                    }
                    append_ids(
                        output,
                        &self.encode_piece(&piece).map_err(TokenizerError::widen)?,
                    )?;
                }
                Ok(())
            };
        let mut plain_start = 0;
        let mut i = 0;
        let mut literal = 0;
        while i < text.len() {
            let mut node = 0;
            // The C++ `match`, a keyword here.
            let mut found = None;
            let mut end = i;
            for (cursor, c) in text.iter().enumerate().skip(i) {
                let Some(&next) = self.markers[node].next.get(c) else {
                    break;
                };
                node = next as usize;
                if let Some(id) = self.markers[node].id {
                    found = Some(id);
                    end = cursor + 1;
                }
            }
            let Some(id) = found else {
                i += 1;
                continue;
            };
            while literal < literal_ranges.len() && literal_ranges[literal].end <= i {
                literal += 1;
            }
            if literal < literal_ranges.len() && literal_ranges[literal].start < end {
                i += 1;
                continue;
            }
            ordinary(&mut output, &text[plain_start..i])?;
            if output.len() >= TOKEN_LIMIT {
                return Err(TokenizerError::Encode("tokenizer output exceeds limit"));
            }
            output.push(id);
            i = end;
            plain_start = i;
        }
        ordinary(&mut output, &text[plain_start..])?;
        Ok(output)
    }

    /// `decode_bytes`: the concatenated token bytes, optionally without special added tokens. The C++ default for `skip_special` is `false`.
    pub fn decode_bytes(&self, ids: &[u32], skip_special: bool) -> Result<Vec<u8>, TokenizerError> {
        if ids.len() > TOKEN_LIMIT {
            return Err(TokenizerError::DecodeBytes("decoder token limit exceeded"));
        }
        let mut output = Vec::new();
        for &id in ids {
            let id = id as usize;
            if id >= self.vocabulary.len() {
                return Err(TokenizerError::DecodeBytes("token ID outside vocabulary"));
            }
            if skip_special && self.special[id] {
                continue;
            }
            if self.vocabulary[id].len() > INPUT_LIMIT - output.len() {
                return Err(TokenizerError::DecodeBytes("decoded byte limit exceeded"));
            }
            output.extend_from_slice(&self.vocabulary[id]);
        }
        Ok(output)
    }

    /// `decode_text`: `decode_bytes` with invalid UTF-8 replaced.
    pub fn decode_text(&self, ids: &[u32], skip_special: bool) -> Result<Vec<u8>, TokenizerError> {
        Ok(replace_invalid_utf8(&self.decode_bytes(ids, skip_special)?))
    }

    /// The ID of the added token whose content is exactly `content`, if there is one.
    pub fn added_token(&self, content: &[u8]) -> Option<u32> {
        let mut node = 0usize;
        for c in content {
            node = usize::try_from(*self.markers.get(node)?.next.get(c)?).ok()?;
        }
        self.markers.get(node)?.id
    }

    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary.len()
    }

    pub fn merge_count(&self) -> usize {
        self.merges.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> BpeData {
        let mut vocabulary: Vec<Vec<u8>> = (0..=u8::MAX).map(|byte| vec![byte]).collect();
        for word in [
            "ab", "aa", "aba", "abab", "aaaa", " a", "<X>", "<X>!", "<Y>", "zz",
        ] {
            vocabulary.push(word.as_bytes().to_vec());
        }
        let merges = [
            ("a", "b"),
            ("a", "a"),
            ("ab", "a"),
            ("ab", "ab"),
            ("aa", "aa"),
            (" ", "a"),
        ]
        .map(|(left, right)| (left.as_bytes().to_vec(), right.as_bytes().to_vec()))
        .to_vec();
        let added = [(262, "<X>", true), (263, "<X>!", true), (264, "<Y>", false)]
            .map(|(id, content, special)| AddedToken {
                id,
                content: content.as_bytes().to_vec(),
                special,
            })
            .to_vec();
        BpeData {
            vocabulary,
            merges,
            added,
        }
    }

    #[test]
    fn alphabet_extends_past_latin1_in_byte_order() {
        assert_eq!(CODEPOINTS[0], 256);
        assert_eq!(CODEPOINTS[usize::from(b'\n')], 266);
        assert_eq!(CODEPOINTS[usize::from(b' ')], 288);
        assert_eq!(CODEPOINTS[usize::from(b'!')], 33);
        assert_eq!(CODEPOINTS[usize::from(b'~')], 126);
        assert_eq!(CODEPOINTS[127], 289);
        assert_eq!(CODEPOINTS[160], 322);
        assert_eq!(CODEPOINTS[161], 161);
        assert_eq!(CODEPOINTS[173], 323);
        assert_eq!(CODEPOINTS[255], 255);
        let extensions = CODEPOINTS.iter().filter(|&&point| point >= 256).count();
        assert_eq!(extensions, 324 - 256);
        for (byte, &point) in CODEPOINTS.iter().enumerate() {
            assert_eq!(REVERSE_CODEPOINTS[usize::from(point)], byte as i16);
        }
        let unused = REVERSE_CODEPOINTS.iter().filter(|&&byte| byte < 0).count();
        assert_eq!(unused, 324 - 256);
    }

    #[test]
    fn candidates_pop_by_lowest_rank_then_leftmost() {
        let candidate = |rank, left| Candidate {
            rank,
            left,
            right: left + 1,
            left_version: 0,
            right_version: 0,
            output: 0,
        };
        let mut queue = BinaryHeap::new();
        for (rank, left) in [(2, 0), (1, 5), (1, 3), (0, 9), (2, 1)] {
            queue.push(Reverse(candidate(rank, left)));
        }
        let order: Vec<(u32, u32)> = std::iter::from_fn(|| queue.pop())
            .map(|Reverse(c)| (c.rank, c.left))
            .collect();
        assert_eq!(order, [(0, 9), (1, 3), (1, 5), (2, 0), (2, 1)]);
    }

    #[test]
    fn debug_redacts_token_text_and_ids() {
        let data = fixture();
        assert_eq!(
            format!("{data:?}"),
            "BpeData { vocabulary: 266 tokens, merges: 6 pairs, added: 3 tokens }"
        );
        let marker = AddedToken {
            id: 151_644,
            content: b"<|secret|>".to_vec(),
            special: true,
        };
        let printed = format!("{marker:?}");
        assert_eq!(
            printed,
            "AddedToken { content: 10 bytes, special: true, .. }"
        );
        assert!(!printed.contains("151644") && !printed.contains("secret"));
        let model = ByteBpe::new(data).unwrap();
        let printed = format!("{model:?}");
        assert_eq!(
            printed,
            "ByteBpe { vocabulary: 266 tokens, merges: 6 pairs, markers: 7 nodes }"
        );
        assert!(!format!("{model:#?}").contains("<X>"));
    }

    #[derive(Debug)]
    struct Refusal;

    impl fmt::Display for Refusal {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter.write_str("pretokenizer piece limit exceeded")
        }
    }

    impl std::error::Error for Refusal {}

    #[test]
    fn errors_print_the_cpp_message_or_the_processor_message() {
        let plain: TokenizerError = TokenizerError::New("invalid added token");
        assert_eq!(plain.to_string(), "invalid added token");
        let refused: TokenizerError<Refusal> = TokenizerError::Pieces(Refusal);
        assert_eq!(refused.to_string(), "pretokenizer piece limit exceeded");
        assert!(std::error::Error::source(&refused).is_some());
        let widened: TokenizerError<Refusal> =
            TokenizerError::EncodePiece("BPE piece exceeds limit").widen();
        assert_eq!(widened.to_string(), "BPE piece exceeds limit");
    }

    #[test]
    fn piece_with_no_ranked_pair_keeps_its_singletons() {
        let model = ByteBpe::new(fixture()).unwrap();
        assert_eq!(model.encode_piece(b"z").unwrap(), [122]);
        assert_eq!(model.encode_piece(b"ba").unwrap(), [98, 97]);
    }
}
