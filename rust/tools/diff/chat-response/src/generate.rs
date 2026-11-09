//! Deterministic case generation: every input of `test_chat.cpp`, `test_response.cpp` and `test_response_protocol.cpp` (with every split), hand-picked edges for each refusal, and seeded random cases.

use tokspur_text::response::{ParameterSpec, ParameterType, ResponseLimits, ToolSpec};
use tokspur_text::response_protocol::{RESPONSE_MAXIMUM_IDS, ResponseAdmission};

use crate::case::{Case, Decoder, Labeled, Op};
use crate::fixture::FIXTURE;

/// SplitMix64: small, seedable and stable across platforms.
pub struct Random(u64);

impl Random {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Uniform in `0..n` (`n > 0`), with negligible bias.
    pub fn below(&mut self, n: usize) -> usize {
        let n = u64::try_from(n.max(1)).unwrap_or(u64::MAX);
        usize::try_from(self.next() % n).unwrap_or(0)
    }

    pub fn chance(&mut self, numerator: u64, denominator: u64) -> bool {
        self.next() % denominator < numerator
    }

    pub fn pick<T: Clone>(&mut self, items: &[T]) -> T {
        items[self.below(items.len())].clone()
    }
}

/// A JSON document to write as text, with lexemes and raw splices kept.
#[derive(Clone)]
pub enum J {
    Null,
    Bool(bool),
    Num(String),
    Str(Vec<u8>),
    Arr(Vec<J>),
    Obj(Vec<(Vec<u8>, J)>),
    /// Bytes spliced in verbatim, for malformed documents.
    Raw(Vec<u8>),
}

fn s(text: &str) -> J {
    J::Str(text.as_bytes().to_vec())
}

fn obj(fields: Vec<(&str, J)>) -> J {
    J::Obj(
        fields
            .into_iter()
            .map(|(name, value)| (name.as_bytes().to_vec(), value))
            .collect(),
    )
}

impl J {
    pub fn text(&self, random: &mut Random) -> Vec<u8> {
        let mut out = Vec::new();
        self.write(random, &mut out);
        out
    }

    fn write(&self, random: &mut Random, out: &mut Vec<u8>) {
        match self {
            Self::Null => out.extend_from_slice(b"null"),
            Self::Bool(flag) => out.extend_from_slice(if *flag { b"true" } else { b"false" }),
            Self::Num(lexeme) => out.extend_from_slice(lexeme.as_bytes()),
            Self::Str(text) => write_string(text, random, out),
            Self::Arr(items) => {
                out.push(b'[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    space(random, out);
                    item.write(random, out);
                }
                out.push(b']');
            }
            Self::Obj(fields) => {
                out.push(b'{');
                for (i, (name, value)) in fields.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    space(random, out);
                    write_string(name, random, out);
                    out.push(b':');
                    space(random, out);
                    value.write(random, out);
                }
                out.push(b'}');
            }
            Self::Raw(bytes) => out.extend_from_slice(bytes),
        }
    }
}

fn space(random: &mut Random, out: &mut Vec<u8>) {
    if random.chance(1, 10) {
        out.extend_from_slice(random.pick(&[b" ".as_slice(), b"\n", b"\t", b"\r\n ", b"  "]));
    }
}

fn write_string(text: &[u8], random: &mut Random, out: &mut Vec<u8>) {
    out.push(b'"');
    let escape_unicode = random.chance(1, 8);
    match std::str::from_utf8(text) {
        Ok(text) => {
            for c in text.chars() {
                match c {
                    '"' => out.extend_from_slice(b"\\\""),
                    '\\' => out.extend_from_slice(b"\\\\"),
                    '\n' if random.chance(1, 2) => out.extend_from_slice(b"\\n"),
                    '\t' if random.chance(1, 2) => out.extend_from_slice(b"\\t"),
                    '/' if random.chance(1, 4) => out.extend_from_slice(b"\\/"),
                    c if u32::from(c) < 0x20 => {
                        out.extend_from_slice(format!("\\u{:04x}", u32::from(c)).as_bytes())
                    }
                    c if escape_unicode && !c.is_ascii() => {
                        let mut units = [0u16; 2];
                        for unit in c.encode_utf16(&mut units) {
                            let escaped = if random.chance(1, 2) {
                                format!("\\u{unit:04x}")
                            } else {
                                format!("\\u{unit:04X}")
                            };
                            out.extend_from_slice(escaped.as_bytes());
                        }
                    }
                    c => {
                        let mut buffer = [0u8; 4];
                        out.extend_from_slice(c.encode_utf8(&mut buffer).as_bytes());
                    }
                }
            }
        }
        Err(_) => {
            // Invalid UTF-8 stays raw: both parsers must refuse it alike.
            for &byte in text {
                match byte {
                    b'"' => out.extend_from_slice(b"\\\""),
                    b'\\' => out.extend_from_slice(b"\\\\"),
                    byte if byte < 0x20 => {
                        out.extend_from_slice(format!("\\u{byte:04x}").as_bytes())
                    }
                    byte => out.push(byte),
                }
            }
        }
    }
    out.push(b'"');
}

const PIECES: &[&str] = &[
    "a",
    "b",
    "Z",
    "q",
    "0",
    "7",
    " ",
    "  ",
    "\n",
    "\n\n",
    "\t",
    "\r",
    "\r\n",
    "\u{1}",
    "\u{8}",
    "\u{c}",
    "\u{1c}",
    "\u{1f}",
    "\u{7f}",
    "\"",
    "\\",
    "/",
    "<",
    ">",
    "&",
    "=",
    "|",
    "é",
    "café",
    "日本",
    "🙂",
    "🦉",
    "\u{85}",
    "\u{a0}",
    "\u{1680}",
    "\u{2000}",
    "\u{200a}",
    "\u{200b}",
    "\u{2028}",
    "\u{2029}",
    "\u{202f}",
    "\u{205f}",
    "\u{3000}",
    "\u{feff}",
    "x.rs",
    "true",
    "null",
    "1",
    "-2.5e2",
    "{",
    "}",
    "[",
    "]",
    ",",
    ":",
    "hello world",
    "<tool_call>",
    "</tool_call>",
    "<think>",
    "</think>",
    "<|im_end|>",
    "<|im_start|>",
    "<|endoftext|>",
    "<function=",
    "</function>",
    "<parameter=",
    "</parameter>",
    "<tool_response>",
    "</tool_response>",
    "<tool",
    "</too",
    "<|im",
    "|>",
    "</",
    "\u{0}",
];

/// Text without reserved markers or NUL.
const PLAIN: &[&str] = &[
    "a",
    "b",
    "Z",
    "q",
    "0",
    "7",
    " ",
    "  ",
    "\n",
    "\n\n",
    "\t",
    "\r",
    "\"",
    "\\",
    "/",
    "<",
    ">",
    "&",
    "=",
    "é",
    "café",
    "日本",
    "🙂",
    "🦉",
    "\u{85}",
    "\u{a0}",
    "\u{3000}",
    "\u{feff}",
    "\u{200b}",
    "x.rs",
    "true",
    "null",
    "1",
    "{",
    "}",
    "hello world",
    "<tool",
    "</too",
    "<|im",
    "|>",
    "</",
];

fn text_from(random: &mut Random, pieces: &[&str], max: usize) -> String {
    let count = random.below(max + 1);
    (0..count).map(|_| random.pick(pieces)).collect()
}

fn text(random: &mut Random, max: usize) -> String {
    text_from(random, PIECES, max)
}

fn plain(random: &mut Random, max: usize) -> String {
    text_from(random, PLAIN, max)
}

const SPECIAL_NUMBERS: &[&str] = &[
    "0",
    "-0",
    "0.0",
    "-0.0",
    "1",
    "-1",
    "1e0",
    "1E0",
    "1e+0",
    "1e-0",
    "1e-4",
    "1e-5",
    "0.0001",
    "0.00001",
    "1e15",
    "1e16",
    "9999999999999998",
    "9999999999999998.0",
    "1e15.0",
    "1.5",
    "-2.5e2",
    "-1.25e2",
    "123456789012345678901234567890",
    "1.2345678901234567e+21",
    "5e-324",
    "-5e-324",
    "4e-324",
    "3e-324",
    "2.4703282292062328e-324",
    "2.4703282292062327e-324",
    "2e-324",
    "1e-324",
    "1e-400",
    "-1e-400",
    "1e-9999",
    "-1e-9999",
    "1e9999",
    "-1e9999",
    "1e309",
    "1.7976931348623157e308",
    "1.7976931348623158e308",
    "1.7976931348623159e308",
    "2.2250738585072014e-308",
    "2.2250738585072011e-308",
    "1e-310",
    "0e999999",
    "0.000e-99999999",
    "1e99999999999999999999",
    "1e-99999999999999999999",
    "0.1",
    "0.2",
    "0.3",
    "1e22",
    "1e23",
    "1e21",
    "100",
    "100.0",
    "1e2",
    "123.456",
    "-0e5",
    "0.000000000000000000000000000001",
    "1000000000000000000000000000000.0",
    "4.35",
    "2.675",
    "1.0000000000000002",
    "0.30000000000000004",
    "9007199254740993",
    "9007199254740993.0",
    "18446744073709551615",
    "18446744073709551616",
];

/// A JSON number lexeme from every class the formatters treat differently.
pub fn number(random: &mut Random) -> String {
    let negative = if random.chance(1, 3) { "-" } else { "" };
    match random.below(12) {
        0 => {
            let digits = 1 + random.below(30);
            let mut out = String::from(negative);
            out.push(char::from(
                b'1' + u8::try_from(random.below(9)).unwrap_or(0),
            ));
            for _ in 1..digits {
                out.push(char::from(
                    b'0' + u8::try_from(random.below(10)).unwrap_or(0),
                ));
            }
            out
        }
        1 | 2 => (random.pick(SPECIAL_NUMBERS)).to_owned(),
        3..=6 => {
            let value = loop {
                let value = f64::from_bits(random.next());
                if value.is_finite() {
                    break value;
                }
            };
            match random.below(4) {
                0 => format!("{value:e}"),
                1 => format!("{value:.16e}"),
                2 => format!("{value:?}"),
                _ => format!("{:.*e}", random.below(25), value),
            }
        }
        7 => {
            let value = f64::from(u32::try_from(random.below(1 << 20)).unwrap_or(0)) / 64.0;
            format!("{negative}{value:?}")
        }
        8 => {
            let exponent = [
                -330, -325, -324, -323, -320, -310, -308, -307, -5, -4, 15, 16, 21, 22, 300, 307,
                308, 309,
            ][random.below(18)]
                + i32::try_from(random.below(3)).unwrap_or(0)
                - 1;
            let digits: String = (0..1 + random.below(18))
                .map(|_| char::from(b'0' + u8::try_from(random.below(10)).unwrap_or(0)))
                .collect();
            format!("{negative}{}.{digits}e{exponent}", 1 + random.below(9))
        }
        9 => {
            let whole: String = (0..1 + random.below(3)).map(|_| '0').collect();
            let zeros: String = (0..random.below(400)).map(|_| '0').collect();
            let tail: String = (0..1 + random.below(40))
                .map(|_| char::from(b'0' + u8::try_from(random.below(10)).unwrap_or(0)))
                .collect();
            let whole = if whole.len() > 1 {
                "0".to_owned()
            } else {
                whole
            };
            format!("{negative}{whole}.{zeros}{tail}")
        }
        10 => {
            let digits: String = (0..1 + random.below(60))
                .map(|_| char::from(b'0' + u8::try_from(random.below(10)).unwrap_or(0)))
                .collect();
            let lead = 1 + random.below(9);
            let exponent = i64::try_from(random.below(2000)).unwrap_or(0) - 1000;
            format!(
                "{negative}{lead}.{digits}E{}{exponent}",
                if exponent >= 0 && random.chance(1, 2) {
                    "+"
                } else {
                    ""
                }
            )
        }
        _ => {
            let value = (random.next() >> 11) as f64
                * f64::powi(10.0, i32::try_from(random.below(40)).unwrap_or(0) - 20);
            format!("{negative}{value:e}")
        }
    }
}

/// Doubles whose exact decimal expansion is short, where the two shortest round-trip candidates can tie; written exactly so the parse is exact.
pub fn tie_number(random: &mut Random) -> String {
    let negative = if random.chance(1, 3) { "-" } else { "" };
    match random.below(4) {
        0 | 1 => {
            // m / 2^j: exactly j fractional decimal digits, the last one a 5.
            let m = (1u64 << 52) | (random.next() >> 12);
            let shift = 1 + random.below(12);
            let value = m as f64 / f64::powi(2.0, i32::try_from(shift).unwrap_or(1));
            format!("{negative}{value:.shift$}")
        }
        2 => {
            // Large integers m * 2^j.
            let m = (1u64 << 52) | (random.next() >> 12);
            let value = m as f64 * f64::powi(2.0, i32::try_from(random.below(40)).unwrap_or(0));
            format!("{negative}{value:e}")
        }
        _ => {
            // odd * 5^a * 2^b: midpoints at higher decimal positions.
            let odd = (random.next() >> (40 + random.below(20))) | 1;
            let fives = i32::try_from(random.below(22)).unwrap_or(0);
            let twos = i32::try_from(random.below(80)).unwrap_or(0) - 40;
            let value = odd as f64 * f64::powi(5.0, fives) * f64::powi(2.0, twos);
            format!("{negative}{value:e}")
        }
    }
}

fn json_value(random: &mut Random, depth: usize) -> J {
    match random.below(if depth >= 4 { 4 } else { 7 }) {
        0 => J::Null,
        1 => J::Bool(random.chance(1, 2)),
        2 => J::Num(number(random)),
        3 => J::Str(text(random, 4).into_bytes()),
        4 | 5 => J::Arr(
            (0..random.below(4))
                .map(|_| json_value(random, depth + 1))
                .collect(),
        ),
        _ => J::Obj(
            (0..random.below(4))
                .map(|i| {
                    (
                        format!("{}{i}", plain(random, 2)).into_bytes(),
                        json_value(random, depth + 1),
                    )
                })
                .collect(),
        ),
    }
}

fn deep(depth: usize, leaf: J) -> J {
    (0..depth).fold(leaf, |inner, i| {
        if i % 2 == 0 {
            J::Arr(vec![inner])
        } else {
            obj(vec![("k", inner)])
        }
    })
}

// ---------------------------------------------------------------- chat

pub const TEST_STRINGS: [&str; 4] = ["XHIGH", "LOW", "# Tools\n\n<tools>", "\n\nINSTRUCTIONS"];
pub const ROUND_TRIP_STRINGS: [&str; 4] = [
    "high",
    "low",
    "# Tools\n\n",
    "\n\nIf you choose to call a function ONLY reply",
];

fn strings(values: [&str; 4]) -> [Vec<u8>; 4] {
    values.map(|value| value.as_bytes().to_vec())
}

const FUNCTION_NAMES: &[&str] = &[
    "f",
    "edit.file",
    "read_file",
    "run_tests",
    "a-b_c.9",
    "ping",
    "Z",
];
const BAD_FUNCTION_NAMES: &[&str] = &[
    "bad>name",
    "bad name",
    "",
    "x<y",
    "a\nb",
    "a\rb",
    "日本",
    "semi;colon",
    "a\u{0}b",
];

fn role(random: &mut Random) -> J {
    match random.below(100) {
        0..=29 => s("user"),
        30..=59 => s("assistant"),
        60..=74 => s("tool"),
        75..=86 => s("system"),
        87..=94 => s("developer"),
        95..=97 => s(random.pick(&["other", "", "USER", "function"])),
        _ => random
            .pick(&[J::Null, J::Num("1".to_owned()), J::Bool(true)])
            .clone(),
    }
}

fn content_item(random: &mut Random) -> J {
    let body = J::Str(text(random, 5).into_bytes());
    match random.below(40) {
        0..=19 => obj(vec![("type", s("text")), ("text", body)]),
        20..=29 => obj(vec![("text", body)]),
        30 => obj(vec![("type", s("image")), ("image_url", s("x"))]),
        31 => obj(vec![("image", s("x")), ("text", body)]),
        32 => obj(vec![("video", J::Null)]),
        33 => obj(vec![("type", s("video"))]),
        34 => obj(vec![("type", s("text"))]),
        35 => obj(vec![("type", J::Num("5".to_owned())), ("text", body)]),
        36 => obj(vec![("type", s("audio")), ("text", body)]),
        37 => obj(vec![("text", J::Num("5".to_owned()))]),
        38 => s("bare"),
        _ => obj(vec![
            ("type", s("image_url")),
            ("image_url", obj(vec![("url", s("x"))])),
        ]),
    }
}

fn content(random: &mut Random) -> Option<J> {
    match random.below(40) {
        0..=3 => None,
        4..=7 => Some(J::Null),
        8..=31 => Some(J::Str(text(random, 8).into_bytes())),
        32..=37 => Some(J::Arr(
            (0..random.below(4)).map(|_| content_item(random)).collect(),
        )),
        _ => Some(
            random
                .pick(&[
                    J::Num("3".to_owned()),
                    J::Bool(false),
                    obj(vec![("text", s("x"))]),
                ])
                .clone(),
        ),
    }
}

fn arguments_object(random: &mut Random) -> J {
    let names = ["path", "x", "p", "count", "meta", "日本", "a b", "items"];
    let mut fields = Vec::new();
    for _ in 0..random.below(4) {
        let name = if random.chance(1, 25) {
            (random.pick(&["<x>", "", "a\rb", "a\nb", "a\u{0}b"])).to_owned()
        } else {
            (random.pick(&names)).to_owned()
        };
        if fields
            .iter()
            .any(|(existing, _): &(Vec<u8>, J)| existing == name.as_bytes())
        {
            continue;
        }
        let value = if random.chance(1, 2) {
            J::Str(text(random, 6).into_bytes())
        } else {
            json_value(random, 1)
        };
        fields.push((name.into_bytes(), value));
    }
    J::Obj(fields)
}

fn tool_call(random: &mut Random) -> J {
    let name = match random.below(20) {
        0 => None,
        1 => Some(J::Null),
        2 => Some(J::Num("1".to_owned())),
        3 | 4 => Some(s(random.pick(BAD_FUNCTION_NAMES))),
        5 => Some(J::Str(vec![b'a'; 64 + random.below(2)])),
        _ => Some(s(random.pick(FUNCTION_NAMES))),
    };
    let arguments = match random.below(40) {
        0..=15 => Some(arguments_object(random)),
        16..=27 => {
            let inner = arguments_object(random).text(random);
            let padded = [
                random.pick(&[b"".as_slice(), b" ", b"\n"]),
                inner.as_slice(),
                random.pick(&[b"".as_slice(), b" \t"]),
            ]
            .concat();
            Some(J::Str(padded))
        }
        28..=30 => Some(s(random.pick(&["", " ", "\n\t", "\u{3000}"]))),
        31 | 32 => Some(s(random.pick(&[
            "not JSON",
            "{",
            "{\"a\":1,\"a\":2}",
            "{\"a\":NaN}",
        ]))),
        33 | 34 => Some(s(random.pick(&["[1]", "1", "\"x\"", "null"]))),
        35 => Some(J::Null),
        36 => None,
        37 => Some(J::Num("2".to_owned())),
        38 => Some(J::Arr(vec![J::Null])),
        _ => Some(J::Str(deep(70, J::Null).text(random))),
    };
    let mut function = Vec::new();
    if let Some(name) = name {
        function.push((b"name".to_vec(), name));
    }
    if let Some(arguments) = arguments {
        function.push((b"arguments".to_vec(), arguments));
    }
    if random.chance(7, 10) {
        let mut call = vec![
            (b"id".to_vec(), s("call_1")),
            (b"type".to_vec(), s("function")),
        ];
        let wrapped = if random.chance(1, 30) {
            J::Null
        } else {
            J::Obj(function)
        };
        call.push((b"function".to_vec(), wrapped));
        J::Obj(call)
    } else {
        J::Obj(function)
    }
}

fn message(random: &mut Random) -> J {
    if random.chance(1, 80) {
        return random
            .pick(&[s("user"), J::Null, J::Arr(Vec::new())])
            .clone();
    }
    let mut fields = Vec::new();
    let role = role(random);
    let assistant = matches!(&role, J::Str(name) if name == b"assistant");
    if !random.chance(1, 60) {
        fields.push((b"role".to_vec(), role));
    }
    if let Some(content) = content(random) {
        fields.push((b"content".to_vec(), content));
    }
    if assistant || random.chance(1, 20) {
        if random.chance(1, 2) {
            let reasoning = match random.below(10) {
                0 => J::Null,
                1 => J::Num("1".to_owned()),
                _ => J::Str(text(random, 6).into_bytes()),
            };
            fields.push((b"reasoning_content".to_vec(), reasoning));
        }
        if random.chance(1, 2) {
            let calls = match random.below(12) {
                0 => J::Null,
                1 => s("x"),
                _ => J::Arr((0..random.below(4)).map(|_| tool_call(random)).collect()),
            };
            fields.push((b"tool_calls".to_vec(), calls));
        }
    }
    if random.chance(1, 10) {
        fields.push((b"name".to_vec(), s("n")));
    }
    J::Obj(fields)
}

fn tool_schema(random: &mut Random) -> J {
    if random.chance(1, 30) {
        return random
            .pick(&[s("x"), J::Null, J::Num("1".to_owned())])
            .clone();
    }
    let properties: Vec<(Vec<u8>, J)> = (0..random.below(4))
        .map(|i| {
            (
                format!("p{i}{}", plain(random, 1)).into_bytes(),
                obj(vec![
                    (
                        "type",
                        s(random
                            .pick(&["string", "number", "integer", "object", "array", "boolean"])),
                    ),
                    ("description", J::Str(text(random, 5).into_bytes())),
                    ("default", json_value(random, 2)),
                    ("minimum", J::Num(number(random))),
                ]),
            )
        })
        .collect();
    obj(vec![
        ("type", s("function")),
        (
            "function",
            obj(vec![
                ("name", s(random.pick(FUNCTION_NAMES))),
                ("description", J::Str(text(random, 6).into_bytes())),
                (
                    "parameters",
                    obj(vec![
                        ("type", s("object")),
                        ("properties", J::Obj(properties)),
                        ("required", J::Arr(vec![s("p0")])),
                    ]),
                ),
            ]),
        ),
    ])
}

fn option_bool(random: &mut Random) -> J {
    match random.below(20) {
        0 => J::Null,
        1 => s("true"),
        2 => J::Num("1".to_owned()),
        _ => J::Bool(random.chance(1, 2)),
    }
}

fn valid_content(random: &mut Random) -> J {
    if random.chance(1, 4) {
        let items = (0..1 + random.below(3))
            .map(|_| {
                let body = J::Str(text(random, 4).into_bytes());
                if random.chance(1, 2) {
                    obj(vec![("type", s("text")), ("text", body)])
                } else {
                    obj(vec![("text", body)])
                }
            })
            .collect();
        J::Arr(items)
    } else {
        J::Str(text(random, 8).into_bytes())
    }
}

fn valid_call(random: &mut Random, parse: bool) -> J {
    let arguments = if parse && random.chance(1, 2) {
        J::Str(arguments_object(random).text(random))
    } else if random.chance(1, 8) {
        s(random.pick(&["", " ", "\n"]))
    } else {
        arguments_object(random)
    };
    let function = obj(vec![
        ("name", s(random.pick(FUNCTION_NAMES))),
        ("arguments", arguments),
    ]);
    if random.chance(1, 4) {
        function
    } else {
        obj(vec![
            ("id", s("call_0")),
            ("type", s("function")),
            ("function", function),
        ])
    }
}

/// A chat request every rule admits (bar rare argument-key faults), so the rendered prompt bytes themselves are compared.
pub fn valid_request(random: &mut Random) -> J {
    let hoist = random.chance(1, 3);
    let parse = !random.chance(1, 5);
    let mut messages = Vec::new();
    for _ in 0..random.below(3) {
        let role = random.pick(&["system", "developer"]);
        messages.push(obj(vec![
            ("role", s(role)),
            ("content", valid_content(random)),
        ]));
    }
    for _ in 0..1 + random.below(7) {
        let roles: &[&str] = if hoist {
            &[
                "user",
                "assistant",
                "tool",
                "user",
                "assistant",
                "system",
                "developer",
            ]
        } else {
            &["user", "assistant", "tool", "user", "assistant"]
        };
        let role = random.pick(roles);
        let mut fields = vec![(b"role".to_vec(), s(role))];
        if !random.chance(1, 10) {
            fields.push((b"content".to_vec(), valid_content(random)));
        }
        if role == "assistant" {
            if random.chance(1, 2) {
                let reasoning = if random.chance(1, 6) {
                    J::Null
                } else {
                    J::Str(text(random, 5).into_bytes())
                };
                fields.push((b"reasoning_content".to_vec(), reasoning));
            }
            if random.chance(1, 2) {
                let calls = (0..random.below(3))
                    .map(|_| valid_call(random, parse))
                    .collect();
                fields.push((b"tool_calls".to_vec(), J::Arr(calls)));
            }
        }
        messages.push(J::Obj(fields));
    }
    let mut fields = vec![(b"messages".to_vec(), J::Arr(messages))];
    if hoist {
        fields.push((b"late_system_policy".to_vec(), s("hoist")));
    }
    if !parse {
        fields.push((b"parse_tool_arguments".to_vec(), J::Bool(false)));
    }
    if random.chance(1, 2) {
        let tools = (0..random.below(3))
            .map(|_| tool_schema(random))
            .filter(|tool| matches!(tool, J::Obj(_)))
            .collect();
        fields.push((b"tools".to_vec(), J::Arr(tools)));
    }
    for name in [
        "enable_thinking",
        "preserve_thinking",
        "add_generation_prompt",
    ] {
        if random.chance(1, 2) {
            fields.push((name.as_bytes().to_vec(), J::Bool(random.chance(1, 2))));
        }
    }
    if random.chance(1, 2) {
        let effort = random.pick(&["xhigh", "high", "medium", "low"]);
        fields.push((b"reasoning_effort".to_vec(), s(effort)));
    }
    J::Obj(fields)
}

/// A chat request: mostly valid structure, with an occasional fault.
pub fn request(random: &mut Random) -> J {
    let mut fields: Vec<(Vec<u8>, J)> = Vec::new();
    if !random.chance(1, 60) {
        let messages = match random.below(60) {
            0 => J::Arr(Vec::new()),
            1 => s("x"),
            _ => J::Arr((0..1 + random.below(8)).map(|_| message(random)).collect()),
        };
        fields.push((b"messages".to_vec(), messages));
    }
    if random.chance(1, 3) {
        let tools = match random.below(15) {
            0 => J::Null,
            1 => s("x"),
            2 => J::Arr(Vec::new()),
            _ => J::Arr(
                (0..1 + random.below(3))
                    .map(|_| tool_schema(random))
                    .collect(),
            ),
        };
        fields.push((b"tools".to_vec(), tools));
    }
    for name in [
        "enable_thinking",
        "preserve_thinking",
        "add_generation_prompt",
        "parse_tool_arguments",
    ] {
        if random.chance(1, 3) {
            fields.push((name.as_bytes().to_vec(), option_bool(random)));
        }
    }
    if random.chance(1, 3) {
        let effort = match random.below(14) {
            0 => J::Null,
            1 => J::Num("1".to_owned()),
            n => s([
                "xhigh", "high", "medium", "low", "bogus", "", "HIGH", "minimal", "xhigh", "low",
                "medium", "high",
            ][n - 2]),
        };
        fields.push((b"reasoning_effort".to_vec(), effort));
    }
    if random.chance(1, 3) {
        let policy = match random.below(10) {
            0 => J::Null,
            1 => s("other"),
            2..=5 => s("reject"),
            _ => s("hoist"),
        };
        fields.push((b"late_system_policy".to_vec(), policy));
    }
    if random.chance(1, 40) {
        fields.push((b"temperature".to_vec(), J::Num("0.5".to_owned())));
    }
    if random.chance(1, 4) {
        let len = fields.len();
        for i in (1..len).rev() {
            let j = random.below(i + 1);
            fields.swap(i, j);
        }
    }
    if random.chance(1, 100) {
        return random.pick(&[J::Arr(Vec::new()), s("x"), J::Null]).clone();
    }
    J::Obj(fields)
}

// ---------------------------------------------------------------- parser

pub fn test_tools() -> Vec<ToolSpec> {
    vec![
        ToolSpec::new(
            "edit.file",
            vec![
                ParameterSpec::new("path", ParameterType::RawString, true),
                ParameterSpec::new("patch", ParameterType::RawString, false),
                ParameterSpec::new("apply", ParameterType::Boolean, false),
                ParameterSpec::new("count", ParameterType::Number, false),
                ParameterSpec::new("meta", ParameterType::Object, false),
                ParameterSpec::new("items", ParameterType::Array, false),
                ParameterSpec::new("empty", ParameterType::NullValue, false),
            ],
        ),
        ToolSpec::new("ping", Vec::new()),
    ]
}

const TYPES: [ParameterType; 6] = [
    ParameterType::RawString,
    ParameterType::Boolean,
    ParameterType::Number,
    ParameterType::Object,
    ParameterType::Array,
    ParameterType::NullValue,
];

fn random_tools(random: &mut Random) -> Vec<ToolSpec> {
    let names: &[&[u8]] = &[
        b"f",
        b"g",
        b"edit.file",
        b"ping",
        b"a-b_c.9",
        b"bad name",
        b"x<",
        b"\x01",
        b"\xc3",
        b"\xc3\xa9",
        b"",
    ];
    let parameters: &[&[u8]] = &[
        b"path",
        b"x",
        b"y",
        b"\xe6\x97\xa5",
        b"a b",
        b"\x7f",
        b"<p",
        b"",
        b"\xff",
        b"q",
    ];
    (0..random.below(4))
        .map(|_| {
            let mut name = random.pick(names).to_vec();
            if random.chance(1, 40) {
                name = vec![b'a'; 65];
            }
            let parameters = (0..random.below(4))
                .map(|_| {
                    let mut parameter = random.pick(parameters).to_vec();
                    if random.chance(1, 60) {
                        parameter = vec![b'p'; 1025];
                    }
                    ParameterSpec::new(parameter, random.pick(&TYPES), random.chance(1, 3))
                })
                .collect();
            ToolSpec::new(name, parameters)
        })
        .collect()
}

fn random_limits(random: &mut Random) -> ResponseLimits {
    match random.below(10) {
        0..=6 => ResponseLimits::default(),
        7 | 8 => {
            let bytes = 1 + random.below(600);
            ResponseLimits {
                bytes,
                call_bytes: 1 + random.below(bytes),
                records: 1 + random.below(6),
                calls: 1 + random.below(3),
                json_nodes: 1 + random.below(20),
                json_depth: 1 + random.below(5),
            }
        }
        _ => {
            let mut limits = ResponseLimits::default();
            match random.below(7) {
                0 => limits.bytes = 0,
                1 => limits.bytes = 8 * 1024 * 1024 + 1,
                2 => limits.call_bytes = limits.bytes + 1,
                3 => limits.records = 65537,
                4 => limits.calls = 129,
                5 => limits.json_nodes = 0,
                _ => limits.json_depth = 65,
            }
            limits
        }
    }
}

fn ws(random: &mut Random) -> &'static str {
    random.pick(&["", "", "", "\n", " ", "\n\n", "\t", "\r\n"])
}

fn typed_value(random: &mut Random, kind: ParameterType) -> String {
    match kind {
        ParameterType::RawString => {
            if random.chance(1, 10) {
                text(random, 5)
            } else {
                plain(random, 6)
            }
        }
        ParameterType::Boolean => (random.pick(&[
            "true",
            "false",
            " true ",
            "\ttrue\n",
            "tru",
            "1",
            "\"true\"",
            "null",
            "true false",
        ]))
        .to_owned(),
        ParameterType::Number => {
            if random.chance(1, 8) {
                (random.pick(&[
                    "NaN", "1e999", "-1e999", "1e-400", "0e999", " 5 ", "+1", "01", "1.", "-",
                ]))
                .to_owned()
            } else {
                number(random)
            }
        }
        ParameterType::Object | ParameterType::Array => {
            let value = match random.below(10) {
                0 => J::Raw(b"{\"a\":1,\"a\":2}".to_vec()),
                1 => deep(random.below(8), J::Num(number(random))),
                2 => J::Raw(b"[1,]".to_vec()),
                3..=5 => J::Arr(
                    (0..random.below(4))
                        .map(|_| json_value(random, 1))
                        .collect(),
                ),
                _ => J::Obj(
                    (0..random.below(4))
                        .map(|i| (format!("k{i}").into_bytes(), json_value(random, 1)))
                        .collect(),
                ),
            };
            String::from_utf8_lossy(&value.text(random)).into_owned()
        }
        ParameterType::NullValue => {
            (random.pick(&["null", " null\n", "nul", "None", "[]"])).to_owned()
        }
    }
}

fn call_frame(random: &mut Random, tools: &[ToolSpec]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"<tool_call>");
    out.extend_from_slice(ws(random).as_bytes());
    if random.chance(1, 40) {
        out.extend_from_slice(b"{\"name\":\"ping\",\"arguments\":{}}");
    } else {
        out.extend_from_slice(b"<function=");
        let tool = if tools.is_empty() || random.chance(1, 20) {
            None
        } else {
            Some(random.pick(tools))
        };
        match &tool {
            Some(tool) => out.extend_from_slice(&tool.name),
            None => out.extend_from_slice(random.pick(&[
                b"unknown".as_slice(),
                b"ping bad",
                b"",
                b"a<b",
            ])),
        }
        if !random.chance(1, 50) {
            out.push(b'>');
        }
        out.extend_from_slice(ws(random).as_bytes());
        if let Some(tool) = tool {
            let mut order: Vec<&ParameterSpec> = tool
                .parameters
                .iter()
                .filter(|_| random.chance(2, 3))
                .collect();
            if random.chance(1, 4) {
                order.reverse();
            }
            if random.chance(1, 20) && !order.is_empty() {
                order.push(order[0]);
            }
            for parameter in order {
                out.extend_from_slice(b"<parameter=");
                out.extend_from_slice(&parameter.name);
                out.push(b'>');
                if !random.chance(1, 40) {
                    out.push(b'\n');
                }
                out.extend_from_slice(typed_value(random, parameter.r#type).as_bytes());
                if !random.chance(1, 40) {
                    out.extend_from_slice(b"\n</parameter>");
                }
                out.extend_from_slice(ws(random).as_bytes());
            }
            if random.chance(1, 30) {
                out.extend_from_slice(b"<parameter=unknown>\nx\n</parameter>");
            }
        }
        if !random.chance(1, 30) {
            out.extend_from_slice(b"</function>");
        }
        out.extend_from_slice(ws(random).as_bytes());
        if random.chance(1, 40) {
            out.extend_from_slice(b"junk");
        }
    }
    if !random.chance(1, 30) {
        out.extend_from_slice(b"</tool_call>");
    }
    out
}

const BAD_BYTES: &[&[u8]] = &[
    b"\x00",
    b"\x80",
    b"\xbf",
    b"\xc0\x80",
    b"\xc1\xbf",
    b"\xc3",
    b"\xc3\x28",
    b"\xe2\x82",
    b"\xe0\x80\x80",
    b"\xed\xa0\x80",
    b"\xef\xbf\xbf",
    b"\xf0\x90\x80",
    b"\xf4\x90\x80\x80",
    b"\xf5\x80\x80\x80",
    b"\xff",
];

/// A generated assistant stream for the given manifest.
pub fn stream(random: &mut Random, tools: &[ToolSpec], thinking: bool) -> Vec<u8> {
    let mut out = Vec::new();
    if thinking {
        out.extend_from_slice(text(random, 6).as_bytes());
        if random.chance(9, 10) {
            out.extend_from_slice(b"</think>");
        }
    } else if random.chance(1, 4) {
        out.extend_from_slice(b"<think>");
        out.extend_from_slice(plain(random, 4).as_bytes());
        out.extend_from_slice(b"</think>");
    }
    for _ in 0..random.below(5) {
        match random.below(10) {
            0..=3 => out.extend_from_slice(plain(random, 5).as_bytes()),
            4..=7 => out.extend_from_slice(&call_frame(random, tools)),
            8 => out.extend_from_slice(ws(random).as_bytes()),
            _ => out.extend_from_slice(text(random, 3).as_bytes()),
        }
    }
    if random.chance(3, 4) {
        out.extend_from_slice(b"<|im_end|>");
        if random.chance(1, 6) {
            out.extend_from_slice(random.pick(&[
                b"\n".as_slice(),
                b" \n\t",
                b"suffix",
                b"<|im_end|>",
            ]));
        }
    }
    if random.chance(1, 12) && !out.is_empty() {
        let at = random.below(out.len() + 1);
        let bad = random.pick(BAD_BYTES);
        out.splice(at..at, bad.iter().copied());
    }
    if random.chance(1, 12) && !out.is_empty() {
        let at = random.below(out.len());
        out[at] = u8::try_from(random.below(256)).unwrap_or(0);
    }
    if random.chance(1, 10) && !out.is_empty() {
        out.truncate(random.below(out.len()));
    }
    out
}

fn split_ops(random: &mut Random, bytes: &[u8]) -> Vec<Op> {
    let mut cuts: Vec<usize> = (0..random.below(4))
        .map(|_| random.below(bytes.len() + 1))
        .collect();
    cuts.sort_unstable();
    let mut ops = Vec::new();
    let mut start = 0;
    for cut in cuts {
        ops.push(Op::Feed(bytes[start..cut].to_vec()));
        start = cut;
    }
    ops.push(Op::Feed(bytes[start..].to_vec()));
    if random.chance(19, 20) {
        ops.push(Op::Finish);
    }
    if random.chance(1, 6) {
        ops.push(Op::Feed(
            random.pick(&[b"".as_slice(), b"more", b"\n"]).to_vec(),
        ));
        ops.push(Op::Finish);
    }
    ops
}

fn call(value: &str) -> Vec<u8> {
    format!("<tool_call>\n<function=edit.file>\n<parameter=path>\n{value}\n</parameter>\n</function>\n</tool_call>")
        .into_bytes()
}

fn parser_case(tools: Vec<ToolSpec>, thinking: bool, limits: ResponseLimits, ops: Vec<Op>) -> Case {
    Case::Parser {
        tools,
        thinking,
        limits,
        ops,
    }
}

fn push(cases: &mut Vec<Labeled>, label: impl Into<String>, case: Case) {
    cases.push(Labeled {
        label: label.into(),
        case,
    });
}

/// The rendered prompt body from `test_response.cpp`'s owned round trip.
fn round_trip_body() -> Vec<u8> {
    [
        b"<think>\ninert <tool_call>\n</think>\n\n".as_slice(),
        b"<tool_call>\n<function=edit.file>\n<parameter=path>\nsrc/hello.rs\n</parameter>\n",
        b"<parameter=apply>\ntrue\n</parameter>\n<parameter=meta>\n{\"version\": 2}\n</parameter>\n",
        b"</function>\n</tool_call><|im_end|>\n",
    ]
    .concat()
}

fn test_response_cases(cases: &mut Vec<Labeled>) {
    let inert = [b"I considered ".as_slice(), &call("x.rs"), b"."].concat();
    let stream = [
        inert.as_slice(),
        "</think>\n\nEditing 日本🙂\n".as_bytes(),
        &call("src/日本🙂.rs"),
        b"\n<tool_call>\n<function=ping>\n</function>\n</tool_call><|im_end|>\n",
    ]
    .concat();
    for split in 0..=stream.len() {
        let ops = vec![
            Op::Feed(stream[..split].to_vec()),
            Op::Feed(stream[split..].to_vec()),
            Op::Finish,
            Op::Finish,
        ];
        push(
            cases,
            format!("test_response stream split {split}"),
            parser_case(test_tools(), true, ResponseLimits::default(), ops),
        );
    }
    let repeated = [call("x.rs").as_slice(), b"\n", &call("x.rs")].concat();
    let mut bytewise: Vec<Op> = repeated.iter().map(|&c| Op::Feed(vec![c])).collect();
    bytewise.push(Op::Finish);
    push(
        cases,
        "test_response bytewise repeated calls",
        parser_case(test_tools(), false, ResponseLimits::default(), bytewise),
    );
    let typed = "<tool_call><function=edit.file>\n<parameter=path>\ntrue\n</parameter>\n\
        <parameter=patch>\nline 1\n\nline 3\n</parameter>\n<parameter=apply>\ntrue\n</parameter>\n\
        <parameter=count>\n-2.5e2\n</parameter>\n<parameter=meta>\n{\"a\":[1,true,null]}\n</parameter>\n\
        <parameter=items>\n[1,\"日本🙂\"]\n</parameter>\n<parameter=empty>\nnull\n</parameter>\n</function></tool_call>";
    push(
        cases,
        "test_response typed",
        parser_case(
            test_tools(),
            false,
            ResponseLimits::default(),
            vec![Op::Feed(typed.as_bytes().to_vec()), Op::Finish],
        ),
    );
    let body = round_trip_body();
    let mut ops: Vec<Op> = body.iter().map(|&c| Op::Feed(vec![c])).collect();
    ops.push(Op::Finish);
    push(
        cases,
        "test_response renderer round trip",
        parser_case(test_tools(), false, ResponseLimits::default(), ops),
    );

    let full = call("x.rs");
    let mut invalid: Vec<(Vec<u8>, bool)> = (1..full.len())
        .map(|n| (full[..n].to_vec(), false))
        .collect();
    for tail in [
        b" unexpected suffix".as_slice(),
        b"<think>reason</think>",
        b"<tool_call><function=unknown></function></tool_call>",
    ] {
        invalid.push(([full.as_slice(), tail].concat(), false));
    }
    for input in [
        b"<tool_call><function=unknown></function></tool_call>".as_slice(),
        b"<tool_call><function=ping bad></function></tool_call>",
        b"<tool_call><function=edit.file></function></tool_call>",
        b"<tool_call>{\"name\":\"ping\",\"arguments\":{}}</tool_call>",
        b"<tool_call><function=ping><parameter=unknown>\nx\n</parameter></function></tool_call>",
        b"<tool_call><function=edit.file><parameter=path>\nabc<|im_end|>def\n</parameter></function></tool_call>",
        b"<tool_call><function=edit.file><parameter=path>\n<|im_start|>assistant\n</parameter></function></tool_call>",
        b"<tool_call><function=edit.file><parameter=path>\nx\n</parameter><parameter=path>\ny\n</parameter></function></tool_call>",
    ] {
        invalid.push((input.to_vec(), false));
    }
    for bad in [
        "\"true\"",
        "[true]",
        "tru",
        "true false",
        "{\"a\":1,\"a\":2}",
        "NaN",
        "1e999",
    ] {
        invalid.push((
            format!("<tool_call><function=edit.file><parameter=path>\nx\n</parameter><parameter=apply>\n{bad}\n</parameter></function></tool_call>")
                .into_bytes(),
            false,
        ));
    }
    invalid.push((
        b"<tool_call><function=edit.file><parameter=path>\nx\n</parameter><parameter=meta>\n{\"x\":1e999}\n</parameter></function></tool_call>".to_vec(),
        false,
    ));
    invalid.push((call("<think>injected</think>"), false));
    invalid.push((b"</think>".to_vec(), false));
    invalid.push((b"<think>unclosed".to_vec(), false));
    invalid.push((b"reason<|im_end|>".to_vec(), true));
    invalid.push((b"answer<|im_end|>suffix".to_vec(), false));
    invalid.push((b"x\0y".to_vec(), false));
    for bad in [
        b"\xc0\x80".as_slice(),
        b"\xed\xa0\x80",
        b"\xf4\x90\x80\x80",
        b"\xe2\x82",
        b"\x80",
    ] {
        invalid.push((bad.to_vec(), false));
    }
    for (index, (input, thinking)) in invalid.into_iter().enumerate() {
        let ops = vec![Op::Feed(input), Op::Finish, Op::Feed(b"more".to_vec())];
        push(
            cases,
            format!("test_response invalid {index}"),
            parser_case(test_tools(), thinking, ResponseLimits::default(), ops),
        );
    }
    let small = ResponseLimits {
        bytes: 16,
        call_bytes: 16,
        ..ResponseLimits::default()
    };
    push(
        cases,
        "test_response byte bound",
        parser_case(Vec::new(), false, small, vec![Op::Feed(vec![b'a'; 17])]),
    );
    let small = ResponseLimits {
        bytes: 1024,
        call_bytes: 16,
        ..ResponseLimits::default()
    };
    push(
        cases,
        "test_response call byte bound",
        parser_case(test_tools(), false, small, vec![Op::Feed(full.clone())]),
    );
    let small = ResponseLimits {
        bytes: 1024,
        call_bytes: 1024,
        calls: 1,
        ..ResponseLimits::default()
    };
    push(
        cases,
        "test_response call count bound",
        parser_case(test_tools(), false, small, vec![Op::Feed(repeated.clone())]),
    );
    let small = ResponseLimits {
        bytes: 1024,
        call_bytes: 1024,
        calls: 2,
        records: 1,
        ..ResponseLimits::default()
    };
    push(
        cases,
        "test_response record bound",
        parser_case(
            test_tools(),
            false,
            small,
            vec![Op::Feed([b"text".as_slice(), &full].concat())],
        ),
    );
    let configurations = [
        vec![
            ToolSpec::new("a", Vec::new()),
            ToolSpec::new("a", Vec::new()),
        ],
        vec![ToolSpec::new(
            "a",
            vec![
                ParameterSpec::new("x", ParameterType::Number, false),
                ParameterSpec::new("x", ParameterType::Number, false),
            ],
        )],
        vec![ToolSpec::new("bad name", Vec::new())],
    ];
    for (index, tools) in configurations.into_iter().enumerate() {
        push(
            cases,
            format!("test_response rejected {index}"),
            parser_case(tools, false, ResponseLimits::default(), Vec::new()),
        );
    }
    let zero = ResponseLimits {
        bytes: 0,
        ..ResponseLimits::default()
    };
    push(
        cases,
        "test_response rejected limits",
        parser_case(Vec::new(), false, zero, Vec::new()),
    );
    // Every split of every C++ test stream, through finish.
    for (name, input, thinking) in [
        ("repeated", repeated.clone(), false),
        ("typed", typed.as_bytes().to_vec(), false),
        ("round trip", body, false),
    ] {
        for split in 0..=input.len() {
            let ops = vec![
                Op::Feed(input[..split].to_vec()),
                Op::Feed(input[split..].to_vec()),
                Op::Finish,
            ];
            push(
                cases,
                format!("test_response {name} split {split}"),
                parser_case(test_tools(), thinking, ResponseLimits::default(), ops),
            );
        }
    }
}

fn parser_edges(cases: &mut Vec<Labeled>) {
    let edges: &[(&[u8], bool)] = &[
        (b"", false),
        (b"", true),
        (b"<", false),
        (b"<thi", false),
        (b"<|im_end", false),
        (b"</think", true),
        (b"plain <tool", false),
        (b"<function=x>", false),
        (b"</function>", false),
        (b"<parameter=x>", false),
        (b"</parameter>", false),
        (b"</tool_call>", false),
        (b"<think></think><think></think>", false),
        (b"visible<think>late</think>", false),
        (b" \n<think>x</think>", false),
        (b"<think>a</think>b<|im_end|>\n \t", false),
        (b"<|im_end|><|im_end|>", false),
        (b"<tool_call><function=ping></function></tool_call> \n<|im_end|>", false),
        (b"<tool_call><function=ping></function></tool_call>\n<tool_call><function=ping></function></tool_call>", false),
        (b"<tool_call><function=ping></function> trailing</tool_call>", false),
        (b"<tool_call> <function=ping>\n</function>\n </tool_call>", false),
        (b"<tool_call><function=ping</function></tool_call>", false),
        (b"<tool_call><function=edit.file><parameter=path>x\n</parameter></function></tool_call>", false),
        (b"<tool_call><function=edit.file><parameter=path\nx\n</parameter></function></tool_call>", false),
        (b"<tool_call><function=edit.file><parameter=path>\n\n</parameter></function></tool_call>", false),
        (b"<tool_call><function=edit.file><parameter=path>\nx</parameter>\n</parameter></function></tool_call>", false),
        (b"<tool_call><function=edit.file><parameter=path>\nx\n</parameter><parameter=count>\n1e-400\n</parameter></function></tool_call>", false),
        (b"<tool_call><function=edit.file><parameter=path>\nx\n</parameter><parameter=count>\n5e-324\n</parameter></function></tool_call>", false),
        (b"<tool_call><function=edit.file><parameter=path>\nx\n</parameter><parameter=items>\n[1e-400]\n</parameter></function></tool_call>", false),
        (b"<tool_call><function=edit.file><parameter=path>\nx\n</parameter><parameter=meta>\n{\"a\":{\"b\":[1,2,{\"c\":-0}]}}\n</parameter></function></tool_call>", false),
        (b"<tool_call><function=edit.file><parameter=path>\n\xc3\xa9\n</parameter><parameter=\xc3\xa9>\nx\n</parameter></function></tool_call>", false),
        (b"<tool_call><function=\xc3\xa9></function></tool_call>", false),
        (b"<tool_call><function=e\x01></function></tool_call>", false),
        (b"<tool_call><function=e\x7f></function></tool_call>", false),
        (b"text\xc3", false),
        (b"text\xf0\x9f\x99", false),
        (b"\xf0\x9f\x99\x82<think>", false),
        (b"<tool_call><function=ping><parameter=x</tool_call>", false),
        (b"<tool_call><function=edit.file><parameter=path>\nx\n</parameter><parameter=count>\n\"\\ud800\"\n</parameter></function></tool_call>", false),
    ];
    for (index, (input, thinking)) in edges.iter().enumerate() {
        for split in 0..=input.len() {
            let ops = vec![
                Op::Feed(input[..split].to_vec()),
                Op::Feed(input[split..].to_vec()),
                Op::Finish,
                Op::Feed(Vec::new()),
            ];
            push(
                cases,
                format!("parser edge {index} split {split}"),
                parser_case(test_tools(), *thinking, ResponseLimits::default(), ops),
            );
        }
    }
    // Deep JSON values against the parser's own nesting limits and dump's.
    for depth in [30, 31, 32, 33, 62, 63, 64, 65] {
        let value =
            String::from_utf8_lossy(&deep(depth, J::Num("1".to_owned())).text(&mut Random::new(0)))
                .into_owned();
        let kind = if depth % 2 == 0 { "items" } else { "meta" };
        let frame = format!(
            "<tool_call><function=edit.file><parameter=path>\nx\n</parameter><parameter={kind}>\n{value}\n</parameter></function></tool_call>"
        );
        for json_depth in [32, 64] {
            let limits = ResponseLimits {
                json_depth,
                ..ResponseLimits::default()
            };
            push(
                cases,
                format!("parser deep {depth} limit {json_depth}"),
                parser_case(
                    test_tools(),
                    false,
                    limits,
                    vec![Op::Feed(frame.clone().into_bytes()), Op::Finish],
                ),
            );
        }
    }
    // A manifest at each schema bound.
    let wide: Vec<ParameterSpec> = (0..256)
        .map(|i| ParameterSpec::new(format!("p{i}"), ParameterType::RawString, false))
        .collect();
    let mut too_wide = wide.clone();
    too_wide.push(ParameterSpec::new("p256", ParameterType::RawString, false));
    let many: Vec<ToolSpec> = (0..512)
        .map(|i| ToolSpec::new(format!("t{i}"), Vec::new()))
        .collect();
    let mut too_many = many.clone();
    too_many.push(ToolSpec::new("t512", Vec::new()));
    let heavy: Vec<ToolSpec> = (0..5)
        .map(|t| {
            ToolSpec::new(
                format!("t{t}"),
                (0..256)
                    .map(|i| {
                        ParameterSpec::new(format!("{i:0>1000}"), ParameterType::RawString, false)
                    })
                    .collect(),
            )
        })
        .collect();
    for (label, tools) in [
        ("256 parameters", vec![ToolSpec::new("w", wide)]),
        ("257 parameters", vec![ToolSpec::new("w", too_wide)]),
        ("512 tools", many),
        ("513 tools", too_many),
        ("schema bytes", heavy),
        (
            "64-byte name",
            vec![ToolSpec::new(vec![b'n'; 64], Vec::new())],
        ),
        (
            "65-byte name",
            vec![ToolSpec::new(vec![b'n'; 65], Vec::new())],
        ),
        (
            "1024-byte parameter",
            vec![ToolSpec::new(
                "w",
                vec![ParameterSpec::new(
                    vec![b'p'; 1024],
                    ParameterType::Number,
                    true,
                )],
            )],
        ),
        (
            "1025-byte parameter",
            vec![ToolSpec::new(
                "w",
                vec![ParameterSpec::new(
                    vec![b'p'; 1025],
                    ParameterType::Number,
                    true,
                )],
            )],
        ),
    ] {
        push(
            cases,
            format!("parser manifest {label}"),
            parser_case(
                tools,
                false,
                ResponseLimits::default(),
                vec![Op::Feed(b"ok".to_vec()), Op::Finish],
            ),
        );
    }
    // Byte, call and record bounds at their edges.
    let ping = b"<tool_call><function=ping></function></tool_call>".to_vec();
    for bytes in [ping.len() - 1, ping.len(), ping.len() + 1] {
        for call_bytes in [bytes.min(37), bytes.min(38), bytes.min(39), bytes] {
            let limits = ResponseLimits {
                bytes,
                call_bytes,
                ..ResponseLimits::default()
            };
            push(
                cases,
                format!("parser bounds {bytes}/{call_bytes}"),
                parser_case(
                    test_tools(),
                    false,
                    limits,
                    vec![Op::Feed(ping.clone()), Op::Finish],
                ),
            );
        }
    }
    for records in 1..5 {
        for calls in 1..4 {
            let limits = ResponseLimits {
                records,
                calls,
                ..ResponseLimits::default()
            };
            let input = [
                b"a<think>".as_slice(),
                b"t</think>",
                &ping,
                b"\n",
                &ping,
                b"\n",
                &ping,
            ]
            .concat();
            push(
                cases,
                format!("parser records {records} calls {calls}"),
                parser_case(
                    test_tools(),
                    false,
                    limits,
                    vec![Op::Feed(input), Op::Finish],
                ),
            );
        }
    }
    let long_call = [
        b"<tool_call><function=edit.file><parameter=path>\n".as_slice(),
        &vec![b'x'; 70_000],
        b"\n</parameter></function></tool_call>",
    ]
    .concat();
    push(
        cases,
        "parser call over 64 KiB",
        parser_case(
            test_tools(),
            false,
            ResponseLimits::default(),
            vec![Op::Feed(long_call), Op::Finish],
        ),
    );
    push(
        cases,
        "parser byte limit across feeds",
        parser_case(
            Vec::new(),
            false,
            ResponseLimits::default(),
            vec![
                Op::Feed(vec![b'a'; 1024 * 1024 - 1]),
                Op::Feed(b"bc".to_vec()),
                Op::Feed(b"b".to_vec()),
            ],
        ),
    );
}

fn random_parser_cases(cases: &mut Vec<Labeled>, random: &mut Random, count: usize) {
    for index in 0..count {
        let tools = if random.chance(4, 5) {
            test_tools()
        } else {
            random_tools(random)
        };
        let thinking = random.chance(1, 3);
        let limits = random_limits(random);
        let bytes = stream(random, &tools, thinking);
        let ops = if random.chance(1, 10) {
            let mut ops: Vec<Op> = bytes.iter().map(|&c| Op::Feed(vec![c])).collect();
            ops.push(Op::Finish);
            ops
        } else {
            split_ops(random, &bytes)
        };
        push(
            cases,
            format!("parser random {index}"),
            parser_case(tools, thinking, limits, ops),
        );
    }
}

// ---------------------------------------------------------------- protocol

pub const ADMISSION: ResponseAdmission = ResponseAdmission {
    vocabulary: 259,
    im_end: 256,
    end_of_text: 257,
    im_start: 258,
};

fn ids_json(ids: &[u32]) -> J {
    J::Arr(ids.iter().map(|id| J::Num(id.to_string())).collect())
}

fn protocol_manifest() -> J {
    J::Arr(vec![
        obj(vec![
            ("name", s("read_file")),
            (
                "parameters",
                J::Arr(vec![obj(vec![
                    ("name", s("path")),
                    ("type", s("string")),
                    ("required", J::Bool(true)),
                ])]),
            ),
        ]),
        obj(vec![
            ("name", s("run_tests")),
            ("parameters", J::Arr(Vec::new())),
        ]),
    ])
}

fn envelope(ids: &[u32], finish: &str, thinking: bool, schema: J, stop: J) -> J {
    obj(vec![
        ("ids", ids_json(ids)),
        ("tools", schema),
        ("initial_thinking", J::Bool(thinking)),
        ("finish_reason", s(finish)),
        ("stop_id", stop),
    ])
}

fn protocol(
    cases: &mut Vec<Labeled>,
    label: impl Into<String>,
    admission: ResponseAdmission,
    envelope: &J,
    decoder: Decoder,
) {
    let text = envelope.text(&mut Random::new(7));
    push(
        cases,
        label,
        Case::Protocol {
            admission,
            envelope: text,
            decoder,
        },
    );
}

fn test_protocol_cases(cases: &mut Vec<Labeled>) {
    let encode = |text: &[u8]| FIXTURE.encode(text);
    let call = "<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/日本🙂.rs\n</parameter>\n</function>\n</tool_call>";
    let batch = format!(
        "Inspecting café\n{call}\n<tool_call><function=run_tests>\n</function></tool_call>"
    );
    let all = encode(format!("{batch}<|im_end|>").as_bytes());
    let stop = || J::Num("256".to_owned());
    let fixture = Decoder::Fixture;
    protocol(
        cases,
        "test_protocol complete",
        ADMISSION,
        &envelope(&all, "stop", false, protocol_manifest(), stop()),
        fixture,
    );
    for split in 0..all.len() {
        protocol(
            cases,
            format!("test_protocol length split {split}"),
            ADMISSION,
            &envelope(&all[..split], "length", false, protocol_manifest(), J::Null),
            fixture,
        );
        protocol(
            cases,
            format!("test_protocol error split {split}"),
            ADMISSION,
            &envelope(&all[..split], "error", false, protocol_manifest(), J::Null),
            fixture,
        );
    }
    protocol(
        cases,
        "test_protocol capped call",
        ADMISSION,
        &envelope(
            &encode(call.as_bytes()),
            "length",
            false,
            protocol_manifest(),
            J::Null,
        ),
        fixture,
    );
    let hidden = format!("considered {call} but this is inert.");
    protocol(
        cases,
        "test_protocol inert thinking",
        ADMISSION,
        &envelope(
            &encode(format!("{hidden}</think>\n\n{call}<|im_end|>").as_bytes()),
            "stop",
            true,
            protocol_manifest(),
            stop(),
        ),
        fixture,
    );
    protocol(
        cases,
        "test_protocol plain",
        ADMISSION,
        &envelope(
            &encode("finished café 🦉<|im_end|>".as_bytes()),
            "stop",
            false,
            J::Arr(Vec::new()),
            stop(),
        ),
        fixture,
    );
    let typed = J::Arr(vec![obj(vec![
        ("name", s("typed")),
        (
            "parameters",
            J::Arr(
                ["string", "boolean", "number", "object", "array", "null"]
                    .iter()
                    .zip(["s", "b", "n", "o", "a", "z"])
                    .map(|(kind, name)| {
                        obj(vec![
                            ("name", s(name)),
                            ("type", s(kind)),
                            ("required", J::Bool(true)),
                        ])
                    })
                    .collect(),
            ),
        ),
    ])]);
    let typed_body = encode(b"<tool_call><function=typed>\n<parameter=s>\ntrue\n</parameter>\n<parameter=b>\ntrue\n</parameter>\n<parameter=n>\n-2.5\n</parameter>\n<parameter=o>\n{\"x\":1}\n</parameter>\n<parameter=a>\n[1,true]\n</parameter>\n<parameter=z>\nnull\n</parameter>\n</function></tool_call><|im_end|>");
    protocol(
        cases,
        "test_protocol typed",
        ADMISSION,
        &envelope(&typed_body, "stop", false, typed, stop()),
        fixture,
    );
    let invalid = [
        envelope(&[], "stop", false, protocol_manifest(), stop()),
        envelope(&all, "stop", false, protocol_manifest(), J::Null),
        envelope(
            &all,
            "stop",
            false,
            protocol_manifest(),
            J::Num("257".to_owned()),
        ),
        envelope(&all, "length", false, protocol_manifest(), J::Null),
        envelope(
            &all,
            "stop",
            false,
            protocol_manifest(),
            J::Num("999999".to_owned()),
        ),
        envelope(
            &encode(call.as_bytes()),
            "stop",
            false,
            protocol_manifest(),
            stop(),
        ),
        envelope(&all, "unknown", false, protocol_manifest(), stop()),
        envelope(
            &encode(b"x<|endoftext|>"),
            "stop",
            false,
            protocol_manifest(),
            stop(),
        ),
        envelope(
            &encode(b"<|im_start|>x<|im_end|>"),
            "stop",
            false,
            protocol_manifest(),
            stop(),
        ),
        envelope(
            &encode(b"x<|im_end|>y<|im_end|>"),
            "stop",
            false,
            protocol_manifest(),
            stop(),
        ),
        envelope(
            &encode(format!("{call}unexpected<|im_end|>").as_bytes()),
            "stop",
            false,
            protocol_manifest(),
            stop(),
        ),
        envelope(
            &encode(b"<think>unfinished<|im_end|>"),
            "stop",
            false,
            protocol_manifest(),
            stop(),
        ),
        envelope(
            &encode(b"<tool_call><function=read_file></function></tool_call><|im_end|>"),
            "stop",
            false,
            protocol_manifest(),
            stop(),
        ),
        envelope(
            &encode(b"<tool_call><function=unknown></function></tool_call><|im_end|>"),
            "stop",
            false,
            protocol_manifest(),
            stop(),
        ),
    ];
    for (index, input) in invalid.iter().enumerate() {
        protocol(
            cases,
            format!("test_protocol invalid {index}"),
            ADMISSION,
            input,
            fixture,
        );
    }
    let mut earlier = all.clone();
    earlier.insert(0, 256);
    protocol(
        cases,
        "test_protocol earlier im_end",
        ADMISSION,
        &envelope(&earlier, "stop", false, protocol_manifest(), stop()),
        fixture,
    );
    protocol(
        cases,
        "test_protocol outside vocabulary",
        ADMISSION,
        &envelope(&[259], "length", false, protocol_manifest(), J::Null),
        fixture,
    );
    protocol(
        cases,
        "test_protocol ID count",
        ADMISSION,
        &envelope(
            &vec![65; RESPONSE_MAXIMUM_IDS + 1],
            "length",
            false,
            protocol_manifest(),
            J::Null,
        ),
        fixture,
    );
    protocol(
        cases,
        "protocol ID count at bound",
        ADMISSION,
        &envelope(
            &vec![65; RESPONSE_MAXIMUM_IDS],
            "length",
            false,
            protocol_manifest(),
            J::Null,
        ),
        fixture,
    );
    for (index, schema) in [
        r#"[{"name":"x","parameters":[],"extra":1}]"#,
        r#"[{"name":"x","parameters":[{"name":"p","type":"string"}]}]"#,
        r#"[{"name":"x","parameters":[{"name":"p","type":"string","required":true,"extra":1}]}]"#,
        r#"[{"name":"x","parameters":[{"name":"p","type":"integer","required":true}]}]"#,
        r#"[{"name":"x","parameters":[{"name":"p","type":"string","required":"true"}]}]"#,
        r#"[{"name":"x","parameters":[]},{"name":"x","parameters":[]}]"#,
        r#"[{"name":"bad name","parameters":[]}]"#,
    ]
    .iter()
    .enumerate()
    {
        protocol(
            cases,
            format!("test_protocol schema {index}"),
            ADMISSION,
            &envelope(
                &all,
                "stop",
                false,
                J::Raw(schema.as_bytes().to_vec()),
                stop(),
            ),
            fixture,
        );
    }
    let J::Obj(mut extra) = envelope(&all, "stop", false, protocol_manifest(), stop()) else {
        return;
    };
    extra.push((b"unknown".to_vec(), J::Bool(true)));
    protocol(
        cases,
        "test_protocol extra field",
        ADMISSION,
        &J::Obj(extra),
        fixture,
    );
    protocol(
        cases,
        "test_protocol ids only",
        ADMISSION,
        &J::Raw(br#"{"ids":[]}"#.to_vec()),
        fixture,
    );
    for (index, admission) in [
        ResponseAdmission {
            vocabulary: 0,
            im_end: 0,
            end_of_text: 1,
            im_start: 2,
        },
        ResponseAdmission {
            vocabulary: 259,
            im_end: 256,
            end_of_text: 256,
            im_start: 258,
        },
        ResponseAdmission {
            vocabulary: 259,
            im_end: 256,
            end_of_text: 257,
            im_start: 259,
        },
        ResponseAdmission {
            vocabulary: 1_048_577,
            im_end: 256,
            end_of_text: 257,
            im_start: 258,
        },
        ResponseAdmission {
            vocabulary: 1_048_576,
            im_end: 256,
            end_of_text: 257,
            im_start: 258,
        },
        ResponseAdmission {
            vocabulary: 259,
            im_end: 256,
            end_of_text: 258,
            im_start: 258,
        },
    ]
    .iter()
    .enumerate()
    {
        protocol(
            cases,
            format!("test_protocol admission {index}"),
            *admission,
            &envelope(&all, "stop", false, protocol_manifest(), stop()),
            fixture,
        );
    }
    protocol(
        cases,
        "test_protocol upstream error",
        ADMISSION,
        &envelope(
            &encode(call.as_bytes()),
            "error",
            false,
            protocol_manifest(),
            J::Null,
        ),
        fixture,
    );
    protocol(
        cases,
        "test_protocol wrong terminal",
        ADMISSION,
        &envelope(&all, "stop", false, protocol_manifest(), stop()),
        Decoder::WrongTerminal,
    );
    protocol(
        cases,
        "test_protocol oversized decode",
        ADMISSION,
        &envelope(&[], "length", false, protocol_manifest(), J::Null),
        Decoder::Oversized,
    );
    protocol(
        cases,
        "protocol missing decoder",
        ADMISSION,
        &envelope(&all, "stop", false, protocol_manifest(), stop()),
        Decoder::Missing,
    );
    protocol(
        cases,
        "protocol missing decoder error",
        ADMISSION,
        &envelope(&all[..3], "error", false, protocol_manifest(), J::Null),
        Decoder::Missing,
    );
    protocol(
        cases,
        "test_protocol invalid UTF-8 lead",
        ADMISSION,
        &envelope(
            &encode(b"x\xc0\x80<|im_end|>"),
            "stop",
            false,
            protocol_manifest(),
            stop(),
        ),
        fixture,
    );
    protocol(
        cases,
        "test_protocol invalid UTF-8 continuation",
        ADMISSION,
        &envelope(
            &encode(b"x\xc3<|im_end|>"),
            "stop",
            false,
            protocol_manifest(),
            stop(),
        ),
        fixture,
    );
    protocol(
        cases,
        "test_protocol partial UTF-8",
        ADMISSION,
        &envelope(
            &encode(b"x\xc3"),
            "length",
            false,
            protocol_manifest(),
            J::Null,
        ),
        fixture,
    );
    let wide = ResponseAdmission {
        vocabulary: 300,
        ..ADMISSION
    };
    protocol(
        cases,
        "protocol decoder refusal",
        wide,
        &envelope(&[65, 280], "length", false, protocol_manifest(), J::Null),
        fixture,
    );
    for (index, stop_id) in [
        "256.0",
        "-256",
        "2.56e2",
        "\"256\"",
        "true",
        "18446744073709551872",
        "256e0",
    ]
    .iter()
    .enumerate()
    {
        protocol(
            cases,
            format!("protocol stop lexeme {index}"),
            ADMISSION,
            &envelope(
                &all,
                "stop",
                false,
                protocol_manifest(),
                J::Raw(stop_id.as_bytes().to_vec()),
            ),
            fixture,
        );
    }
    for (index, id) in [
        "65.0",
        "-1",
        "\"65\"",
        "null",
        "4294967296",
        "1e2",
        "18446744073709551616",
    ]
    .iter()
    .enumerate()
    {
        let ids = J::Raw(format!("[66,{id}]").into_bytes());
        let input = obj(vec![
            ("ids", ids),
            ("tools", protocol_manifest()),
            ("initial_thinking", J::Bool(false)),
            ("finish_reason", s("length")),
            ("stop_id", J::Null),
        ]);
        protocol(
            cases,
            format!("protocol id lexeme {index}"),
            ADMISSION,
            &input,
            fixture,
        );
    }
    // Field order and kinds: the first refusal must match.
    let fields = [
        ("ids", ids_json(&all)),
        ("tools", protocol_manifest()),
        ("initial_thinking", J::Bool(false)),
        ("finish_reason", s("stop")),
        ("stop_id", stop()),
    ];
    for missing in 0..fields.len() {
        let kept: Vec<(&str, J)> = fields
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != missing)
            .map(|(_, (k, v))| (*k, v.clone()))
            .collect();
        protocol(
            cases,
            format!("protocol missing field {missing}"),
            ADMISSION,
            &obj(kept),
            fixture,
        );
        for wrong in [
            J::Null,
            s("x"),
            J::Num("1".to_owned()),
            J::Arr(Vec::new()),
            obj(Vec::new()),
        ] {
            let mut changed: Vec<(&str, J)> = fields.iter().map(|(k, v)| (*k, v.clone())).collect();
            changed[missing].1 = wrong;
            protocol(
                cases,
                format!("protocol wrong field {missing}"),
                ADMISSION,
                &obj(changed),
                fixture,
            );
        }
    }
    // Both constructor arguments refused: clang evaluates them left to right.
    protocol(
        cases,
        "protocol tools and thinking both refused",
        ADMISSION,
        &obj(vec![
            ("ids", ids_json(&all)),
            ("tools", s("x")),
            ("finish_reason", s("stop")),
            ("stop_id", stop()),
        ]),
        fixture,
    );
    protocol(
        cases,
        "protocol bad tools and wrong thinking",
        ADMISSION,
        &obj(vec![
            ("ids", ids_json(&all)),
            (
                "tools",
                J::Raw(br#"[{"name":"bad name","parameters":[]}]"#.to_vec()),
            ),
            ("initial_thinking", s("no")),
            ("finish_reason", s("stop")),
            ("stop_id", stop()),
        ]),
        fixture,
    );
    protocol(
        cases,
        "protocol unknown type and wrong thinking",
        ADMISSION,
        &obj(vec![("ids", ids_json(&all)), ("tools", J::Raw(br#"[{"name":"x","parameters":[{"name":"p","type":"integer","required":true}]}]"#.to_vec())), ("initial_thinking", J::Null), ("finish_reason", s("stop")), ("stop_id", stop())]),
        fixture,
    );
    protocol(
        cases,
        "protocol name and required both refused",
        ADMISSION,
        &obj(vec![
            ("ids", ids_json(&all)),
            (
                "tools",
                J::Raw(
                    br#"[{"name":"x","parameters":[{"type":"string","name":1,"required":"x"}]}]"#
                        .to_vec(),
                ),
            ),
            ("initial_thinking", J::Bool(false)),
            ("finish_reason", s("stop")),
            ("stop_id", stop()),
        ]),
        fixture,
    );
    let tools: Vec<J> = (0..65)
        .map(|i| {
            obj(vec![
                ("name", s(&format!("t{i}"))),
                ("parameters", J::Arr(Vec::new())),
            ])
        })
        .collect();
    protocol(
        cases,
        "protocol 65 tools",
        ADMISSION,
        &envelope(&all, "stop", false, J::Arr(tools[..65].to_vec()), stop()),
        fixture,
    );
    protocol(
        cases,
        "protocol 64 tools",
        ADMISSION,
        &envelope(
            &encode(b"ok<|im_end|>"),
            "stop",
            false,
            J::Arr(tools[..64].to_vec()),
            stop(),
        ),
        fixture,
    );
    let parameters: Vec<J> = (0..33)
        .map(|i| {
            obj(vec![
                ("name", s(&format!("p{i}"))),
                ("type", s("string")),
                ("required", J::Bool(false)),
            ])
        })
        .collect();
    for count in [32, 33] {
        let schema = J::Arr(vec![obj(vec![
            ("name", s("w")),
            ("parameters", J::Arr(parameters[..count].to_vec())),
        ])]);
        protocol(
            cases,
            format!("protocol {count} parameters"),
            ADMISSION,
            &envelope(&encode(b"ok<|im_end|>"), "stop", false, schema, stop()),
            fixture,
        );
    }
}

fn manifest_json(tools: &[ToolSpec]) -> J {
    J::Arr(
        tools
            .iter()
            .map(|tool| {
                J::Obj(vec![
                    (b"name".to_vec(), J::Str(tool.name.clone())),
                    (
                        b"parameters".to_vec(),
                        J::Arr(
                            tool.parameters
                                .iter()
                                .map(|parameter| {
                                    let kind = match parameter.r#type {
                                        ParameterType::RawString => "string",
                                        ParameterType::Boolean => "boolean",
                                        ParameterType::Number => "number",
                                        ParameterType::Object => "object",
                                        ParameterType::Array => "array",
                                        ParameterType::NullValue => "null",
                                    };
                                    J::Obj(vec![
                                        (b"name".to_vec(), J::Str(parameter.name.clone())),
                                        (b"type".to_vec(), s(kind)),
                                        (b"required".to_vec(), J::Bool(parameter.required)),
                                    ])
                                })
                                .collect(),
                        ),
                    ),
                ])
            })
            .collect(),
    )
}

/// A generated protocol envelope.
pub fn random_envelope(random: &mut Random) -> (ResponseAdmission, J, Decoder) {
    let tools = if random.chance(3, 4) {
        test_tools()
    } else {
        random_tools(random)
    };
    let thinking = random.chance(1, 3);
    let bytes = stream(random, &tools, thinking);
    let mut ids = FIXTURE.encode(&bytes);
    let finish = match random.below(20) {
        0..=10 => "stop",
        11..=16 => "length",
        17 | 18 => "error",
        _ => random.pick(&["", "STOP", "max_tokens"]),
    };
    if finish == "stop" && ids.last() != Some(&256) && random.chance(9, 10) {
        ids.push(256);
    }
    if random.chance(1, 20) && !ids.is_empty() {
        let at = random.below(ids.len());
        ids[at] = random.pick(&[256, 257, 258, 259, 4_000_000]);
    }
    let stop = match (finish, random.below(20)) {
        ("stop", 0) => J::Null,
        ("stop", 1) => J::Num("257".to_owned()),
        ("stop", 2) => J::Raw(b"\"256\"".to_vec()),
        ("stop", _) => J::Num("256".to_owned()),
        (_, 0) => J::Num("256".to_owned()),
        _ => J::Null,
    };
    let schema = if random.chance(1, 30) {
        random
            .pick(&[J::Null, s("x"), J::Arr(vec![J::Null])])
            .clone()
    } else {
        manifest_json(&tools)
    };
    let mut fields = vec![
        (b"ids".to_vec(), ids_json(&ids)),
        (b"tools".to_vec(), schema),
        (b"initial_thinking".to_vec(), J::Bool(thinking)),
        (b"finish_reason".to_vec(), s(finish)),
        (b"stop_id".to_vec(), stop),
    ];
    if random.chance(1, 5) {
        for i in (1..fields.len()).rev() {
            let j = random.below(i + 1);
            fields.swap(i, j);
        }
    }
    if random.chance(1, 40) {
        let at = random.below(fields.len());
        fields.remove(at);
    }
    if random.chance(1, 40) {
        fields.push((b"extra".to_vec(), J::Null));
    }
    let admission = match random.below(40) {
        0 => ResponseAdmission {
            vocabulary: 300,
            ..ADMISSION
        },
        1 => ResponseAdmission {
            im_start: 256,
            ..ADMISSION
        },
        _ => ADMISSION,
    };
    let decoder = match random.below(40) {
        0 => Decoder::Missing,
        1 => Decoder::WrongTerminal,
        2 => Decoder::Oversized,
        _ => Decoder::Fixture,
    };
    (admission, J::Obj(fields), decoder)
}

// ---------------------------------------------------------------- chat cases

fn test_chat_cases(cases: &mut Vec<Labeled>) {
    for number in [
        "-0",
        "0.0",
        "-0.0",
        "1e0",
        "1e-4",
        "1e-5",
        "1e15",
        "1e16",
        "-1.25e2",
        "5e-324",
        "1e-9999",
        "-1e-9999",
        "123456789012345678901234567890",
        "1e9999",
    ] {
        push(
            cases,
            format!("test_chat number {number}"),
            Case::TemplateJson {
                json: number.as_bytes().to_vec(),
            },
        );
    }
    push(
        cases,
        "test_chat escaping",
        Case::TemplateJson {
            json: r#"{"z":[true,null,"<é>&\n\t\u0000"],"a":1.0}"#.as_bytes().to_vec(),
        },
    );
    let requests = [
        r#"{"messages":[{"role":"user","content":" hi "}],"enable_thinking":false}"#,
        r#"{"messages":[{"role":"system","content":" A "},{"role":"developer","content":[{"text":" B "}]},{"role":"user","content":null}],"reasoning_effort":"low","add_generation_prompt":false}"#,
        r#"{"messages":[{"role":"assistant","content":" old ","reasoning_content":" hide "},{"role":"user","content":"next"},{"role":"assistant","content":" new ","reasoning_content":" keep "}],"preserve_thinking":false,"reasoning_effort":"medium","add_generation_prompt":false}"#,
        r#"{"messages":[{"role":"user","content":"Q"},{"role":"assistant","content":"","tool_calls":[{"id":"x","function":{"name":"f","arguments":"{\"x\":1.0,\"p\":\"é\"}"}}]},{"role":"tool","content":"one"},{"role":"tool","content":[{"text":"two"}]},{"role":"user","content":"<tool_response>\nthree\n</tool_response>"}],"reasoning_effort":"medium"}"#,
        r#"{"messages":[{"role":"tool","content":"x"}],"enable_thinking":false}"#,
        r#"{"messages":[{"role":"system","content":"first"},{"role":"user","content":"Q"},{"role":"developer","content":"second"},{"role":"assistant","content":"A"},{"role":"system","content":"third"}],"late_system_policy":"hoist","enable_thinking":false}"#,
        r#"{"messages":[]}"#,
        r#"{"messages":[{"role":"other"}]}"#,
        r#"{"messages":[{"role":"user","content":"Q"},{"role":"system","content":"late"}]}"#,
        r#"{"messages":[{"role":"user","content":[{"type":"image","image_url":"x"}]}]}"#,
        r#"{"messages":[{"role":"user","content":"Q"}],"reasoning_effort":"bogus"}"#,
        r#"{"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"bad>name","arguments":{}}}]}]}"#,
        r#"{"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":"not JSON"}}]}]}"#,
        r#"{"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":"{\"x\":1}"}}],"parse_tool_arguments":false}"#,
        r#"{"messages":[{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":"{\"x\":1}"}}]}],"parse_tool_arguments":false}"#,
    ];
    for (index, request) in requests.iter().enumerate() {
        push(
            cases,
            format!("test_chat request {index}"),
            Case::Render {
                strings: strings(TEST_STRINGS),
                request: request.as_bytes().to_vec(),
            },
        );
    }
    push(
        cases,
        "test_response round trip render",
        Case::Render {
            strings: strings(ROUND_TRIP_STRINGS),
            request: br#"{"messages":[{"role":"assistant","content":"","reasoning_content":"inert <tool_call>","tool_calls":[{"function":{"name":"edit.file","arguments":{"path":"src/hello.rs","apply":true,"meta":{"version":2}}}}]}],"enable_thinking":false,"add_generation_prompt":false}"#.to_vec(),
        },
    );
    push(
        cases,
        "test_chat literals",
        Case::Extract {
            source: br##"{% set x = 'Reasoning effort is set to xhigh. X' %}
{% set l = 'Reasoning effort is set to low. L' %}
{{ '# Tools\n\nT' }} {{ '\n\nIf you choose to call a function ONLY reply X' }}
{# 'ignored unmatched quote #}"##
                .to_vec(),
        },
    );
    push(
        cases,
        "test_chat missing profile",
        Case::Extract {
            source: b"'missing profile'".to_vec(),
        },
    );
    push(
        cases,
        "test_chat bad escape",
        Case::Extract {
            source: b"'\\x00'".to_vec(),
        },
    );
    for (index, text) in [" \u{1c}\u{85}\u{3000}x\u{a0}", " \u{feff}x\u{200b} "]
        .iter()
        .enumerate()
    {
        push(
            cases,
            format!("test_chat trim {index}"),
            Case::Trim {
                text: text.as_bytes().to_vec(),
            },
        );
    }
}

fn chat_edges(cases: &mut Vec<Labeled>) {
    let base = |messages: &str| format!(r#"{{"messages":{messages}}}"#);
    let mut requests: Vec<String> = vec![
        r#"[]"#.to_owned(),
        "{}".to_owned(),
        r#"{"messages":null}"#.to_owned(),
        r#"{"messages":[{"role":"user","content":"x"}],"tools":[1]}"#.to_owned(),
        r#"{"messages":[{"role":"user","content":"x"}],"tools":{}}"#.to_owned(),
        r#"{"messages":[{"role":"user","content":"x"}],"tools":null,"reasoning_effort":"high"}"#.to_owned(),
        r#"{"messages":[{"role":"user","content":"x"}],"tools":[],"reasoning_effort":"low"}"#.to_owned(),
        r#"{"messages":[{"role":"user","content":"x"}],"enable_thinking":false,"reasoning_effort":5}"#.to_owned(),
        r#"{"messages":[{"role":"user","content":"x"}],"enable_thinking":null}"#.to_owned(),
        r#"{"messages":[{"role":"user","content":"x"}],"late_system_policy":5}"#.to_owned(),
        r#"{"messages":[{"role":"user","content":"x"}],"add_generation_prompt":"no"}"#.to_owned(),
        r#"{"messages":[{"role":"user","content":"x"}],"extra":1}"#.to_owned(),
        r#"{"messages":[{"role":"system","content":""},{"role":"system","content":"  "},{"role":"user","content":"x"}]}"#.to_owned(),
        r#"{"messages":[{"role":"system","content":"s"}],"enable_thinking":false,"add_generation_prompt":false}"#.to_owned(),
        r#"{"messages":[{"role":"system","content":"s"}],"tools":[{"type":"function","function":{"name":"f","parameters":{"x":1.5e300,"y":-0.0,"z":[1e-7,123456789.0]}}}]}"#.to_owned(),
        r#"{"messages":[{"role":"user","content":"x"}],"tools":[{"a":1e400}]}"#.to_owned(),
        r#"{"messages":[{"role":"user","content":"x"}],"tools":[{"a":1e-400,"b":-1e-400}]}"#.to_owned(),
        r#"{"messages":[{"role":"user","content":"x"}],"tools":[{"a":"\u0001\u001f\u007f\b\f\"\\/"}]}"#.to_owned(),
        r#"{"messages":[{"role":"tool","content":"a"},{"role":"tool","content":"b"},{"role":"user","content":"q"},{"role":"tool","content":"c"}]}"#.to_owned(),
        r#"{"messages":[{"role":"system","content":"s"},{"role":"tool","content":"a"}]}"#.to_owned(),
        r#"{"messages":[{"role":"user","content":"<tool_response>x</tool_response>"},{"role":"assistant","content":"a","reasoning_content":"r"}],"preserve_thinking":false}"#.to_owned(),
        r#"{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a","reasoning_content":5}],"preserve_thinking":false}"#.to_owned(),
        r#"{"messages":[{"role":"assistant","content":"a","reasoning_content":5},{"role":"user","content":"q"}],"preserve_thinking":false}"#.to_owned(),
        r#"{"messages":[{"role":"assistant","content":"t","tool_calls":[{"name":"f","arguments":{"a":"x","b":[1,{"c":null}],"c":2.50}},{"function":{"name":"g"}}]}]}"#.to_owned(),
        r#"{"messages":[{"role":"assistant","content":"","tool_calls":[{"function":{"name":"f","arguments":"  "}}]}],"parse_tool_arguments":false}"#.to_owned(),
        r#"{"messages":[{"role":"assistant","content":"","tool_calls":[{"function":{"name":"f","arguments":5}}]}]}"#.to_owned(),
        r#"{"messages":[{"role":"assistant","content":"","tool_calls":[{"function":{"name":null}}]}]}"#.to_owned(),
        r#"{"messages":[{"role":"assistant","content":"","tool_calls":[{"function":{"name":"f","arguments":{"<k>":1}}}]}]}"#.to_owned(),
        r#"{"messages":[{"role":"assistant","content":"","tool_calls":[{"function":{"name":"f","arguments":"[1]"}}]}]}"#.to_owned(),
        r#"{"messages":[{"role":"assistant","content":"","tool_calls":[{"function":null}]}]}"#.to_owned(),
        r#"{"messages":[{"role":"assistant","content":"","tool_calls":"x"}]}"#.to_owned(),
        r#"{"messages":[{"role":"assistant","content":"","tool_calls":[5]}]}"#.to_owned(),
        r#"{"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"text":"b"}]}]}"#.to_owned(),
        r#"{"messages":[{"role":"user","content":[{"type":5}]}]}"#.to_owned(),
        r#"{"messages":[{"role":"user","content":[{"type":"text"}]}]}"#.to_owned(),
        r#"{"messages":[{"role":"user","content":[5]}]}"#.to_owned(),
        r#"{"messages":[{"role":"user","content":5}]}"#.to_owned(),
        r#"{"messages":[{"role":"user","content":[{"video":1,"type":5}]}]}"#.to_owned(),
        r#"{"messages":[{"role":"user","content":[{"type":"video"}]}]}"#.to_owned(),
        r#"{"messages":[{"content":"x"}]}"#.to_owned(),
        r#"{"messages":[{"role":1}]}"#.to_owned(),
        r#"{"messages":[5]}"#.to_owned(),
        r#"{"messages":[{"role":"user","content":"x"}],"late_system_policy":"hoist"}"#.to_owned(),
        r#"{"messages":[{"role":"developer","content":"d"},{"role":"user","content":"x"},{"role":"system","content":""}],"late_system_policy":"hoist"}"#.to_owned(),
    ];
    requests.push(base(&format!(
        "[{}]",
        vec![r#"{"role":"user","content":"x"}"#; 4096].join(",")
    )));
    requests.push(base(&format!(
        "[{}]",
        vec![r#"{"role":"user","content":"x"}"#; 4097].join(",")
    )));
    let calls = |count: usize| {
        format!(
            "[{}]",
            vec![r#"{"function":{"name":"f","arguments":{}}}"#; count].join(",")
        )
    };
    for count in [128, 129] {
        requests.push(base(&format!(
            r#"[{{"role":"assistant","content":"","tool_calls":{}}}]"#,
            calls(count)
        )));
        requests.push(format!(r#"{{"messages":[{{"role":"assistant","content":"","tool_calls":{}}}],"parse_tool_arguments":false}}"#, calls(count)));
    }
    let parameters = |count: usize| {
        (0..count)
            .map(|i| format!(r#""p{i}":{i}"#))
            .collect::<Vec<_>>()
            .join(",")
    };
    for count in [256, 257] {
        requests.push(base(&format!(r#"[{{"role":"assistant","content":"","tool_calls":[{{"function":{{"name":"f","arguments":{{{}}}}}}}]}}]"#, parameters(count))));
    }
    let tools = |count: usize| format!("[{}]", vec!["{}"; count].join(","));
    for count in [512, 513] {
        requests.push(format!(
            r#"{{"messages":[{{"role":"user","content":"x"}}],"tools":{}}}"#,
            tools(count)
        ));
    }
    for (name, length) in [("f", 64), ("f", 65)] {
        let long = name.repeat(length);
        requests.push(base(&format!(r#"[{{"role":"assistant","content":"","tool_calls":[{{"function":{{"name":"{long}"}}}}]}}]"#)));
    }
    for length in [1024, 1025] {
        let key = "k".repeat(length);
        requests.push(base(&format!(r#"[{{"role":"assistant","content":"","tool_calls":[{{"function":{{"name":"f","arguments":{{"{key}":1}}}}}}]}}]"#)));
    }
    for digits in [4299, 4300, 4301] {
        let number = "9".repeat(digits);
        requests.push(format!(
            r#"{{"messages":[{{"role":"user","content":"x"}}],"tools":[{{"n":{number}}}]}}"#
        ));
        requests.push(format!(
            r#"{{"messages":[{{"role":"user","content":"x"}}],"tools":[{{"n":-{number}}}]}}"#
        ));
    }
    // Nesting: request depth stays within parse_json's 64; tool arguments parsed from a string add their own depth (dump refuses beyond 64).
    for depth in [55, 58, 59, 60, 61, 62] {
        let deep_value =
            String::from_utf8_lossy(&deep(depth, J::Num("1".to_owned())).text(&mut Random::new(1)))
                .into_owned();
        // Inside a JSON string: no raw whitespace, quotes escaped.
        let escaped = deep_value
            .replace(['\n', '\t', '\r', ' '], "")
            .replace('"', "\\\"");
        requests.push(base(&format!(r#"[{{"role":"assistant","content":"","tool_calls":[{{"function":{{"name":"f","arguments":"{{\"a\":{escaped}}}"}}}}]}}]"#)));
        requests.push(format!(
            r#"{{"messages":[{{"role":"user","content":"x"}}],"tools":[{{"a":{deep_value}}}]}}"#
        ));
    }
    // Output bounds: 8 MiB of rendered text from the request's own bytes.
    let big = "a".repeat(4 * 1024 * 1024);
    requests.push(base(&format!(
        r#"[{{"role":"user","content":"{big}"}},{{"role":"user","content":"{big}"}}]"#
    )));
    requests.push(base(&format!(
        r#"[{{"role":"user","content":"{}"}}]"#,
        "a".repeat(8 * 1024 * 1024 - 64)
    )));
    requests.push(base(&format!(
        r#"[{{"role":"user","content":[{{"text":"{big}"}},{{"text":"{big}"}}]}}]"#
    )));
    for (index, request) in requests.into_iter().enumerate() {
        push(
            cases,
            format!("render edge {index}"),
            Case::Render {
                strings: strings(TEST_STRINGS),
                request: request.into_bytes(),
            },
        );
    }
    for (index, values) in [
        ["", "b", "c", "d"],
        ["a", "", "c", "d"],
        ["a", "b", "", "d"],
        ["a", "b", "c", ""],
    ]
    .iter()
    .enumerate()
    {
        push(
            cases,
            format!("render profile {index}"),
            Case::Render {
                strings: strings(*values),
                request: br#"{"messages":[{"role":"user","content":"x"}]}"#.to_vec(),
            },
        );
    }
    for (index, source) in [
        b"".as_slice(),
        b"{#",
        b"{# x",
        b"{#}",
        b"{##}",
        b"'a",
        b"'a\\",
        b"\"a\\\"b\"",
        b"'\\n\\r\\t\\\\\\'\\\"'",
        b"'\\u0041'",
        b"'Reasoning effort is set to xhigh.' 'Reasoning effort is set to xhigh. again'",
        b"'Reasoning effort is set to xhigh.' \"Reasoning effort is set to low.\" '# Tools\n\nT' '\n\nIf you choose to call a function ONLY reply'",
        b"'Reasoning effort is set to xhigh.' 'Reasoning effort is set to low.' '# Tools\\n\\nT' '\\n\\nIf you choose to call a function ONLY reply' {# 'Reasoning effort is set to low.' #}",
    ]
    .iter()
    .enumerate()
    {
        push(cases, format!("extract edge {index}"), Case::Extract { source: source.to_vec() });
    }
    push(
        cases,
        "extract over 64 KiB",
        Case::Extract {
            source: vec![b' '; 64 * 1024 + 1],
        },
    );
    push(
        cases,
        "extract at 64 KiB",
        Case::Extract {
            source: vec![b' '; 64 * 1024],
        },
    );
    for (index, text) in [
        b"".as_slice(),
        b" ",
        b"\t\n\x0b\x0c\r\x1c\x1d\x1e\x1f ",
        "\u{85}\u{a0}\u{1680}\u{2000}\u{2001}\u{2009}\u{200a}\u{2028}\u{2029}\u{202f}\u{205f}\u{3000}".as_bytes(),
        "\u{180e}x\u{200b}\u{2060}\u{feff}".as_bytes(),
        b"\xc3",
        b" \xc3\xa9 ",
        b"\xed\xa0\x80",
        b"\xf4\x90\x80\x80",
        b"\xe0\x80\x80",
        b"a\x00b",
    ]
    .iter()
    .enumerate()
    {
        push(cases, format!("trim edge {index}"), Case::Trim { text: text.to_vec() });
    }
    push(
        cases,
        "trim at the Unicode work limit",
        Case::Trim {
            text: vec![b' '; 8 * 1024 * 1024],
        },
    );
    push(
        cases,
        "trim over the Unicode work limit",
        Case::Trim {
            text: vec![b' '; 8 * 1024 * 1024 + 1],
        },
    );
    for (index, text) in [
        "\"\\ud800\"",
        "\"\\ud800x\"",
        "\"\\ud800\\u0041\"",
        "\"\\udc00\"",
        "\"\\ud83e\\udd89\"",
        "\"\\uD83E\\uDD89\"",
        "\"\\u00e9\\u0000\\u001f\"",
        r#"[1,]"#,
        r#"{"a":1,"a":2}"#,
        "\"\u{7f}\"",
    ]
    .iter()
    .enumerate()
    {
        push(
            cases,
            format!("template_json JSON edge {index}"),
            Case::TemplateJson {
                json: text.as_bytes().to_vec(),
            },
        );
    }
}

fn tree_count(out: &mut Vec<u8>, count: usize) {
    out.extend_from_slice(&u32::try_from(count).unwrap_or(u32::MAX).to_le_bytes());
}

/// String bytes for a constructed value: sometimes no longer valid UTF-8.
fn tree_string(text: &[u8], random: &mut Random, out: &mut Vec<u8>) {
    let mut bytes = text.to_vec();
    if random.chance(1, 25) {
        let at = random.below(bytes.len() + 1);
        let bad = random.pick(BAD_BYTES);
        bytes.splice(at..at, bad.iter().copied());
    }
    tree_count(out, bytes.len());
    out.extend_from_slice(&bytes);
}

/// Encodes `value` for the driver's `tree`, the way only constructed values can be: invalid UTF-8 strings and keys, duplicate keys, nesting past 64.
pub fn encode_tree(value: &J, random: &mut Random, out: &mut Vec<u8>) {
    let wrap = matches!(value, J::Arr(_) | J::Obj(_)) && random.chance(1, 150);
    let levels = if wrap { 60 + random.below(10) } else { 0 };
    for _ in 0..levels {
        out.push(4);
        tree_count(out, 1);
    }
    match value {
        J::Null | J::Raw(_) => out.push(0),
        J::Bool(flag) => {
            out.push(1);
            out.push(u8::from(*flag));
        }
        J::Num(lexeme) => {
            out.push(2);
            tree_count(out, lexeme.len());
            out.extend_from_slice(lexeme.as_bytes());
        }
        J::Str(text) => {
            out.push(3);
            tree_string(text, random, out);
        }
        J::Arr(items) => {
            out.push(4);
            tree_count(out, items.len());
            for item in items {
                encode_tree(item, random, out);
            }
        }
        J::Obj(fields) => {
            out.push(5);
            let duplicate = !fields.is_empty() && random.chance(1, 40);
            tree_count(out, fields.len() + usize::from(duplicate));
            for (name, item) in fields {
                tree_string(name, random, out);
                encode_tree(item, random, out);
            }
            if duplicate {
                tree_string(&fields[0].0, random, out);
                encode_tree(&J::Str(b"duplicate".to_vec()), random, out);
            }
        }
    }
}

fn tree_cases(cases: &mut Vec<Labeled>, random: &mut Random, profiles: &[[Vec<u8>; 4]]) {
    for depth in [62, 63, 64, 65, 66] {
        let mut tree = Vec::new();
        encode_tree(&deep(depth, J::Null), &mut Random::new(3), &mut tree);
        push(
            cases,
            format!("template_json tree depth {depth}"),
            Case::TemplateTree { tree },
        );
        let request = obj(vec![
            (
                "messages",
                J::Arr(vec![obj(vec![("role", s("user")), ("content", s("x"))])]),
            ),
            (
                "tools",
                J::Arr(vec![obj(vec![("a", deep(depth - 1, J::Null))])]),
            ),
        ]);
        let mut tree = Vec::new();
        encode_tree(&request, &mut Random::new(3), &mut tree);
        push(
            cases,
            format!("render tree tool depth {depth}"),
            Case::RenderTree {
                strings: strings(TEST_STRINGS),
                tree,
            },
        );
    }
    for (index, bad) in BAD_BYTES.iter().enumerate() {
        let mut tree = vec![3];
        tree_count(&mut tree, bad.len());
        tree.extend_from_slice(bad);
        push(
            cases,
            format!("template_json tree bad string {index}"),
            Case::TemplateTree { tree },
        );
        let mut tree = vec![5];
        tree_count(&mut tree, 1);
        tree_count(&mut tree, bad.len());
        tree.extend_from_slice(bad);
        tree.push(0);
        push(
            cases,
            format!("template_json tree bad key {index}"),
            Case::TemplateTree { tree },
        );
    }
    for index in 0..20_000 {
        let mut tree = Vec::new();
        encode_tree(&json_value(random, 0), random, &mut tree);
        push(
            cases,
            format!("template_json tree random {index}"),
            Case::TemplateTree { tree },
        );
    }
    for index in 0..20_000 {
        let mut tree = Vec::new();
        encode_tree(&request(random), random, &mut tree);
        let profile = random.pick(profiles).clone();
        push(
            cases,
            format!("render tree random {index}"),
            Case::RenderTree {
                strings: profile,
                tree,
            },
        );
    }
}

/// All cases. `template` is the pinned template source, when available.
pub fn cases(seed: u64, template: Option<&[u8]>) -> Vec<Labeled> {
    let mut random = Random::new(seed);
    let mut cases = Vec::new();
    test_chat_cases(&mut cases);
    test_response_cases(&mut cases);
    test_protocol_cases(&mut cases);
    chat_edges(&mut cases);
    parser_edges(&mut cases);
    for index in 0..60_000 {
        let value = match random.below(10) {
            0..=6 => J::Num(number(&mut random)),
            7 => J::Str(text(&mut random, 10).into_bytes()),
            _ => json_value(&mut random, 0),
        };
        push(
            &mut cases,
            format!("template_json random {index}"),
            Case::TemplateJson {
                json: value.text(&mut random),
            },
        );
    }
    for index in 0..60_000 {
        let lexeme = tie_number(&mut random);
        push(
            &mut cases,
            format!("template_json tie {index}"),
            Case::TemplateJson {
                json: lexeme.into_bytes(),
            },
        );
    }
    let mut profiles = vec![strings(TEST_STRINGS), strings(ROUND_TRIP_STRINGS)];
    if let Some(source) = template {
        push(
            &mut cases,
            "pinned template",
            Case::Extract {
                source: source.to_vec(),
            },
        );
        if let Ok(literals) = tokspur_text::chat::extract_flash_next_literals(source) {
            profiles.push([
                literals.reasoning_xhigh,
                literals.reasoning_low,
                literals.tools_header,
                literals.tool_instructions,
            ]);
        }
        for index in 0..400 {
            let mut mutated = source.to_vec();
            match random.below(3) {
                0 => mutated.truncate(random.below(source.len() + 1)),
                1 => {
                    let at = random.below(mutated.len() + 1);
                    let inserts = [b"'".as_slice(), b"\"", b"\\", b"{#", b"#}", b"\\x"];
                    let insert = random.pick(&inserts);
                    mutated.splice(at..at, insert.iter().copied());
                }
                _ => {
                    if !mutated.is_empty() {
                        let at = random.below(mutated.len());
                        mutated.remove(at);
                    }
                }
            }
            push(
                &mut cases,
                format!("pinned template mutation {index}"),
                Case::Extract { source: mutated },
            );
        }
    }
    for index in 0..40_000 {
        let profile = random.pick(&profiles).clone();
        let mut request = request(&mut random).text(&mut random);
        if random.chance(1, 50) && !request.is_empty() {
            let at = random.below(request.len());
            request[at] = u8::try_from(random.below(256)).unwrap_or(0);
        }
        push(
            &mut cases,
            format!("render random {index}"),
            Case::Render {
                strings: profile,
                request,
            },
        );
    }
    for index in 0..30_000 {
        let profile = random.pick(&profiles).clone();
        let request = valid_request(&mut random).text(&mut random);
        push(
            &mut cases,
            format!("render valid {index}"),
            Case::Render {
                strings: profile,
                request,
            },
        );
    }
    for index in 0..4000 {
        let mut value = text(&mut random, 8).into_bytes();
        if random.chance(1, 6) {
            let at = random.below(value.len() + 1);
            let bad = random.pick(BAD_BYTES);
            value.splice(at..at, bad.iter().copied());
        }
        push(
            &mut cases,
            format!("chat_trim random {index}"),
            Case::Trim { text: value },
        );
    }
    for index in 0..2000 {
        let source: Vec<u8> = (0..random.below(40))
            .flat_map(|_| {
                random
                    .pick(&[
                        b"'".as_slice(),
                        b"\"",
                        b"\\",
                        b"\\n",
                        b"{#",
                        b"#}",
                        b"{%",
                        b"%}",
                        b"x",
                        b" ",
                        b"Reasoning effort is set to xhigh.",
                        b"Reasoning effort is set to low.",
                        b"# Tools\n\n",
                        b"\n\nIf you choose to call a function ONLY reply",
                    ])
                    .to_vec()
            })
            .collect();
        push(
            &mut cases,
            format!("extract random {index}"),
            Case::Extract { source },
        );
    }
    tree_cases(&mut cases, &mut random, &profiles);
    random_parser_cases(&mut cases, &mut random, 40_000);
    for index in 0..30_000 {
        let (admission, envelope, decoder) = random_envelope(&mut random);
        let text = envelope.text(&mut random);
        push(
            &mut cases,
            format!("protocol random {index}"),
            Case::Protocol {
                admission,
                envelope: text,
                decoder,
            },
        );
    }
    cases
}
