//! Port of `native/tests/test_json.cpp` (`basics`, `generated`).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stderr)]

use tokspur_json::{Json, JsonError, JsonLimits, parse_json, parse_json_limited, scalar_utf8};
use tokspur_testkit::Mt19937;

fn parse(text: &str) -> Json {
    parse_json(text.as_bytes()).unwrap()
}

#[test]
fn basics_preserve_bom_nul_and_unsigned_maxima() {
    let object = parse(
        r#"{"text":"\ufeffusing café\u0000","ids":[0,1,4294967295],"flag":true,"nothing":null}"#,
    );
    let mut expected = "\u{feff}using café".as_bytes().to_vec();
    expected.push(0);
    assert_eq!(
        object.find("text").unwrap().unwrap().string().unwrap(),
        expected
    );
    assert!(object.find("flag").unwrap().unwrap().boolean().unwrap());
    assert!(object.find("nothing").unwrap().unwrap().is_null());
    assert!(object.find("missing").unwrap().is_none());
    let ids = object.find("ids").unwrap().unwrap().array().unwrap();
    assert_eq!(ids[2].unsigned_integer().unwrap(), 4_294_967_295);
    assert_eq!(
        parse("18446744073709551615").unsigned_integer().unwrap(),
        u64::MAX
    );
    let dumped = object.dump().unwrap();
    assert_eq!(parse_json(&dumped).unwrap().dump().unwrap(), dumped);
}

#[test]
fn basics_decode_surrogate_pairs_and_escapes() {
    assert_eq!(
        parse(r#""\ud83e\udd89""#).string().unwrap(),
        "🦉".as_bytes()
    );
    assert_eq!(
        parse(r#""\"\\\/\b\f\n\r\t""#).string().unwrap(),
        b"\"\\/\x08\x0c\n\r\t"
    );
    assert_eq!(
        parse(" \n [true, false, null, -1.25e+20] \t")
            .dump()
            .unwrap(),
        b"[true,false,null,-1.25e+20]"
    );
}

#[test]
fn basics_refuse_malformed_documents() {
    let bad = [
        "",
        " ",
        "{",
        "[",
        "[1,]",
        "{\"x\":1,}",
        "{x:1}",
        "01",
        "-",
        "1.",
        "1e",
        "+1",
        "NaN",
        "Infinity",
        "true false",
        "\"\\x00\"",
        "\"\\uD800\"",
        "\"\\uDC00\"",
        "\"\\uD800\\u0000\"",
        "{\"x\":1,\"x\":2}",
        "{\"x\":1,\"\\u0078\":2}",
    ];
    for text in bad {
        assert!(parse_json(text.as_bytes()).is_err(), "accepted {text:?}");
    }
    for bytes in [b"\"\xc0\x80\"".as_slice(), b"\"\xed\xa0\x80\"", b"\"\n\""] {
        assert!(parse_json(bytes).is_err());
    }
}

#[test]
fn basics_refuse_nonintegral_or_signed_unsigned_reads() {
    for text in [
        "-0",
        "1.0",
        "1e2",
        "18446744073709551616",
        "-1",
        "null",
        "true",
        "\"1\"",
    ] {
        assert!(parse(text).unsigned_integer().is_err(), "accepted {text}");
    }
}

#[test]
fn basics_refuse_wrong_kind_access() {
    let object = parse(r#"{"text":"x","ids":[0]}"#);
    assert_eq!(
        object.find("ids").unwrap().unwrap().string(),
        Err(JsonError::Access("expected JSON string"))
    );
    assert_eq!(
        object.find("text").unwrap().unwrap().array(),
        Err(JsonError::Access("expected JSON array"))
    );
    assert_eq!(
        parse("[]").find("x"),
        Err(JsonError::Access("expected JSON object"))
    );
}

#[test]
fn basics_enforce_parse_and_dump_limits() {
    let structural = Err(JsonError::Parse("JSON structural limit exceeded"));
    assert_eq!(
        parse_json_limited(b"[1,2]", JsonLimits::new(1024, 2, 4)),
        structural
    );
    assert_eq!(
        parse_json_limited(b"[[1]]", JsonLimits::new(1024, 20, 1)),
        structural
    );
    assert_eq!(
        parse_json_limited(b"null", JsonLimits::new(3, 20, 1)),
        Err(JsonError::Parse("JSON input byte limit exceeded"))
    );
    let object = parse(r#"{"text":"x"}"#);
    assert_eq!(
        object.dump_limited(3),
        Err(JsonError::Dump("JSON output byte limit exceeded"))
    );
    let duplicate = Json::Object(vec![
        (b"x".to_vec(), Json::Null),
        (b"x".to_vec(), Json::Null),
    ]);
    assert_eq!(
        duplicate.dump(),
        Err(JsonError::Dump("duplicate JSON output key"))
    );
    assert!(Json::String(vec![0x80]).dump().is_err());
}

#[test]
fn generated_unicode_round_trips_match_the_cpp_sequence() {
    let mut random = Mt19937::new(0x6a73_6f6e);
    for _ in 0..5000 {
        let mut text = Vec::new();
        let count = random.next_u32() % 100;
        for _ in 0..count {
            let mut point = random.next_u32() % 0x11_0000;
            if (0xd800..=0xdfff).contains(&point) {
                point = 0;
            }
            text.extend(scalar_utf8(point).unwrap());
        }
        let value = Json::new_object([
            ("text", Json::String(text.clone())),
            ("id", Json::integer(u64::from(random.next_u32()))),
        ]);
        let decoded = parse_json(&value.dump().unwrap()).unwrap();
        assert_eq!(
            decoded.find("text").unwrap().unwrap().string().unwrap(),
            text
        );
        assert_eq!(decoded.dump().unwrap(), value.dump().unwrap());
    }
}
