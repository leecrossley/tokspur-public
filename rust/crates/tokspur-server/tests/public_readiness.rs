#![forbid(unsafe_code)]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use tokspur_json::{Json, Kind, parse_json};
use tokspur_server::admission::{admit_chat, validate_tool_value};
use tokspur_text::response_protocol::{
    RESPONSE_MAXIMUM_ACCUMULATED_IDS, ResponseAdmission, parse_native_response,
};

fn json(text: &str) -> Json {
    parse_json(text.as_bytes()).unwrap()
}

#[test]
fn reset_peers_do_not_remove_the_listener() {
    use std::io::Write;
    use std::net::TcpStream;
    use std::os::fd::AsFd;
    use std::time::{Duration, Instant};
    use tokspur_server::http::{Limits, Listener};

    let listener = Listener::new(0, 8).unwrap();
    for _ in 0..20 {
        let peer = TcpStream::connect(("127.0.0.1", listener.port())).unwrap();
        tokspur_sys::net::abortive_close_on_drop(peer.as_fd()).unwrap();
        drop(peer);
        std::thread::sleep(Duration::from_millis(2));
        match listener.accept(
            Instant::now() + Duration::from_millis(10),
            Limits::default(),
        ) {
            Ok(connection) => drop(connection),
            Err(error) => assert_eq!(error.status(), 408, "{error}"),
        }
    }
    let mut peer = TcpStream::connect(("127.0.0.1", listener.port())).unwrap();
    peer.write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .unwrap();
    let mut connection = listener
        .accept(Instant::now() + Duration::from_secs(1), Limits::default())
        .unwrap();
    let request = connection
        .read_request(Instant::now() + Duration::from_secs(1))
        .unwrap();
    assert_eq!(request.path, "/health");
}

#[test]
fn non_loopback_peers_do_not_remove_the_listener() {
    use std::io::Write;
    use std::net::{IpAddr, TcpStream, UdpSocket};
    use std::time::{Duration, Instant};
    use tokspur_server::http::{Limits, Listener};

    // A UDP connect picks the host's outbound address without sending anything.
    let Some(local) = UdpSocket::bind(("0.0.0.0", 0))
        .and_then(|socket| socket.connect(("192.0.2.1", 9)).map(|()| socket))
        .and_then(|socket| socket.local_addr())
        .ok()
        .map(|address| address.ip())
        .filter(|ip| matches!(ip, IpAddr::V4(v4) if !v4.is_loopback() && !v4.is_unspecified()))
    else {
        return;
    };
    let listener = Listener::new(0, 8).unwrap();
    // std cannot choose a source address, so BSD nc connects from the host address to loopback.
    let Ok(mut outside) = std::process::Command::new("/usr/bin/nc")
        .args([
            "-s",
            &local.to_string(),
            "127.0.0.1",
            &listener.port().to_string(),
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    else {
        return;
    };
    match listener.accept(
        Instant::now() + Duration::from_millis(500),
        Limits::default(),
    ) {
        Ok(_) => panic!("a non-loopback peer was accepted"),
        Err(error) => assert_eq!(error.status(), 408, "{error}"),
    }
    let _cleanup_kill = outside.kill();
    let _cleanup_wait = outside.wait();
    let mut peer = TcpStream::connect(("127.0.0.1", listener.port())).unwrap();
    peer.write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
        .unwrap();
    let mut connection = listener
        .accept(Instant::now() + Duration::from_secs(1), Limits::default())
        .unwrap();
    let request = connection
        .read_request(Instant::now() + Duration::from_secs(1))
        .unwrap();
    assert_eq!(request.path, "/health");
}

#[test]
fn measured_sampling_is_checked_before_submission() {
    use tokspur_server::admission::validate_calibrated_sampling;
    let ready = json(
        r#"{"native_measured_controller":true,"native_controller_sampling_modes":["greedy","topk"]}"#,
    );
    for (temperature, top_k, top_p, accepted) in [
        (0.0, 20, 0.95, true),
        (1.0, 20, 1.0, true),
        (1.0, 20, 0.95, false),
        (0.7, 20, 1.0, false),
        (1.0, 4, 1.0, false),
    ] {
        let sampling = json(&format!(
            r#"{{"temperature":{temperature},"top_k":{top_k},"top_p":{top_p},"min_p":0}}"#
        ));
        assert_eq!(
            validate_calibrated_sampling(&sampling, &ready).is_ok(),
            accepted
        );
    }
}

#[test]
fn client_fields_are_admitted_without_changing_worker_lifetime() {
    for keep in ["0", "-1", "\"5m\"", "\"1.5h\"", "\"250ms\""] {
        let request = json(&format!(
            r#"{{"model":"m","messages":[{{"role":"user","content":"hello"}}],"n":1,"parallel_tool_calls":false,"keep_alive":{keep}}}"#
        ));
        let admitted = admit_chat(&request, b"m").unwrap();
        assert!(admitted.single_tool_call);
        assert_eq!(admitted.maximum_tokens, 8192);
    }
    for extra in [
        r#""n":2"#,
        r#""n":1.5"#,
        r#""parallel_tool_calls":0"#,
        r#""keep_alive":"forever""#,
    ] {
        let request = json(&format!(
            r#"{{"model":"m","messages":[{{"role":"user","content":"hello"}}],{extra}}}"#
        ));
        assert!(admit_chat(&request, b"m").is_err());
    }
}

#[test]
fn translated_thinking_history_requires_an_exact_plaintext_mirror() {
    for blocks in [
        r#"[{"type":"thinking","thinking":"fixture &amp; <tool_call>","signature":""}]"#,
        r#"[{"type":"thinking","thinking":"fixture &amp; "},{"type":"thinking","thinking":"<tool_call>"}]"#,
    ] {
        let request = json(&format!(
            r#"{{"model":"m","messages":[{{"role":"assistant","content":"done","reasoning_content":"fixture &amp; <tool_call>","thinking_blocks":{blocks}}},{{"role":"user","content":"next"}}]}}"#
        ));
        let admitted = admit_chat(&request, b"m").unwrap();
        assert_eq!(
            admitted.render.find("messages").unwrap(),
            request.find("messages").unwrap()
        );
    }
    for message in [
        r#"{"role":"assistant","content":"done","thinking_blocks":[{"type":"thinking","thinking":"hidden"}]}"#,
        r#"{"role":"assistant","content":"done","reasoning_content":"visible","thinking_blocks":[{"type":"thinking","thinking":"different"}]}"#,
        r#"{"role":"user","content":"done","thinking_blocks":[]}"#,
        r#"{"role":"assistant","content":"done","thinking_blocks":null}"#,
        r#"{"role":"assistant","content":"done","thinking_blocks":[{"type":"redacted_thinking","data":"opaque"}]}"#,
        r#"{"role":"assistant","content":"done","thinking_blocks":[{"type":"thinking","thinking":"","signature":"unverified"}]}"#,
        r#"{"role":"assistant","content":"done","thinking_blocks":[{"type":"thinking","thinking":"","extra":true}]}"#,
    ] {
        let request = json(&format!(r#"{{"model":"m","messages":[{message}]}}"#));
        assert!(admit_chat(&request, b"m").is_err());
    }
    let request = json(
        r#"{"model":"m","messages":[{"role":"assistant","content":"done","thinking_blocks":[]}] }"#,
    );
    assert!(admit_chat(&request, b"m").is_ok());
}

#[test]
fn translated_tool_indexes_are_bounded_metadata() {
    for (index, accepted) in [
        ("0", true),
        ("1", true),
        ("127", true),
        ("128", false),
        ("-1", false),
        ("0.0", false),
        ("\"0\"", false),
    ] {
        let request = json(&format!(
            r#"{{"model":"m","messages":[{{"role":"assistant","tool_calls":[{{"id":"call-a","type":"function","index":{index},"function":{{"name":"f","arguments":"{{}}"}}}}]}},{{"role":"tool","tool_call_id":"call-a","content":"done"}}],"tools":[{{"type":"function","function":{{"name":"f"}}}}]}}"#
        ));
        assert_eq!(admit_chat(&request, b"m").is_ok(), accepted);
    }
    for (second_index, accepted) in [(0, true), (1, true), (2, true), (128, false)] {
        let request = json(&format!(
            r#"{{"model":"m","messages":[{{"role":"assistant","tool_calls":[{{"id":"a","type":"function","index":0,"function":{{"name":"f","arguments":"{{}}"}}}},{{"id":"b","type":"function","index":{second_index},"function":{{"name":"f","arguments":"{{}}"}}}}]}},{{"role":"tool","tool_call_id":"a","content":"one"}},{{"role":"tool","tool_call_id":"b","content":"two"}}],"tools":[{{"type":"function","function":{{"name":"f"}}}}]}}"#
        ));
        assert_eq!(admit_chat(&request, b"m").is_ok(), accepted);
    }
}

#[test]
fn valid_history_exercises_output_bounds_and_literal_controls() {
    for (limit, accepted) in [(0, false), (1, true), (32768, true), (32769, false)] {
        let request = json(&format!(
            r#"{{"model":"m","messages":[{{"role":"user","content":"hello"}}],"max_tokens":{limit}}}"#
        ));
        assert_eq!(admit_chat(&request, b"m").is_ok(), accepted);
    }
    for marker in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"] {
        let request = json(&format!(
            r#"{{"model":"m","messages":[{{"role":"user","content":"quoted {marker}"}}]}}"#
        ));
        assert!(admit_chat(&request, b"m").is_ok());
    }
}

#[test]
fn schema_annotations_and_unions_keep_value_validation() {
    let request = json(
        r#"{"model":"m","messages":[{"role":"user","content":"hello"}],"tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"key":{"anyOf":[{"type":"string","format":"uri","default":"x"},{"type":"integer","minimum":0},{"type":"null"}],"title":"Key","examples":["x",null]}},"required":["key"],"additionalProperties":false}}}]}"#,
    );
    let admitted = admit_chat(&request, b"m").unwrap();
    for value in [r#"{"key":"value"}"#, r#"{"key":5}"#, r#"{"key":null}"#] {
        validate_tool_value(&admitted.schemas[0], &json(value)).unwrap();
    }
    for value in [
        r#"{"key":true}"#,
        r#"{"key":-1}"#,
        r#"{"key":5.5}"#,
        r#"{}"#,
    ] {
        assert!(validate_tool_value(&admitted.schemas[0], &json(value)).is_err());
    }
}

fn decode(ids: &[u32]) -> Result<Vec<u8>, std::convert::Infallible> {
    let mut bytes = Vec::new();
    for &id in ids {
        if id == 256 {
            bytes.extend_from_slice(b"<|im_end|>");
        } else {
            bytes.push(u8::try_from(id).unwrap());
        }
    }
    Ok(bytes)
}

const ADMISSION: ResponseAdmission = ResponseAdmission {
    vocabulary: 300,
    im_end: 256,
    end_of_text: 257,
    im_start: 258,
};

#[test]
fn accumulated_response_limit_includes_the_largest_http_budget() {
    for count in [16384, 16385, 32768, 32769] {
        let ids = Json::Array(vec![Json::integer(65); count]);
        let request = Json::new_object([
            ("ids", ids),
            ("tools", Json::Array(Vec::new())),
            ("initial_thinking", Json::Bool(false)),
            ("finish_reason", Json::from("length")),
            ("stop_id", Json::Null),
        ]);
        let result = parse_native_response(
            &request,
            ADMISSION,
            Some(&decode),
            RESPONSE_MAXIMUM_ACCUMULATED_IDS,
        );
        if count <= 32768 {
            assert_eq!(
                result
                    .unwrap()
                    .find("content")
                    .unwrap()
                    .unwrap()
                    .string()
                    .unwrap()
                    .len(),
                count
            );
        } else {
            assert!(result.is_err());
        }
    }
}

#[test]
fn union_parameters_publish_their_json_types() {
    for (raw, kind) in [
        ("null", Kind::Null),
        ("5", Kind::Number),
        ("hello", Kind::String),
    ] {
        let text = format!(
            "<tool_call>\n<function=lookup>\n<parameter=key>\n{raw}\n</parameter>\n</function>\n</tool_call>"
        );
        let mut ids: Vec<Json> = text.bytes().map(|b| Json::integer(u64::from(b))).collect();
        ids.push(Json::integer(256));
        let request = Json::new_object([
            ("ids", Json::Array(ids)),
            (
                "tools",
                json(
                    r#"[{"name":"lookup","parameters":[{"name":"key","type":"any","required":true}]}]"#,
                ),
            ),
            ("initial_thinking", Json::Bool(false)),
            ("finish_reason", Json::from("stop")),
            ("stop_id", Json::integer(256)),
        ]);
        let output = parse_native_response(
            &request,
            ADMISSION,
            Some(&decode),
            RESPONSE_MAXIMUM_ACCUMULATED_IDS,
        )
        .unwrap();
        let call = &output.find("tool_calls").unwrap().unwrap().array().unwrap()[0];
        let arguments = call
            .find("function")
            .unwrap()
            .unwrap()
            .find("arguments")
            .unwrap()
            .unwrap();
        assert_eq!(arguments.find("key").unwrap().unwrap().kind(), kind);
    }
}
