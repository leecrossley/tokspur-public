use super::MeasuredObservation;

#[test]
fn packet_timings_cover_each_action_and_preserve_the_final_binary64_input() {
    use tokspur_json::Json;
    let ready = Json::new_object([
        ("native_measured_controller", Json::Bool(true)),
        ("native_measured_timing_schema", Json::integer(1)),
    ]);
    let observation = MeasuredObservation {
        generation: 44,
        actions: 2,
        last_decode_ms: 71.12345678901234,
        acceptance: [0.5; 7],
        cost_multiplier: 1.2,
        rate: 0.01,
        history_size: 2,
        since_deep_probe: 2,
        converged: true,
        calibrated: true,
    };
    let completion = |first, times: Vec<f64>, observation: MeasuredObservation| {
        Json::new_object([
            ("native_measured_timing_schema", Json::integer(1)),
            (
                "native_measured_decode_observations",
                Json::new_object([
                    ("first_action", Json::integer(first)),
                    (
                        "milliseconds",
                        Json::Array(
                            times
                                .into_iter()
                                .map(|v| {
                                    tokspur_json::parse_json(
                                        tokspur_json::to_chars_shortest(v).unwrap().as_bytes(),
                                    )
                                    .unwrap()
                                })
                                .collect(),
                        ),
                    ),
                ]),
            ),
            (
                "native_mtp_generation",
                Json::new_object([(
                    "stage_a_controller",
                    Json::new_object([("measured", observation.json().unwrap())]),
                )]),
            ),
        ])
    };
    let done = completion(1, vec![1.0 / 3.0, observation.last_decode_ms], observation);
    let empty = Json::new_object(std::iter::empty::<(&str, Json)>());
    assert!(super::validate_measured_timing_packet(&ready, &empty, &done, true, 44, 2).is_ok());
    for (first, times) in [
        (1, vec![observation.last_decode_ms]),
        (2, vec![1.0 / 3.0, observation.last_decode_ms]),
        (1, vec![-1.0, observation.last_decode_ms]),
        (1, vec![1.0 / 3.0, 70.0]),
    ] {
        assert!(
            super::validate_measured_timing_packet(
                &ready,
                &empty,
                &completion(first, times, observation),
                true,
                44,
                2
            )
            .is_err()
        );
    }
    assert!(super::validate_measured_timing_packet(&empty, &empty, &done, true, 44, 2).is_err());
    assert!(super::validate_measured_timing_packet(&ready, &empty, &done, true, 45, 2).is_err());
    let next = MeasuredObservation {
        actions: 3,
        last_decode_ms: 12.0,
        ..observation
    };
    assert!(
        super::validate_measured_timing_packet(
            &ready,
            &done,
            &completion(3, vec![12.0], next),
            false,
            44,
            1
        )
        .is_ok()
    );
    assert!(
        super::validate_measured_timing_packet(
            &ready,
            &done,
            &completion(0, vec![], next),
            false,
            44,
            0
        )
        .is_ok()
    );
}

#[test]
fn experimental_kernel_selection_cannot_change_during_exchange() {
    use tokspur_json::Json;
    let default = Json::new_object(std::iter::empty::<(&str, Json)>());
    let selected = Json::new_object([("native_verify_narrow_rows", Json::Bool(true))]);
    let changed = Json::new_object([("native_verify_narrow_rows", Json::Bool(false))]);
    assert!(super::validate_runtime_policy_identity(&default, &default).is_ok());
    assert!(super::validate_runtime_policy_identity(&selected, &selected).is_ok());
    for observed in [&default, &changed] {
        assert!(super::validate_runtime_policy_identity(&selected, observed).is_err());
    }
    assert!(super::validate_runtime_policy_identity(&default, &selected).is_err());
}

#[test]
fn pack_mapping_cannot_change_during_exchange() {
    use tokspur_json::Json;
    let default = Json::new_object(std::iter::empty::<(&str, Json)>());
    let selected = Json::new_object([("native_pack_shared_mapping", Json::Bool(true))]);
    let changed = Json::new_object([("native_pack_shared_mapping", Json::Bool(false))]);
    assert!(super::validate_runtime_policy_identity(&default, &default).is_ok());
    assert!(super::validate_runtime_policy_identity(&selected, &selected).is_ok());
    for observed in [&default, &changed] {
        assert!(super::validate_runtime_policy_identity(&selected, observed).is_err());
    }
    assert!(super::validate_runtime_policy_identity(&default, &selected).is_err());
}

#[test]
fn observations_round_trip_binary64_and_refuse_unbounded_history() {
    let observation = MeasuredObservation {
        generation: 44,
        actions: 16,
        last_decode_ms: 71.12345678901234,
        acceptance: [0.0, 1.0, 1.0 / 3.0, 0.45, 0.6, 0.2, 0.1],
        cost_multiplier: 1.2345678901234567,
        rate: 0.012345678901234567,
        history_size: 32,
        since_deep_probe: 16,
        converged: true,
        calibrated: false,
    };
    assert_eq!(
        MeasuredObservation::parse(&observation.json().unwrap()).unwrap(),
        observation
    );
    let invalid = MeasuredObservation {
        history_size: 33,
        ..observation
    };
    assert!(MeasuredObservation::parse(&invalid.json().unwrap()).is_err());
    let invalid = MeasuredObservation {
        actions: 0,
        ..observation
    };
    assert!(MeasuredObservation::parse(&invalid.json().unwrap()).is_err());
}
