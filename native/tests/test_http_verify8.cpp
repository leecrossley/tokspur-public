#include "tokspur/serve_verify8.hpp"
#include "tokspur/owned_worker.hpp"
#include "tokspur/serve.hpp"
#include "tokspur/gguf.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <poll.h>
#include <unistd.h>

#ifndef TOKSPUR_PRIVATE_HTTP_VERIFY8
#error This generated CPU fixture belongs only to the private HTTP V8 build.
#endif

// Independently generated protocol records only. No model, tokenizer, Metal,
// external engine or actual inference asset is used by this fixture.
using namespace tokspur;
namespace {
using Clock = std::chrono::steady_clock;
using namespace std::chrono_literals;
size_t checks = 0;
void check(bool ok, const char* why = "private HTTP V8 fixture failed") {
    ++checks; if (!ok) throw std::runtime_error(why);
}
template<class F> void refuses(F&& action) {
    bool refused = false;
    try { action(); } catch (const FormatError&) { refused = true; } catch (const WorkerError&) { refused = true; }
    check(refused, "unsafe private V8 operation accepted");
}
Json with(Json value, std::string_view key, Json replacement) {
    auto fields = value.object();
    for (auto& [name, item] : fields) if (name == key) { item = std::move(replacement); return Json(std::move(fields)); }
    fields.emplace_back(key, std::move(replacement)); return Json(std::move(fields));
}
Json without(const Json& value, std::string_view key) {
    Json::Object fields;
    for (const auto& item : value.object()) if (item.first != key) fields.push_back(item);
    return Json(std::move(fields));
}
uint64_t integer(const Json& value, std::string_view key) { return value.find(key)->unsigned_integer(); }
Json number(double value) {
    std::ostringstream text; text << std::setprecision(17) << value; return parse_json(text.str());
}
Json policy() { return parse_json(R"({"temperature":0,"top_k":0,"top_p":1,"min_p":0})"); }
Json hc(const std::array<uint64_t, 5>& counts = {}) {
    return Json(Json::Object{{"scalar", Json::integer(counts[0])}, {"many1", Json::integer(counts[1])},
        {"many2", Json::integer(counts[2])}, {"many4", Json::integer(counts[3])}, {"many8", Json::integer(counts[4])}});
}
Json ready_record() {
    return parse_json(R"({
      "event":"ready","small_gemv_column_splits":16,"block_gemv_formats":3,"stack_execution":"direct",
      "maximum_generated_tokens":4096,"known_token_capacity":8,"known_residual_reuse":true,
      "prompt_lookup":true,"verification_max_tokens":8,"verification_head_many":true,
      "hc_q8_splits":0,"hc_q8_numerical_tier":"ordinary","hc_q8_scratch_logical_bytes":0,"hc_q8_scratch_charged_bytes":0,
      "replay_prepared":false,"recurrent_replay_captures":0,"mapped_weight_region_limit_bytes":0,
      "capture_resource_bytes":0,"capture_binding_aliases":0,"capture_binding_alias_span_bytes":0,
      "capture_binding_alias_resource_bytes":0,"live_captures":0,"profile_counter_payload_bytes":0,
      "checkpoint_slots":0,"checkpoint_combined_budget_bytes":0,"checkpoint_host_bytes":0,
      "checkpoint_store_host_bytes":0,"checkpoint_gpu_bytes":0,"checkpoint_broker":false,"broker_epoch":0,
      "native_limit_bytes":103079215104,"weight_copy":false,"gpu_profile_mode":"disabled","timings_qualified":false,
      "allow_prompt_lookup_experiments":true,"allow_request_kernel_experiments":false,
      "allow_request_stack_experiments":false,"allow_request_prefill_experiments":false,
      "lookup_match_tokens":16,"lookup_history_tokens":65536,"lookup_proposal_tokens":7,
      "lookup_verification_budget_bytes":134217728,"lookup_caller_host_bytes":2709960,
      "verification_host_bytes":9293008,"verification_gpu_bytes":50076576,
      "lookup_verification_charged_bytes":62079544,"verification_tape_payload_bytes":33140736,
      "tracked_metal_buffer_bytes":83805224124,"peak_metal_buffer_bytes":83805224124,
      "gpu_buffer_limit_bytes":100394860544,"imported_weight_backing_bytes":81848123392,"gpu_allocation_count":612,
      "hc_exact_mode":"combined","hc_exact_rows_per_group":1,"hc_exact_unroll":16,
      "hc_exact_scratch_logical_bytes":0,"hc_exact_scratch_charged_bytes":0,
      "hc_exact_routes":{"scalar":0,"many1":0,"many2":0,"many4":0,"many8":0},
      "wide_q8_exact_enabled":false,"wide_q8_exact_rows_per_group":4,"wide_q8_exact_unroll":16,
      "wide_q8_exact_scratch_logical_bytes":0,"wide_q8_exact_scratch_charged_bytes":0,
      "wide_q8_exact_routes":{"qkv_scalar":0,"qkv_many1":0,"output_scalar":0,"output_many1":0},
      "wide_q8_many_enabled":false,"wide_q8_many_rows_per_group":4,"wide_q8_many_unroll":4,
      "wide_q8_many_scratch_logical_bytes":0,"wide_q8_many_scratch_charged_bytes":0,
      "wide_q8_many_routes":{"qkv_many2":0,"qkv_many4":0,"qkv_many8":0,"output_many2":0,"output_many4":0,"output_many8":0}
    })");
}
Json input_job(uint32_t count, bool reset, const Json& sampling, uint64_t seed) {
    Json::Array ids; for (uint32_t i = 0; i < count; ++i) ids.push_back(Json::integer(100 + i));
    Json result(Json::Object{{"tokens", Json(std::move(ids))}, {"generate", Json::integer(0)},
        {"reset", Json(reset)}});
    if (reset) { result = with(std::move(result), "sampling", sampling); result = with(std::move(result), "seed", Json::integer(seed)); result = with(std::move(result), "prompt_lookup", Json(true)); }
    return result;
}
Json generation_job(uint32_t count) {
    return Json(Json::Object{{"tokens", Json(Json::Array{})}, {"generate", Json::integer(count)},
        {"reset", Json(false)}, {"stop_ids", Json(Json::Array{Json::integer(248046)})}});
}
struct Round { uint32_t n, k, ordinary; bool terminal = false; uint32_t proposed = 0; };
struct Transcript {
    Json previous, job, sampling, done;
    uint32_t before = 0;
    uint64_t seed = 0;
    std::vector<Json> events;
};
uint32_t grams(uint32_t prefix) { return prefix < 16 ? 0 : prefix - 15; }
uint64_t serial_dispatches(uint32_t prefix) {
    // Independent, fixed-prefix threshold table rather than the runtime helper.
    // Thresholds are the first token of each extra merge level, not powers of2.
    const uint32_t tokens = prefix + 1;
    uint64_t extra = tokens >= 2052 ? 48 : 0;
    for (const uint32_t threshold : {4100u, 8196u, 16388u, 32772u}) if (tokens >= threshold) extra += 12;
    return 2222 + (tokens % 4 == 0 ? 36 : 0) + extra;
}
Transcript make_transcript(const Json& ready, const Json& previous, Json job,
                           const Json& sampling, uint64_t seed, std::span<const Round> rounds = {}) {
    Transcript result{previous, std::move(job), sampling, Json(), previous.is_null() ? 0u : uint32_t(integer(previous, "committed_prefix")), seed, {}};
    const bool reset = result.job.find("reset")->boolean();
    const auto inputs = uint32_t(result.job.find("tokens")->array().size());
    const auto base = reset ? 0u : result.before;
    uint32_t position = base + inputs, total_ordinary = 0, total_proposed = 0, verified = 0, recovery = 0;
    std::array<uint64_t, 5> routes{};
    Json::Array output;
    double wall_sum = 0, batch_wall_sum = 0, batch_gpu_sum = 0;
    bool stopped = false;
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
    const bool stage_prepared = ready.find("qsa_stage_v8_prepared") != nullptr;
    const bool stage_enabled = stage_prepared && ready.find("qsa_stage_v8_enabled")->boolean();
    uint64_t stage_request_calls = stage_enabled ? 12ull * inputs : 0;
#endif
    if (inputs) {
        // All N inputs traverse 24 ordinary HC projections. Only the final
        // retained row executes the scalar head; bit chunks cover 72 others.
        routes[0] = 24 * inputs + 1;
        for (uint32_t width = 1, i = 1; width <= 8; width *= 2, ++i) if (inputs & width) routes[i] = 72;
    }
    for (size_t r = 0; r < rounds.size(); ++r) {
        const auto spec = rounds[r]; const auto count = spec.k + spec.ordinary;
        const auto proposed = std::max(spec.n, spec.proposed);
        const bool partial = spec.k && spec.k < spec.n;
        uint32_t chunks = 0; for (uint32_t n = spec.n; n; n /= 2) chunks += n % 2;
        uint64_t serial = 0; for (uint32_t t = 0; t < spec.n; ++t) serial += serial_dispatches(position + t);
        const auto forward = spec.n ? serial - 72 * (spec.n - 1) - 253 * (spec.n - chunks) : 0;
        const auto dispatches = forward + (partial ? 72 : 0) + (spec.ordinary ? serial_dispatches(position + spec.k) : 0)
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
            + (stage_enabled ? 12ull * (spec.n + spec.ordinary) : 0)
#endif
            ;
        std::array<uint64_t, 5> round_routes{}; round_routes[0] = 25 * spec.n + 97 * spec.ordinary;
        for (uint32_t width = 1, i = 1; width <= 8; width *= 2, ++i) if (spec.n & width) round_routes[i] = 72;
        for (size_t i = 0; i < routes.size(); ++i) routes[i] += round_routes[i];
        const double gather = spec.n ? 2 : 0, verify = spec.n ? 10 : 0, fallback = spec.ordinary ? 4 : 0;
        const double wall = 1 + gather + verify + fallback;
        const double batch_wall = (spec.n ? 8 : 0) + (partial ? 2 : 0) + (spec.ordinary ? 3 : 0);
        const double batch_gpu = (spec.n ? 7 : 0) + (partial ? 1 : 0) + (spec.ordinary ? 2 : 0);
        Json round(Json::Object{{"event", Json("generation_round")}, {"round_id", Json::integer(r + 1)},
            {"round_base", Json::integer(position)}, {"committed_prefix", Json::integer(position + count)},
            {"proposed_tokens", Json::integer(proposed)}, {"verified_tokens", Json::integer(spec.n)},
            {"verification_max_tokens", Json::integer(8)}, {"verification_head_dispatches", Json::integer(chunks)},
            {"accepted_verified_tokens", Json::integer(spec.k)}, {"ordinary_tokens", Json::integer(spec.ordinary)},
            {"useful_tokens", Json::integer(count)}, {"round_wall_ms", number(wall)},
            {"lookup_and_publication_ms", number(1)}, {"gather_wall_ms", number(gather)},
            {"verification_including_recovery_wall_ms", number(verify)}, {"recovery_batch_wall_ms", number(partial ? 2 : 0)},
            {"fallback_wall_ms", number(fallback)}, {"batch_wall_ms", number(batch_wall)}, {"batch_gpu_ms", number(batch_gpu)},
            {"hc_exact_round_routes", hc(round_routes)}, {"wide_q8_exact_round_routes", *ready.find("wide_q8_exact_routes")},
            {"wide_q8_many_round_routes", *ready.find("wide_q8_many_routes")}, {"dispatches", Json::integer(dispatches)},
            {"capture_executes", Json::integer(0)}, {"useful_tokens_per_second", number(count * 1000.0 / wall)},
            {"accepted_tokens_per_second", number(spec.k * 1000.0 / wall)}, {"timings_qualified", Json(false)}, {"gpu_profile", Json()}});
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
        if (stage_prepared) {
            const auto calls = stage_enabled ? 12ull * (spec.n + spec.ordinary) : 0;
            stage_request_calls += calls; round = with(std::move(round), "qsa_stage_v8_round_calls", Json::integer(calls));
        }
#endif
        result.events.push_back(round);
        for (uint32_t t = 0; t < count; ++t) {
            const uint32_t id = spec.terminal && t + 1 == count ? 248046 : uint32_t(200 + output.size());
            output.push_back(Json::integer(id));
            result.events.emplace_back(Json::Object{{"event", Json("token")}, {"id", Json::integer(id)},
                {"source_logit", number(1.25 + t)}, {"committed_prefix", Json::integer(position + count)},
                {"token_prefix", Json::integer(position + t + 1)}, {"round_id", Json::integer(r + 1)},
                {"round_base", Json::integer(position)}, {"round_committed_prefix", Json::integer(position + count)},
                {"verified", Json(t < spec.k)}, {"batched_commit", Json(spec.k >= 2)}, {"entropy_words", Json::integer(0)}, {"gpu_profile", Json()}});
        }
        position += count; total_ordinary += spec.ordinary; total_proposed += proposed;
        verified += spec.n != 0; recovery += partial; stopped = spec.terminal;
        wall_sum += wall; batch_wall_sum += batch_wall; batch_gpu_sum += batch_gpu;
    }
    // Ready-only capability/charge fields are deliberately absent from done,
    // matching the actual target schema rather than inventing stronger echoes.
    Json done = ready;
    for (const auto key : {"maximum_generated_tokens", "checkpoint_broker", "broker_epoch", "weight_copy", "gpu_profile_mode",
        "allow_prompt_lookup_experiments", "allow_request_kernel_experiments", "allow_request_stack_experiments", "allow_request_prefill_experiments",
        "lookup_match_tokens", "lookup_history_tokens", "lookup_proposal_tokens", "lookup_verification_budget_bytes", "lookup_caller_host_bytes",
        "verification_host_bytes", "verification_gpu_bytes", "lookup_verification_charged_bytes", "verification_tape_payload_bytes"}) done = without(done, key);
    auto put = [&](std::string_view key, Json value) { done = with(std::move(done), key, std::move(value)); };
    auto count = [&](std::string_view key, uint64_t value) { put(key, Json::integer(value)); };
    put("event", Json("done")); count("committed_prefix", position); put("generated_ids", Json(output));
    put("known_token_prefill", Json(true)); put("status", Json(inputs ? "prefilled" : stopped ? "stopped" : "length"));
    put("finish_reason", inputs ? Json() : Json(stopped ? "stop" : "length")); put("continuation_allowed", Json(!stopped));
    put("stop_id", stopped ? Json::integer(248046) : Json());
    Json sampled = sampling; sampled = with(std::move(sampled), "valid", Json(true)); sampled = with(std::move(sampled), "seed", Json::integer(seed));
    sampled = with(std::move(sampled), "entropy_prefix", Json::integer(position)); sampled = with(std::move(sampled), "entropy_words", Json::integer(0)); put("sampling", std::move(sampled));
    put("hc_exact_final_logit_sha256", Json(std::string(64, 'a')));
    Json::Array top; for (uint32_t i = 0; i < 8; ++i) top.emplace_back(Json::Object{{"id", Json::integer(i + 200)}, {"logit", number(1.25 - i * 0.25)}});
    put("final_logit_top8", Json(std::move(top))); put("hc_exact_request_routes", hc(routes));
    auto cumulative = routes; const auto& old_routes = *(previous.is_null() ? ready.find("hc_exact_routes") : previous.find("hc_exact_routes"));
    constexpr std::array<const char*, 5> names{"scalar", "many1", "many2", "many4", "many8"};
    for (size_t i = 0; i < names.size(); ++i) cumulative[i] += integer(old_routes, names[i]); put("hc_exact_routes", hc(cumulative));
    put("wide_q8_exact_request_routes", *ready.find("wide_q8_exact_routes")); put("wide_q8_many_request_routes", *ready.find("wide_q8_many_routes"));
    count("lookup_rounds", rounds.size()); count("lookup_verified_rounds", verified); count("lookup_recovery_rounds", recovery);
    count("lookup_proposed_tokens", total_proposed); count("lookup_accepted_tokens", output.size() - total_ordinary); count("lookup_ordinary_tokens", total_ordinary);
    count("lookup_history_epoch", (previous.is_null() ? 1 : integer(previous, "lookup_history_epoch")) + reset);
    for (const auto& [total, delta, amount] : {std::tuple{"lookup_appended_tokens", "lookup_request_appended_tokens", uint64_t(inputs + output.size())},
        std::tuple{"lookup_indexed_grams", "lookup_request_indexed_grams", uint64_t(grams(position) - grams(base))},
        std::tuple{"lookup_queries", "lookup_request_queries", uint64_t(rounds.size())},
        std::tuple{"lookup_resets", "lookup_request_resets", uint64_t(reset)}}) {
        count(delta, amount); count(total, (previous.is_null() ? 0 : integer(previous, total)) + amount);
    }
    count("lookup_truncated_grams", 0); count("lookup_cleared_grams", 0); count("lookup_truncations", 0); count("replayed_input_tokens", inputs);
    put("checkpoint_reuse", Json("off")); count("checkpoint_restored_tokens", 0); count("checkpoint_retained", 0);
    put("lookup_round_wall_ms", number(wall_sum)); put("generation_batch_wall_ms", number(batch_wall_sum)); put("generation_batch_gpu_ms", number(batch_gpu_sum));
    put("prefill_wall_ms", number(inputs ? 1 : 0)); put("request_wall_ms", number(wall_sum + 2)); put("gpu_profile", Json());
#ifdef TOKSPUR_PRIVATE_HTTP_QSA_STAGE_V8
    if (stage_prepared) {
        const auto calls = (previous.is_null() ? 0 : integer(previous, "qsa_stage_v8_calls")) + stage_request_calls;
        count("qsa_stage_v8_request_calls", stage_request_calls); count("qsa_stage_v8_calls", calls);
        count("qsa_stage_v8_physical_dispatches", 2 * calls); count("qsa_stage_v8_extra_dispatches", calls);
    }
#endif
    result.done = std::move(done); return result;
}
void admit(const Json& ready, const Transcript& t) {
    const auto job_pin = t.job.dump(), policy_pin = t.sampling.dump(), previous_pin = t.previous.dump();
    Verify8Reply reply(ready, t.previous, t.job, t.before, t.sampling, t.seed);
    refuses([&] { static_cast<void>(reply.events()); });
    for (const auto& event : t.events) { reply.event(event); refuses([&] { static_cast<void>(reply.events()); }); }
    reply.finish(t.done); check(reply.events().size() == t.events.size());
    for (size_t i = 0; i < t.events.size(); ++i) check(reply.events()[i].dump() == t.events[i].dump(), "admission rewrote source records");
    check(t.job.dump() == job_pin && t.sampling.dump() == policy_pin && t.previous.dump() == previous_pin, "sampler or input ancestry was rewritten");
    refuses([&] { reply.event(t.events.empty() ? Json() : t.events.front()); }); refuses([&] { reply.finish(t.done); });
}
void reject(const Json& ready, const Transcript& t) {
    Verify8Reply reply(ready, t.previous, t.job, t.before, t.sampling, t.seed);
    refuses([&] { for (const auto& event : t.events) reply.event(event); reply.finish(t.done); });
    refuses([&] { static_cast<void>(reply.events()); });
}
Transcript seed_prefix(const Json& ready, uint32_t prefix = 16) {
    Json previous;
    while (prefix) {
        const auto count = std::min(prefix, 8u); auto t = make_transcript(ready, previous, input_job(count, previous.is_null(), policy(), 0), policy(), 0);
        admit(ready, t); previous = std::move(t.done); prefix -= count;
    }
    Transcript result; result.done = std::move(previous); return result;
}
void ready_checks() {
    const auto ready = ready_record(); validate_verify8_ready(ready); ++checks;
    for (const auto& [key, ignored] : ready.object()) {
        static_cast<void>(ignored); refuses([&] { validate_verify8_ready(without(ready, key)); });
        for (const auto& invalid : {Json(), Json("wrong"), Json(Json::Array{})}) refuses([&] { validate_verify8_ready(with(ready, key, invalid)); });
    }
    for (const auto& [key, value] : Json::Object{{"verification_max_tokens", Json::integer(2)}, {"known_token_capacity", Json::integer(4)},
        {"known_residual_reuse", Json(false)}, {"stack_execution", Json("recurrent-replay")}, {"hc_exact_mode", Json("scalar")},
        {"wide_q8_many_enabled", Json(true)}, {"gpu_profile_mode", Json("intrusive")}, {"lookup_proposal_tokens", Json::integer(8)},
        {"checkpoint_slots", Json::integer(1)}, {"lookup_verification_budget_bytes", Json::integer((128ull << 20) + 1)},
        {"lookup_verification_charged_bytes", Json::integer(62079545)}, {"verification_gpu_bytes", Json::integer(128ull << 20)},
        {"verification_host_bytes", Json::integer(0)}, {"verification_tape_payload_bytes", Json::integer(50076577)},
        {"tracked_metal_buffer_bytes", Json::integer(83805224125)}, {"peak_metal_buffer_bytes", Json::integer(100394860545)},
        {"imported_weight_backing_bytes", Json::integer(83805224125)}, {"gpu_allocation_count", Json::integer(0)},
        {"trace_expert_routes", Json(false)}, {"expert_route_trace", Json(Json::Object{})}, {"moe_token_tile_enabled", Json(false)}})
        refuses([&] { validate_verify8_ready(with(ready, key, value)); });
    for (const auto& n : {Json(true), parse_json("8.0"), parse_json("8e0"), parse_json("-1"), parse_json("18446744073709551616")})
        refuses([&] { validate_verify8_ready(with(ready, "verification_max_tokens", n)); });
    auto renamed = without(*ready.find("wide_q8_exact_routes"), "qkv_scalar");
    renamed = with(std::move(renamed), "unknown_zero", Json::integer(0));
    refuses([&] { validate_verify8_ready(with(ready, "wide_q8_exact_routes", renamed)); });
}
void positive_checks() {
    const auto ready = ready_record();
    for (uint32_t n = 1; n <= 8; ++n) admit(ready, make_transcript(ready, Json(), input_job(n, true, policy(), 0), policy(), 0));
    const auto previous = seed_prefix(ready).done;
    for (uint32_t n = 2; n <= 7; ++n) for (uint32_t k = 1; k <= n; ++k) {
        const std::array rounds{Round{n, k, uint32_t(k < n)}};
        // Submitted/evaluated work must itself fit the reservation, not merely
        // the smaller committed prefix. Finish unused output with ordinary work.
        std::vector<Round> sequence(rounds.begin(), rounds.end());
        for (uint32_t at = k + (k < n); at < n; ++at) sequence.push_back({0, 0, 1});
        admit(ready, make_transcript(ready, previous, generation_job(n), policy(), 0, sequence));
    }
    const std::array full_eight{Round{7, 7, 0}, Round{0, 0, 1}};
    const auto eight = make_transcript(ready, previous, generation_job(8), policy(), 0, full_eight); admit(ready, eight);
    const std::array eight_serial{Round{0, 0, 1}, Round{0, 0, 1}, Round{0, 0, 1}, Round{0, 0, 1},
        Round{0, 0, 1}, Round{0, 0, 1}, Round{0, 0, 1}, Round{0, 0, 1}};
    const auto bounded = make_transcript(ready, previous, generation_job(8), policy(), 0, eight_serial); check(bounded.events.size() == 16); admit(ready, bounded);
    for (const auto& rounds : {std::vector<Round>{{7, 7, 0, true}},
        std::vector<Round>{{7, 2, 1, true}}, std::vector<Round>{{2, 2, 0}, {0, 0, 1, true}}})
        admit(ready, make_transcript(ready, previous, generation_job(8), policy(), 0, rounds));
    auto stop_top = previous.find("final_logit_top8")->array(); stop_top[0] = with(stop_top[0], "id", Json::integer(248046));
    const auto stop_previous = with(previous, "final_logit_top8", Json(stop_top));
    const std::array terminal{Round{0, 0, 1, true}};
    const auto stop = make_transcript(ready, stop_previous, generation_job(1), policy(), 0, terminal); admit(ready, stop);
    for (const auto& stops : {std::array{248046u,248044u}, std::array{248044u,248046u}}) {
        auto paired = stop;
        paired.job = with(paired.job,"stop_ids",Json(Json::Array{Json::integer(stops[0]),Json::integer(stops[1])}));
        admit(ready,paired);
        auto top = paired.previous.find("final_logit_top8")->array(); top[0] = with(top[0],"id",Json::integer(248044));
        paired.previous = with(paired.previous,"final_logit_top8",Json(top));
        for (auto& event : paired.events) if (event.find("event")->string() == "token") event = with(event,"id",Json::integer(248044));
        paired.done = with(with(paired.done,"generated_ids",Json(Json::Array{Json::integer(248044)})),"stop_id",Json::integer(248044));
        admit(ready,paired);
    }

    const std::array proposed_stop{Round{0, 0, 1, true, 7}};
    admit(ready, make_transcript(ready, stop_previous, generation_job(8), policy(), 0, proposed_stop));
    auto append = make_transcript(ready, stop.done, input_job(3, false, policy(), 0), policy(), 0); admit(ready, append);
    auto reset = make_transcript(ready, append.done, input_job(8, true, policy(), 9), policy(), 9); admit(ready, reset);
    const auto selected = with(with(policy(), "top_k", Json::integer(7)), "min_p", number(0.1));
    admit(ready, make_transcript(ready, reset.done, input_job(2, true, selected, 17), selected, 17));
}
void context_checks() {
    const auto ready = ready_record(); Json previous; uint32_t prefix = 0;
    // Carry a generated, fully admitted input ancestry to each boundary.
    for (const uint32_t boundary : {2044u, 2048u, 2051u, 2052u, 4096u, 4099u,
                                   8192u, 8195u, 16387u, 32771u, 65528u}) {
        while (prefix < boundary) {
            const auto count = std::min(8u, boundary - prefix);
            auto fill = make_transcript(ready, previous, input_job(count, previous.is_null(), policy(), 0), policy(), 0);
            admit(ready, fill); previous = std::move(fill.done); prefix += count;
        }
        std::vector<std::vector<Round>> plans;
        plans.emplace_back(8, Round{0, 0, 1});
        for (uint32_t accepted : {1u, 3u, 6u, 7u}) {
            std::vector<Round> plan{{7, accepted, uint32_t(accepted < 7)}};
            for (uint32_t used = accepted + (accepted < 7); used < 8; ++used) plan.push_back({0, 0, 1});
            plans.push_back(std::move(plan));
        }
        for (const auto& plan : plans) {
            const auto t = make_transcript(ready, previous, generation_job(8), policy(), 0, plan);
            admit(ready, t);
            for (const int delta : {-1, 1}) {
                auto wrong = t;
                wrong.events[0] = with(wrong.events[0], "dispatches", Json::integer(uint64_t(int64_t(integer(wrong.events[0], "dispatches")) + delta)));
                reject(ready, wrong);
            }
        }
        if (boundary == 2051) {
            const std::vector<Round> serial(8, Round{0, 0, 1});
            auto old_short_rule = make_transcript(ready, previous, generation_job(8), policy(), 0, serial);
            check(integer(old_short_rule.events[0], "dispatches") == 2306, "token2052 includes48 selection dispatches");
            old_short_rule.events[0] = with(old_short_rule.events[0], "dispatches", Json::integer(2258));
            reject(ready, old_short_rule);
        }
    }
}
void command_checks() {
    const auto ready = ready_record(), previous = seed_prefix(ready).done, greedy = policy();
    const auto good = generation_job(2);
    auto refused = [&](Json job, const Json& before, uint32_t prefix, const Json& p = policy(), uint64_t seed = 0) {
        refuses([&] { Verify8Reply reply(ready, before, job, prefix, p, seed); });
    };
    for (const auto& [key, value] : Json::Object{{"generate", Json::integer(9)}, {"generate", Json::integer(0)},
        {"generate", Json(true)}, {"generate", parse_json("2.0")}, {"tokens", Json(Json::Array{Json::integer(1)})},
        {"reset", Json(true)}, {"stop_ids", Json(Json::Array{})}, {"stop_ids", Json(Json::Array{Json::integer(248046), Json::integer(2)})},
        {"stop_ids", Json(Json::Array{Json::integer(248320)})}, {"sampling", greedy}, {"seed", Json::integer(0)},
        {"prompt_lookup", Json(false)}, {"prompt_lookup", Json(true)}, {"checkpoint_mode", Json("auto")}, {"op", Json("branch_capture")}, {"unknown", Json(true)}})
        refused(with(good, key, value), previous, 16);
    for (const auto key : {"tokens", "generate", "reset", "stop_ids"}) refused(without(good, key), previous, 16);
    refused(good, Json(), 16); refused(good, previous, 15); refused(good, previous, 65537);
    refused(good, with(previous, "continuation_allowed", Json(false)), 16);
    refused(good, with(previous, "sampling", with(*previous.find("sampling"), "entropy_prefix", Json::integer(15))), 16);
    refused(good, previous, 16, greedy, 1);
    for (const auto& [key, value] : Json::Object{{"temperature", number(0.1)}, {"top_p", number(0)}, {"top_p", number(1.1)},
        {"min_p", number(-0.1)}, {"min_p", number(1.1)}, {"top_k", Json::integer(248321)}, {"top_k", Json(true)},
        {"temperature", parse_json("1e9999")}, {"unknown", Json::integer(0)}})
        refused(good, previous, 16, with(greedy, key, value));
    for (const auto n : {0u, 9u}) refused(input_job(n, true, greedy, 0), Json(), 0);
    refused(with(input_job(1, true, greedy, 0), "tokens", Json(Json::Array{Json::integer(248320)})), Json(), 0);
    refused(with(input_job(1, true, greedy, 0), "seed", Json(true)), Json(), 0);
    refused(with(input_job(1, true, greedy, 0), "sampling", with(greedy, "top_k", Json::integer(1))), Json(), 0);
    refused(with(input_job(1, true, greedy, 0), "stop_ids", Json(Json::Array{})), Json(), 0);
    refused(without(input_job(1, true, greedy, 0), "prompt_lookup"), Json(), 0);
    refused(with(input_job(1, true, greedy, 0), "prompt_lookup", Json(false)), Json(), 0);
    const auto end = with(with(previous, "committed_prefix", Json::integer(65535)), "sampling", with(*previous.find("sampling"), "entropy_prefix", Json::integer(65535)));
    refused(good, end, 65535); // Entire requested output reservation must fit.
}
void event_checks() {
    const auto ready = ready_record(), previous = seed_prefix(ready).done;
    const std::array full{Round{7, 7, 0}};
    const auto source = make_transcript(ready, previous, generation_job(7), policy(), 0, full);
    for (const auto& [key, value] : Json::Object{{"round_id", Json::integer(2)}, {"round_base", Json::integer(17)},
        {"committed_prefix", Json::integer(22)}, {"verified_tokens", Json::integer(8)}, {"verified_tokens", Json::integer(1)},
        {"accepted_verified_tokens", Json::integer(0)}, {"accepted_verified_tokens", Json::integer(8)},
        {"ordinary_tokens", Json::integer(1)}, {"proposed_tokens", Json::integer(6)}, {"proposed_tokens", Json::integer(8)},
        {"useful_tokens", Json::integer(8)}, {"verification_max_tokens", Json::integer(2)}, {"verification_head_dispatches", Json::integer(1)},
        {"dispatches", Json::integer(1)}, {"capture_executes", Json::integer(1)}, {"timings_qualified", Json(true)},
        {"gpu_profile", Json(Json::Object{})}, {"round_wall_ms", number(-1)}, {"round_wall_ms", parse_json("1e9999")},
        {"batch_gpu_ms", number(100)}, {"batch_gpu_ms", number(0)}, {"batch_wall_ms", number(100)},
        {"round_wall_ms", number(0)}, {"recovery_batch_wall_ms", number(1)}, {"fallback_wall_ms", number(1)},
        {"useful_tokens_per_second", number(0)}, {"accepted_tokens_per_second", number(0)}, {"unknown", Json(true)}}) {
        auto bad = source; bad.events[0] = with(bad.events[0], key, value); reject(ready, bad);
    }
    for (const auto& [key, ignored] : source.events[0].object()) { static_cast<void>(ignored); auto bad = source; bad.events[0] = without(bad.events[0], key); reject(ready, bad); }
    for (const auto& [key, value] : Json::Object{{"round_id", Json::integer(2)}, {"round_base", Json::integer(17)},
        {"token_prefix", Json::integer(18)}, {"committed_prefix", Json::integer(17)}, {"round_committed_prefix", Json::integer(17)},
        {"verified", Json(false)}, {"batched_commit", Json(false)}, {"entropy_words", Json::integer(1)},
        {"id", Json::integer(248320)}, {"id", Json::integer(201)}, {"id", Json(true)}, {"source_logit", Json("1.0")}, {"source_logit", number(1.5)},
        {"source_logit", parse_json("1e9999")}, {"gpu_profile", Json(Json::Object{})}, {"unknown", Json(true)}}) {
        auto bad = source; bad.events[1] = with(bad.events[1], key, value); reject(ready, bad);
    }
    for (const auto& [key, ignored] : source.events[1].object()) { static_cast<void>(ignored); auto bad = source; bad.events[1] = without(bad.events[1], key); reject(ready, bad); }
    { auto bad = source; bad.events.erase(bad.events.begin()); reject(ready, bad); }
    { auto bad = source; bad.events.insert(bad.events.begin() + 1, bad.events[0]); reject(ready, bad); }
    { auto bad = source; bad.events.pop_back(); reject(ready, bad); }
    { auto bad = source; std::swap(bad.events[1], bad.events[2]); reject(ready, bad); }
    { auto bad = source; bad.events[1] = with(bad.events[1], "id", Json::integer(248046)); reject(ready, bad); }
    { auto bad = source; bad.events.push_back(source.events[0]); reject(ready, bad); }
    { auto bad = source; bad.events[0] = with(bad.events[0], "hc_exact_round_routes", hc()); reject(ready, bad); }
    { auto bad = source; bad.events[0] = with(bad.events[0], "wide_q8_exact_round_routes", with(*ready.find("wide_q8_exact_routes"), "qkv_scalar", Json::integer(1))); reject(ready, bad); }
    { auto bad = source; bad.job = generation_job(2); reject(ready, bad); }
    const std::array partial{Round{7, 2, 1, true}};
    const auto correction = make_transcript(ready, previous, generation_job(8), policy(), 0, partial); admit(ready, correction);
    { auto bad = correction; bad.events.back() = with(bad.events.back(), "verified", Json(true)); reject(ready, bad); }
    { auto bad = correction; bad.events[0] = with(bad.events[0], "recovery_batch_wall_ms", number(0)); reject(ready, bad); }
    // An interrupted/uncertain response has no finish, hence no publishable span.
    Verify8Reply incomplete(ready, source.previous, source.job, source.before, source.sampling, source.seed);
    for (const auto& event : source.events) incomplete.event(event);
    refuses([&] { static_cast<void>(incomplete.events()); });
}
void done_checks() {
    const auto ready = ready_record(), previous = seed_prefix(ready).done;
    const std::array rounds{Round{2, 1, 1}};
    const auto source = make_transcript(ready, previous, generation_job(2), policy(), 0, rounds);
    for (const auto& [key, value] : Json::Object{{"committed_prefix", Json::integer(17)}, {"generated_ids", Json(Json::Array{})},
        {"status", Json("stopped")}, {"finish_reason", Json("stop")}, {"stop_id", Json::integer(248046)},
        {"continuation_allowed", Json(false)}, {"known_token_prefill", Json(false)}, {"known_token_capacity", Json::integer(4)},
        {"block_gemv_formats", Json::integer(1)}, {"hc_exact_final_logit_sha256", Json(std::string(64, 'A'))},
        {"hc_exact_final_logit_sha256", Json(std::string(63, 'a'))}, {"lookup_history_epoch", Json::integer(3)},
        {"lookup_request_appended_tokens", Json::integer(7)}, {"lookup_appended_tokens", Json::integer(1)},
        {"lookup_indexed_grams", Json::integer(0)}, {"lookup_request_indexed_grams", Json::integer(0)},
        {"lookup_rounds", Json::integer(2)}, {"lookup_verified_rounds", Json::integer(0)}, {"lookup_recovery_rounds", Json::integer(0)},
        {"lookup_accepted_tokens", Json::integer(2)}, {"lookup_ordinary_tokens", Json::integer(0)},
        {"lookup_round_wall_ms", number(0)}, {"generation_batch_gpu_ms", number(0)}, {"prefill_wall_ms", number(-1)},
        {"request_wall_ms", parse_json("1e9999")}, {"gpu_profile", Json(Json::Object{})}, {"final_logit_top8", Json(Json::Array{})}}) {
        auto bad = source; bad.done = with(bad.done, key, value); reject(ready, bad);
    }
    for (const auto key : {"tracked_metal_buffer_bytes", "peak_metal_buffer_bytes", "gpu_allocation_count", "gpu_buffer_limit_bytes", "imported_weight_backing_bytes"}) {
        auto bad = source; bad.done = with(bad.done, key, Json::integer(integer(bad.done, key) + 1)); reject(ready, bad);
    }
    for (const auto& [key, value] : Json::Object{{"seed", Json::integer(42)}, {"entropy_prefix", Json::integer(17)},
        {"entropy_words", Json::integer(1)}, {"temperature", number(1)}, {"valid", Json(false)}}) {
        auto bad = source; bad.done = with(bad.done, "sampling", with(*bad.done.find("sampling"), key, value)); reject(ready, bad);
    }
    { auto bad = source; bad.done = with(bad.done, "hc_exact_routes", hc()); reject(ready, bad); }
    { auto bad = source; bad.done = with(bad.done, "hc_exact_request_routes", hc()); reject(ready, bad); }
    { auto bad = source; auto ids = bad.done.find("generated_ids")->array(); ids[1] = Json::integer(42); bad.done = with(bad.done, "generated_ids", Json(ids)); reject(ready, bad); }
    for (const auto key : {"id", "logit", "unknown"}) {
        auto bad = source; auto top = bad.done.find("final_logit_top8")->array();
        top[1] = with(top[1], key, std::string_view(key) == "logit" ? parse_json("1e9999") : Json::integer(200));
        bad.done = with(bad.done, "final_logit_top8", Json(top)); reject(ready, bad);
    }
    { auto good = source; auto top = good.done.find("final_logit_top8")->array(); top[1] = with(top[1], "logit", number(1.25)); good.done = with(good.done, "final_logit_top8", Json(top)); admit(ready, good); }
    { auto bad = source; auto top = bad.done.find("final_logit_top8")->array(); std::swap(top[0], top[1]); bad.done = with(bad.done, "final_logit_top8", Json(top)); reject(ready, bad); }
}

void pause_ms(int duration) { while (poll(nullptr, 0, duration) < 0 && errno == EINTR) {} }
void emit(const Json& value) { std::cout << value.dump() << '\n' << std::flush; }
int fake_target(int argc, char** argv) {
    // Refuse a changed private startup command rather than silently accepting
    // the fixture's own expected ready fields under a different runtime policy.
    std::filesystem::path policy_path;
    const std::array<std::pair<std::string_view, std::string_view>, 13> fixed{{
        {"--allow-full-model", "true"}, {"--small-gemv-splits", "16"}, {"--block-gemv-formats", "3"},
        {"--stack-execution", "direct"}, {"--known-token-capacity", "8"}, {"--reuse-residual-projections", "true"},
        {"--prompt-lookup", "true"}, {"--verification-budget-bytes", "134217728"}, {"--allow-prompt-lookup-experiments", "true"},
        {"--verification-max-tokens", "8"}, {"--hc-exact", "combined"}, {"--wide-q8-exact", "off"}, {"--wide-q8-many", "off"}}};
    if (argc != 2 + 2 * int(fixed.size() + 2) + 4) return 81;
    for (const auto& [name, value] : fixed) {
        size_t found = 0;
        for (int i = 2; i + 1 < argc; ++i) if (std::string_view(argv[i]) == name && std::string_view(argv[i + 1]) == value) ++found;
        if (found != 1) return 82;
    }
    for (int i = 2; i + 1 < argc; ++i) if (std::string_view(argv[i]) == "--policy") policy_path = argv[i + 1];
    if (policy_path.empty()) return 83;
    const auto directory = policy_path.parent_path(), mode = policy_path.filename();
    { std::ofstream marker(directory / "started"); marker << "fixed private argv\n"; }
    const auto ready = ready_record(); emit(ready); Json previous, installed_policy; uint64_t seed = 0;
    std::string line;
    while (std::getline(std::cin, line)) {
        { std::ofstream count(directory / "commands", std::ios::app); count << "command\n"; }
        const auto job = parse_json(line);
        if (job.find("reset")->boolean()) { installed_policy = *job.find("sampling"); seed = integer(job, "seed"); }
        const auto n = uint32_t(integer(job, "generate")); std::vector<Round> rounds;
        for (uint32_t i = 0; i < n; ++i) rounds.push_back({0, 0, 1});
        auto t = make_transcript(ready, previous, job, installed_policy, seed, rounds);
        if (n && mode == "bad-round") t.events[0] = with(t.events[0], "round_id", Json::integer(2));
        for (const auto& event : t.events) emit(event);
        if (n) {
            { std::ofstream marker(directory / "events-sent"); marker << "native records emitted; terminal not yet sent\n"; }
            if (mode == "truncated") return 0;
            if (mode == "slow-done") {
                // Bounded handshake, not a race against a guessed sleep: the
                // parent observes emitted native events before terminal release.
                for (unsigned wait = 0; wait < 100 && !std::filesystem::exists(directory / "cancel-observed"); ++wait) pause_ms(5);
                if (!std::filesystem::exists(directory / "cancel-observed")) return 84;
            }
            if (mode == "bad-done") t.done = with(t.done, "hc_exact_final_logit_sha256", Json("not-a-complete-hash"));
        }
        emit(t.done); previous = std::move(t.done);
    }
    return 0;
}
struct WorkerFixture {
    std::filesystem::path directory;
    OwnedWorkerOptions options;
    WorkerFixture(const std::filesystem::path& executable, std::string_view mode) {
        auto name = (std::filesystem::temp_directory_path() / "tokspur-http-v8-XXXXXX").string();
        std::vector<char> writable(name.begin(), name.end()); writable.push_back('\0');
        const char* created = mkdtemp(writable.data()); if (!created) throw std::runtime_error("cannot create private V8 fixture directory"); directory = created;
        options.executable = directory / "verify8-target";
        std::error_code error; std::filesystem::create_hard_link(executable, options.executable, error);
        if (error) std::filesystem::copy_file(executable, options.executable);
        options.policy = directory / mode; options.library = directory / "owned.metallib";
        for (const auto& file : {options.policy, options.library}) { std::ofstream out(file); out << "CPU fixture, never a model\n"; }
        for (unsigned i = 0; i < 4; ++i) { const auto file = directory / ("fixture-shard-" + std::to_string(i)); { std::ofstream out(file); out << i; } options.shards.push_back(file); }
        options.verify8 = true; options.startup_timeout = 8s; options.step_timeout = 3s; options.drain_timeout = 800ms;
    }
    ~WorkerFixture() { std::error_code ignored; std::filesystem::remove_all(directory, ignored); }
    size_t commands() const { std::ifstream input(directory / "commands"); std::string line; size_t count = 0; while (std::getline(input, line)) ++count; return count; }
};
void worker_checks(const std::filesystem::path& executable) {
    {
        WorkerFixture f(executable, "normal"); OwnedWorker worker(f.options); size_t callbacks = 0;
        auto bad = input_job(1, true, with(policy(), "temperature", number(1)), 0);
        refuses([&] { worker.request(bad, Clock::now() + 3s, [] { return false; }); });
        check(f.commands() == 0 && worker.alive(), "non-greedy admission performed transport IO");
        worker.request(input_job(8, true, policy(), 0), Clock::now() + 3s, [] { return false; }, [&](const Json&) { ++callbacks; });
        check(callbacks == 0); worker.request(generation_job(8), Clock::now() + 3s, [] { return false; }, [&](const Json& event) {
            ++callbacks; check(std::filesystem::exists(f.directory / "events-sent"), "callback before completed native event batch");
            check(event.find("event")->string() == (callbacks % 2 ? "generation_round" : "token"), "worker reordered admitted callbacks");
        });
        check(callbacks == 16 && f.commands() == 2 && worker.alive()); worker.close(); check(!worker.alive());
    }
    for (const auto mode : {"bad-round", "bad-done", "truncated"}) {
        WorkerFixture f(executable, mode); OwnedWorker worker(f.options); size_t callbacks = 0;
        worker.request(input_job(8, true, policy(), 0), Clock::now() + 3s, [] { return false; });
        refuses([&] { worker.request(generation_job(2), Clock::now() + 3s, [] { return false; }, [&](const Json&) { ++callbacks; }); });
        check(callbacks == 0 && !worker.alive() && f.commands() == 2, "uncertain reply exposed tokens or kept admission");
        refuses([&] { worker.request(input_job(1, true, policy(), 0), Clock::now() + 3s, [] { return false; }); });
        check(f.commands() == 2, "failed private worker retried"); worker.close();
    }
    {
        WorkerFixture f(executable, "slow-done"); OwnedWorker worker(f.options); size_t callbacks = 0;
        worker.request(input_job(8, true, policy(), 0), Clock::now() + 3s, [] { return false; });
        bool cancelled = false;
        try { worker.request(generation_job(2), Clock::now() + 3s,
            [&] { if (!std::filesystem::exists(f.directory / "events-sent")) return false;
                std::ofstream observed(f.directory / "cancel-observed"); observed << "parent cancellation observed before done\n"; return true;
            }, [&](const Json&) { ++callbacks; }); }
        catch (const WorkerCancelled&) { cancelled = true; }
        check(cancelled && callbacks == 0 && worker.alive() && f.commands() == 2, "mid-response cancellation did not drain without publication");
        refuses([&] { worker.request(generation_job(1), Clock::now() + 3s, [] { return false; }); }); check(f.commands() == 2);
        worker.request(input_job(1, true, policy(), 0), Clock::now() + 3s, [] { return false; }); check(f.commands() == 3); worker.close();
    }
    for (const auto mode : {"wrong-basename", "disabled", "replay", "checkpoint"}) {
        WorkerFixture f(executable, "normal");
        if (std::string_view(mode) == "wrong-basename") f.options.executable = "/bin/sh";
        if (std::string_view(mode) == "disabled") f.options.verify8 = false;
        if (std::string_view(mode) == "replay") f.options.stack_execution = "recurrent-replay";
        if (std::string_view(mode) == "checkpoint") { f.options.checkpoint_slots = 1; f.options.checkpoint_budget_bytes = 1 << 20; }
        refuses([&] { OwnedWorker invalid(f.options); }); check(!std::filesystem::exists(f.directory / "started") && f.commands() == 0);
    }
}
std::vector<uint32_t> encode_text(std::string_view text) {
    std::vector<uint32_t> result;
    for (size_t at = 0; at < text.size();) {
        if (text.substr(at).starts_with("<|im_end|>")) { result.push_back(248046); at += 10; }
        else result.push_back(uint32_t(static_cast<unsigned char>(text[at++])) + 1);
    }
    return result;
}
ServeCodec session_codec() {
    return {[](const Json& request) {
        auto text = std::string(request.find("messages")->array().back().find("content")->string());
        for (size_t at = 0; (at = text.find("[fixture_end]", at)) != std::string::npos;) text.replace(at, 13, "<|im_end|>");
        return RenderedChat{std::move(text), Json(Json::Array{}), 0, 0};
    }, [](const RenderedChat& rendered) { return encode_text(rendered.prompt); }, [](std::span<const uint32_t> ids) {
        std::string text;
        for (const auto id : ids) {
            if (id == 248046) text += "<|im_end|>";
            else { check(id > 0 && id < 257); text += char(id - 1); }
        }
        return text;
    }, {248320, 248046, 248044, 248045}};
}
Json chat(std::string text, bool stream = false, uint32_t maximum = 8) {
    for (size_t at = 0; (at = text.find("<|im_end|>", at)) != std::string::npos;) text.replace(at, 10, "[fixture_end]");
    return Json(Json::Object{{"model", Json("tokspur")},
        {"messages", Json(Json::Array{Json(Json::Object{{"role", Json("user")}, {"content", Json(std::move(text))}})})},
        {"stream", Json(stream)}, {"enable_thinking", Json(false)}, {"temperature", Json::integer(0)},
        {"seed", Json::integer(0)}, {"max_tokens", Json::integer(maximum)}});
}
struct SessionFake {
    Json ready = ready_record(), previous, sampling;
    uint64_t seed = 0;
    std::vector<uint32_t> output;
    size_t next = 0, commands = 0, resets = 0;
    bool bad_done = false;
    void plan(std::string_view text) { output = encode_text(text); next = 0; }
    Json operator()(const Json& job, const std::function<void(const Json&)>& callback) {
        ++commands;
        if (job.find("reset")->boolean()) { ++resets; sampling = *job.find("sampling"); seed = integer(job, "seed"); }
        const auto requested = uint32_t(integer(job, "generate")); std::vector<Round> rounds;
        const size_t first = next;
        for (uint32_t i = 0; i < requested && next < output.size(); ++i, ++next) {
            const bool terminal = output[next] == 248046; rounds.push_back({0, 0, 1, terminal});
            if (terminal) { ++next; break; }
        }
        auto transcript = make_transcript(ready, previous, job, sampling, seed, rounds);
        Json::Array ids; size_t at = first;
        for (auto& event : transcript.events) if (event.find("event")->string() == "token") {
            ids.push_back(Json::integer(output.at(at))); event = with(std::move(event), "id", Json::integer(output.at(at++)));
        }
        transcript.done = with(std::move(transcript.done), "generated_ids", Json(std::move(ids)));
        auto top = transcript.done.find("final_logit_top8")->array();
        top[0] = with(top[0], "id", Json::integer(next < output.size() ? output[next] : 200));
        transcript.done = with(std::move(transcript.done), "final_logit_top8", Json(top));
        for (const auto& event : transcript.events) callback(event);
        previous = transcript.done; // Native state commits even if caller cancels after this done.
        if (requested && bad_done) return with(transcript.done, "hc_exact_final_logit_sha256", Json("uncertain"));
        return transcript.done;
    }
};
void session_checks() {
    const auto ready = ready_record(); ServeSession session(session_codec(), "tokspur", "direct", Json(), true, ready);
    SessionFake fake; fake.plan("ok<|im_end|>"); bool cancelled = false;
    std::vector<Json> chunks;
    const ServeSink sink{[&](const Json& value) { chunks.push_back(value); }, [&] { return cancelled; }, [] { return 0; }};
    const ServeTarget target = [&](const Json& job, const auto& event) { return fake(job, event); };
    refuses([&] { session.complete(with(chat("A"), "temperature", number(0.1)), "non-greedy", target, sink); });
    check(fake.commands == 0, "HTTP non-greedy refusal contacted target");
    const ServeTarget cancel_after_done = [&](const Json& job, const auto& event) {
        auto done = fake(job, event); if (integer(job, "generate")) cancelled = true; return done;
    };
    refuses([&] { session.complete(chat("A", true), "cancel-after-done", cancel_after_done, sink); });
    check(session.committed_tokens() == 0 && fake.commands == 2, "session retained uncertain cancelled prefix");
    for (const auto& value : chunks) {
        const auto* delta = value.find("choices")->array()[0].find("delta");
        check(!delta->find("tool_calls") && !delta->find("content"), "cancelled native batch released visible content/tools");
    }
    cancelled = false; chunks.clear(); fake.plan("ok<|im_end|>");
    auto result = session.complete(chat("AB"), "cold-recover", target, sink);
    check(result.find("choices")->array()[0].find("finish_reason")->string() == "stop" && fake.resets == 2);
    const auto reset_count = fake.resets; fake.plan("ok<|im_end|>");
    result = session.complete(chat("ABok<|im_end|>C"), "append-after-recover", target, sink);
    check(fake.resets == reset_count && result.find("usage")->find("prompt_tokens_details")->find("cached_tokens")->unsigned_integer() == 5,
        "fresh reset failed to reestablish exact append ancestry");
    const auto tool_request = with(chat("T", true, 10), "tools", parse_json(R"([{"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"],"additionalProperties":false}}}])"));
    fake.plan("<tool_call><function=read_file><parameter=path>\nx\n</parameter></function></tool_call><|im_end|>"); chunks.clear();
    result = session.complete(tool_request, "tool-cutoff", target, sink);
    const auto& choice = result.find("choices")->array()[0];
    check(choice.find("finish_reason")->string() == "length" && !choice.find("message")->find("tool_calls"), "length-cutoff frame released a tool");
    for (const auto& value : chunks) check(!value.find("choices")->array()[0].find("delta")->find("tool_calls"));
    fake.plan("ok<|im_end|>"); fake.bad_done = true; chunks.clear();
    refuses([&] { session.complete(chat("X", true), "bad-batch", target, sink); });
    check(session.committed_tokens() == 0);
    for (const auto& value : chunks) check(!value.find("choices")->array()[0].find("delta")->find("content"), "malformed batch released content");
}
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc > 1 && std::string_view(argv[1]) == "target") return fake_target(argc, argv);
        ready_checks(); positive_checks(); context_checks(); command_checks(); event_checks(); done_checks();
        session_checks();
        worker_checks(std::filesystem::canonical(argv[0]));
        std::cout << "Private HTTP V8 CPU checks=" << checks << '\n'; return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
