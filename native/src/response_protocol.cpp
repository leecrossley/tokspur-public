#include "tokspur/response_protocol.hpp"
#include "tokspur/response.hpp"
#include "tokspur/gguf.hpp"

#include <set>

namespace tokspur {
namespace {
void require(bool ok, const char* message) { if (!ok) throw FormatError(message); }
const Json& field(const Json& object, std::string_view name) {
    const auto* value = object.find(name); require(value, "missing explicit response field"); return *value;
}
void keys(const Json& object, const std::set<std::string_view>& allowed) {
    for (const auto& [key, value] : object.object()) {
        static_cast<void>(value); require(allowed.contains(key), "unknown response protocol field");
    }
}
std::vector<ToolSpec> tools(const Json& value) {
    const auto& manifest = value.array(); require(manifest.size() <= 64, "response tool manifest limit");
    std::vector<ToolSpec> result; result.reserve(manifest.size());
    for (const auto& entry : manifest) {
        keys(entry, {"name", "parameters"}); ToolSpec tool{std::string(field(entry, "name").string()), {}};
        const auto& parameters = field(entry, "parameters").array(); require(parameters.size() <= 32, "response parameter manifest limit");
        tool.parameters.reserve(parameters.size());
        for (const auto& parameter : parameters) {
            keys(parameter, {"name", "type", "required"});
            const auto type = field(parameter, "type").string(); ParameterType kind;
            if (type == "string") kind = ParameterType::raw_string;
            else if (type == "boolean") kind = ParameterType::boolean;
            else if (type == "number") kind = ParameterType::number;
            else if (type == "object") kind = ParameterType::object;
            else if (type == "array") kind = ParameterType::array;
            else if (type == "null") kind = ParameterType::null_value;
            else if (type == "any") kind = ParameterType::json_value;
            else throw FormatError("unsupported response parameter type");
            tool.parameters.push_back({std::string(field(parameter, "name").string()), kind, field(parameter, "required").boolean()});
        }
        result.push_back(std::move(tool));
    }
    return result;
}
Json result(std::string_view status, std::string_view finish, bool complete,
            std::string content, std::string thinking, Json::Array calls) {
    return Json(Json::Object{{"event", Json("response")}, {"status", Json(std::string(status))},
        {"finish_reason", Json(std::string(finish))}, {"complete", Json(complete)},
        {"content", Json(std::move(content))}, {"reasoning_content", Json(std::move(thinking))},
        {"tool_calls", Json(std::move(calls))}, {"tools_executed", Json(false)}});
}
}
Json native_response_error(std::string_view message) {
    auto out = result("error", "error", false, {}, {}, {});
    auto fields = out.object(); fields.emplace_back("message", Json(std::string(message.substr(0, 2048)))); return Json(std::move(fields));
}
Json parse_native_response(const Json& data, ResponseAdmission admission, const ResponseDecoder& decode, size_t maximum_ids) {
    require(admission.vocabulary && admission.vocabulary <= 1'048'576 && admission.im_end < admission.vocabulary &&
        admission.end_of_text < admission.vocabulary && admission.im_start < admission.vocabulary &&
        admission.im_end != admission.end_of_text && admission.im_end != admission.im_start && admission.end_of_text != admission.im_start,
        "invalid tokenizer-derived response admission");
    keys(data, {"ids", "tools", "initial_thinking", "finish_reason", "stop_id", "executor_parameters"});
    const bool executor_parameters = data.find("executor_parameters") ? data.find("executor_parameters")->boolean() : false;
    require(maximum_ids && maximum_ids <= response_maximum_accumulated_ids, "invalid response ID admission bound");
    const auto& input = field(data, "ids").array(); require(input.size() <= maximum_ids, "response ID count limit");
    const auto finish = field(data, "finish_reason").string();
    require(finish == "stop" || finish == "length" || finish == "error", "unknown response finish reason");
    const auto& stop = field(data, "stop_id");
    if (finish == "stop") require(!stop.is_null() && admission.is_stop(stop.unsigned_integer()), "response stop ID is not an admitted terminal token");
    else require(stop.is_null(), "non-stop response must have null stop_id");
    std::vector<uint32_t> ids; ids.reserve(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        const auto id = input[i].unsigned_integer(); require(id < admission.vocabulary, "response token outside vocabulary");
        require(id != admission.im_start, "unexpected response terminal or role marker");
        require(!admission.is_stop(id) || (finish == "stop" && i + 1 == input.size() && id == stop.unsigned_integer()), "earlier or unacknowledged response terminal token");
        ids.push_back(uint32_t(id));
    }
    if (finish == "stop") require(!ids.empty() && ids.back() == stop.unsigned_integer(), "stopped response lacks exact final stop token");
    ResponseParser parser(tools(field(data, "tools")), field(data, "initial_thinking").boolean(), {}, executor_parameters, true);
    if (finish == "error") return native_response_error("target reported an error; response is not executable");
    require(bool(decode), "missing raw response decoder"); auto bytes = decode(ids);
    require(bytes.size() <= response_request_bytes, "decoded response byte limit");
    if (finish == "stop") {
        const std::string_view marker = stop.unsigned_integer() == admission.im_end ? "<|im_end|>" : "<|endoftext|>";
        require(bytes.ends_with(marker), "response decoder/terminal identity mismatch"); bytes.resize(bytes.size() - marker.size());
    }
    require(parser.feed(bytes) != ResponseStatus::error, "invalid response bytes or frames");
    if (finish == "stop") require(parser.finish() == ResponseStatus::complete, "incomplete or invalid stopped response");
    std::string content, thinking; Json::Array calls;
    for (const auto& record : parser.records()) {
        if (record.kind == ResponseKind::text) content += record.text;
        else if (record.kind == ResponseKind::thinking) thinking += record.text;
        else {
            require(finish == "stop" && parser.status() == ResponseStatus::complete, "tool record before terminal admission");
            calls.push_back(Json(Json::Object{{"ordinal", Json::integer(record.ordinal)}, {"type", Json("function")},
                {"function", Json(Json::Object{{"name", Json(record.name)}, {"arguments", record.arguments}})}}));
        }
    }
    return result(finish == "stop" ? "complete" : "incomplete", finish, finish == "stop", std::move(content), std::move(thinking), std::move(calls));
}
} // namespace tokspur
