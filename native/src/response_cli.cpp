#include "tokspur/text_assets.hpp"
#include "tokspur/text.hpp"
#include "tokspur/response_protocol.hpp"
#include "tokspur/gguf.hpp"

#include <iostream>
#include <set>

namespace tokspur {
namespace {
constexpr std::string_view tokenizer_pin = text_assets::tokenizer_sha256;
void require(bool ok, const char* message) { if (!ok) throw FormatError(message); }
bool read_line(std::istream& input, std::string& line) {
    line.clear(); bool any = false, oversized = false; char c;
    while (input.get(c)) {
        any = true; if (c == '\n') break;
        if (line.size() < response_request_bytes) line.push_back(c); else oversized = true;
    }
    require(!oversized, "response request exceeds 1-MiB bound"); return any;
}
}
int response_cli(int argc, char** argv) {
    std::filesystem::path tokenizer_path; std::string pin; std::set<std::string_view> seen;
    for (int i = 2; i < argc; ++i) {
        const std::string_view option(argv[i]);
        require(seen.insert(option).second && ++i < argc, "duplicate response option or missing value");
        if (option == "--tokenizer") tokenizer_path = argv[i];
        else if (option == "--sha256") pin = argv[i];
        else throw FormatError("unknown response option");
    }
    require(!tokenizer_path.empty() && pin == tokenizer_pin, "response requires the supported pinned tokenizer and explicit SHA256");
    NativeText tokenizer(tokenizer_path, pin);
    const auto token = [&](std::string_view marker) {
        const auto ids = tokenizer.encode(marker);
        require(ids.size() == 1 && tokenizer.decode_bytes(ids) == marker, "response marker is not one exact pinned tokenizer token"); return ids[0];
    };
    const ResponseAdmission admission{uint32_t(tokenizer.vocabulary_size()), token("<|im_end|>"), token("<|endoftext|>"), token("<|im_start|>")};
    std::cout << Json(Json::Object{{"event", Json("ready")}, {"scope", Json("model-free accumulated-response parser; no tools are executed")},
        {"tokenizer_sha256", Json(pin)}, {"vocabulary_size", Json::integer(admission.vocabulary)}, {"im_end_id", Json::integer(admission.im_end)},
        {"end_of_text_id", Json::integer(admission.end_of_text)}, {"im_start_id", Json::integer(admission.im_start)},
        {"maximum_ids", Json::integer(response_maximum_ids)}, {"maximum_request_bytes", Json::integer(response_request_bytes)}, {"tools_executed", Json(false)}}).dump() << std::endl;
    std::string line; bool failed = false;
    while (true) {
        try {
            if (!read_line(std::cin, line)) break;
            const auto input = parse_json(line, {response_request_bytes, 150000, 32});
            const auto output = parse_native_response(input, admission, [&](auto ids) { return tokenizer.decode_bytes(ids); });
            std::cout << output.dump(response_output_bytes) << std::endl;
        } catch (const FormatError& error) {
            failed = true; std::cout << native_response_error(error.what()).dump(8192) << std::endl;
            if (!std::cin) break;
        } catch (const std::exception& error) {
            std::cout << native_response_error(error.what()).dump(8192) << std::endl; return 1;
        }
    }
    return failed ? 1 : 0;
}
} // namespace tokspur
