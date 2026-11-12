#include "tokspur/gguf.hpp"
#include "tokspur/numeric.hpp"
#include "tokspur/model.hpp"
#include "tokspur/derived_pack.hpp"
#include "tokspur/planar_pack_v3.hpp"
#ifdef TOKSPUR_HAS_METAL
#include "tokspur/metal.hpp"
#include "tokspur/graph.hpp"
#include "tokspur/ffn_check.hpp"
#include "tokspur/recurrent_check.hpp"
#include "tokspur/attention_check.hpp"
#include "tokspur/memory.hpp"
#include "tokspur/target.hpp"
#include "tokspur/text.hpp"
#include "tokspur/q8_many.hpp"
#include "tokspur/gdn_core_tile.hpp"
#include "tokspur/recurrent_tile.hpp"
#include "tokspur/recurrent_timing.hpp"
#include "tokspur/recurrent_replay_timing.hpp"
#endif

#include <filesystem>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <string_view>

namespace {
void quote(std::string_view value) {
    std::cout << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') std::cout << '\\' << c;
        else if (c < 0x20) {
            constexpr char hex[] = "0123456789abcdef";
            std::cout << "\\u00" << hex[c >> 4] << hex[c & 15];
        } else std::cout << c;
    }
    std::cout << '"';
}

uint32_t positive_integer(std::string_view text, bool allow_zero = false) {
    uint32_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || (!allow_zero && !value)) {
        throw tokspur::FormatError("expected a bounded positive integer option");
    }
    return value;
}

struct Options {
    std::string tensor;
    uint32_t row = 0;
    uint32_t rows = 0;
    uint32_t repeats = 5;
    std::filesystem::path library;
    std::vector<std::filesystem::path> paths;
};

Options options(int argc, char** argv) {
    Options result;
    result.library = std::filesystem::absolute(argv[0]).parent_path() / "tokspur-owned.metallib";
    for (int i = 2; i < argc; ++i) {
        const std::string_view flag(argv[i]);
        if (!flag.starts_with("--")) { result.paths.emplace_back(argv[i]); continue; }
        if (++i == argc) throw tokspur::FormatError("missing option value");
        const std::string_view argument(argv[i]);
        if (flag == "--tensor") result.tensor = argument;
        else if (flag == "--row") result.row = positive_integer(argument, true);
        else if (flag == "--rows") result.rows = positive_integer(argument);
        else if (flag == "--repeats") result.repeats = positive_integer(argument);
        else if (flag == "--library") result.library = argument;
        else throw tokspur::FormatError("unknown option");
    }
    if (result.paths.empty()) throw tokspur::FormatError("no GGUF shards supplied");
    return result;
}

int operator_check(std::string_view command, const Options& args, const tokspur::ModelFiles& model) {
    const auto& tensor = model.tensor(args.tensor);
    const auto columns64 = tensor.dimensions.front();
    const auto available_rows = tensor.elements / columns64;
    if (columns64 > 65536 || args.row >= available_rows) throw tokspur::FormatError("operator shape/row limit exceeded");
    const auto columns = static_cast<uint32_t>(columns64);
    const auto& layout = tokspur::encoding(tensor.type);
    const auto row_bytes = tokspur::checked_mul(columns / layout.elements_per_block, layout.bytes_per_block);
    const auto rows = command == "row" ? 1 : (args.rows ? args.rows : static_cast<uint32_t>(std::min(uint64_t(64), available_rows - args.row)));
    if (rows > 16384 || rows > available_rows - args.row || tokspur::checked_mul(row_bytes, rows) > 256 * 1024 * 1024) {
        throw tokspur::FormatError("operator slice exceeds tensor or component memory limit");
    }
    const auto matrix = model.tensor_bytes(args.tensor).subspan(static_cast<size_t>(args.row * row_bytes), static_cast<size_t>(rows * row_bytes));
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (command == "row") {
        std::vector<float> decoded(columns);
        tokspur::dequantize(tensor.type, matrix, decoded);
        std::cout << "{\"schema\":1,\"tensor\":";
        quote(args.tensor);
        std::cout << ",\"type\":" << tensor.type << ",\"row\":" << args.row << ",\"values\":[";
        for (size_t i = 0; i < decoded.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << decoded[i];
        }
        std::cout << "]}\n";
        model.assert_unchanged();
        return 0;
    }
#ifdef TOKSPUR_HAS_METAL
    std::vector<float> input(columns);
    for (size_t i = 0; i < input.size(); ++i) input[i] = (int((i * 17 + 3) % 257) - 128) / 128.0f;
    const auto cpu_start = std::chrono::steady_clock::now();
    const auto expected = tokspur::reference_gemv(tensor.type, matrix, columns, rows, input);
    const double cpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cpu_start).count();
    const auto metal_start = std::chrono::steady_clock::now();
    tokspur::MetalResult actual;
    std::string device;
    if (command == "verify-persistent") {
        if (!args.repeats || args.repeats > 1000) throw tokspur::FormatError("unsupported persistent repetition count");
        tokspur::MetalGraph graph(args.library, 512 * 1024 * 1024);
        device = graph.device_name();
        const auto weights = graph.upload(matrix, "component weights");
        const auto x = graph.upload(std::as_bytes(std::span(input)), "component input");
        const auto y = graph.zeros(rows * sizeof(float), "component output");
        actual.buffer_bytes = graph.memory().live_bytes;
        actual.setup_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - metal_start).count();
        std::vector<double> walls, executions;
        const auto allocations = graph.memory().allocations;
        for (uint32_t iteration = 0; iteration <= args.repeats; ++iteration) {
            graph.begin();
            graph.gemv(tensor.type, weights, x, y, columns, rows);
            const auto batch = graph.submit();
            if (graph.memory().allocations != allocations || batch.dispatches != 1) throw tokspur::FormatError("persistent GEMV resource invariant failed");
            if (!iteration) actual.first_submission_ms = batch.wall_ms;
            else { walls.push_back(batch.wall_ms); executions.push_back(batch.gpu_ms); }
        }
        const auto median = [](std::vector<double> values) {
            std::sort(values.begin(), values.end());
            const auto middle = values.size() / 2;
            return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
        };
        actual.median_submission_ms = median(walls);
        actual.median_gpu_ms = median(executions);
        const auto bytes = graph.read(y);
        actual.output.resize(rows);
        std::memcpy(actual.output.data(), bytes.data(), bytes.size());
        for (const float value : actual.output) if (!std::isfinite(value)) throw tokspur::FormatError("nonfinite persistent GEMV output");
    } else {
        tokspur::MetalGemv gpu(args.library);
        device = gpu.device_name();
        actual = gpu.run(tensor.type, matrix, columns, rows, input, args.repeats);
    }
    const double total_metal_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - metal_start).count();
    double worst_absolute = 0;
    double worst_scaled = 0;
    for (size_t i = 0; i < rows; ++i) {
        const auto error = std::abs(expected[i] - actual.output[i]);
        worst_absolute = std::max(worst_absolute, error);
        worst_scaled = std::max(worst_scaled, error / std::max(1.0, std::abs(expected[i])));
    }
    constexpr double tolerance = 5e-5;
    model.assert_unchanged();
    std::cout << "{\"schema\":1,\"scope\":\"single owned operator; not full inference\",\"tensor\":";
    quote(args.tensor);
    std::cout << ",\"device\":";
    quote(device);
    std::cout << ",\"persistent_buffers\":" << (command == "verify-persistent" ? "true" : "false");
    std::cout << ",\"type\":" << tensor.type << ",\"columns\":" << columns << ",\"rows\":" << rows
              << ",\"first_row\":" << args.row << ",\"repetitions\":" << args.repeats
              << ",\"max_absolute_error\":" << worst_absolute << ",\"max_scaled_error\":" << worst_scaled
              << ",\"scaled_tolerance\":" << tolerance << ",\"buffer_bytes\":" << actual.buffer_bytes
              << ",\"cpu_reference_ms\":" << cpu_ms << ",\"metal_total_ms\":" << total_metal_ms
              << ",\"pipeline_and_buffer_setup_ms\":" << actual.setup_ms << ",\"first_submission_ms\":" << actual.first_submission_ms
              << ",\"median_submission_ms\":" << actual.median_submission_ms << ",\"median_gpu_ms\":" << actual.median_gpu_ms
              << ",\"pass\":" << (worst_scaled <= tolerance ? "true" : "false") << "}\n";
    return worst_scaled <= tolerance ? 0 : 1;
#else
    throw tokspur::FormatError("verify requires the owned macOS Metal build");
#endif
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "host-info") {
#ifdef TOKSPUR_HAS_METAL
            std::cout << "{\"thermal_state\":" << tokspur::host_thermal_state() << "}\n"; return 0;
#else
            throw tokspur::FormatError("host thermal query requires the owned macOS build");
#endif
        }
        if (argc >= 2 && std::string_view(argv[1]) == "mapping-check") {
#ifdef TOKSPUR_HAS_METAL
            return tokspur::mapping_check_cli(argc, argv);
#else
            throw tokspur::FormatError("mapped-weight check requires the owned macOS Metal build");
#endif
        }
        if (argc >= 2 && std::string_view(argv[1]) == "layout-check") {
#ifdef TOKSPUR_HAS_METAL
            return tokspur::layout_check_cli(argc, argv);
#else
            throw tokspur::FormatError("bounded real-weight layout check requires the owned macOS Metal build");
#endif
        }
        if (argc >= 2 && std::string_view(argv[1]) == "recurrent-tile-check") {
#ifdef TOKSPUR_HAS_METAL
            return tokspur::recurrent_tile_check_cli(argc, argv);
#else
            throw tokspur::FormatError("the recurrent tile check requires the owned macOS build");
#endif
        }
        if (argc >= 2 && std::string_view(argv[1]) == "recurrent-timing") {
#ifdef TOKSPUR_HAS_METAL
            return tokspur::recurrent_timing_cli(argc, argv);
#else
            throw tokspur::FormatError("the recurrent timing worker requires the owned macOS build");
#endif
        }
        if (argc >= 2 && std::string_view(argv[1]) == "recurrent-replay-timing") {
#ifdef TOKSPUR_HAS_METAL
            return tokspur::recurrent_replay_timing_cli(argc, argv);
#else
            throw tokspur::FormatError("the recurrent replay timing worker requires the owned macOS build");
#endif
        }
        if (argc >= 2 && std::string_view(argv[1]) == "gdn-tile-check") {
#ifdef TOKSPUR_HAS_METAL
            return tokspur::gdn_core_tile_check_cli(argc, argv);
#else
            throw tokspur::FormatError("the GDN core tile check requires the owned macOS build");
#endif
        }
        if (argc >= 2 && std::string_view(argv[1]) == "q8-many-check") {
#ifdef TOKSPUR_HAS_METAL
            return tokspur::q8_many_check_cli(argc, argv);
#else
            throw tokspur::FormatError("the Q8-many check requires the owned macOS build");
#endif
        }
        if (argc >= 2 && (std::string_view(argv[1]) == "pack-v3-plan" ||
                          std::string_view(argv[1]) == "pack-v3-build")) {
            return tokspur::planar_v3::pack_v3_cli(argc, argv);
        }
        if (argc >= 2 && std::string_view(argv[1]) == "pack-build") {
            std::filesystem::path output_path;
            std::optional<std::filesystem::path> mtp_path;
            uint64_t reserve_bytes = tokspur::derived_pack::default_reserve_bytes;
            uint64_t chunk_bytes = tokspur::derived_pack::default_chunk_bytes;
            std::vector<std::filesystem::path> paths;
            for (int i = 2; i < argc; ++i) {
                std::string_view arg(argv[i]);
                if (arg == "--output" && i + 1 < argc) {
                    output_path = argv[++i];
                } else if (arg == "--mtp" && i + 1 < argc) {
                    mtp_path = argv[++i];
                } else if (arg == "--reserve-bytes" && i + 1 < argc) {
                    reserve_bytes = std::stoull(argv[++i]);
                } else if (arg == "--chunk-bytes" && i + 1 < argc) {
                    chunk_bytes = std::stoull(argv[++i]);
                } else if (!arg.starts_with("--")) {
                    paths.emplace_back(argv[i]);
                } else {
                    throw tokspur::FormatError("unknown pack-build option");
                }
            }
            if (output_path.empty() || paths.empty()) {
                throw tokspur::FormatError("pack-build requires --output and at least one model shard");
            }
            auto source = std::make_shared<tokspur::ModelFiles>(paths);
            std::shared_ptr<tokspur::ModelFiles> mtp;
            if (mtp_path) {
                mtp = std::make_shared<tokspur::ModelFiles>(std::vector<std::filesystem::path>{*mtp_path});
            }
            const auto plan = tokspur::derived_pack::build_derived_pack(source, mtp, output_path, reserve_bytes, chunk_bytes);
            const auto adopted = tokspur::derived_pack::adopt_derived_pack(source, mtp, output_path, chunk_bytes, 16);
            std::cout << "{\"status\":\"built\",\"recipe\":\"" << plan.recipe << "\",\"tensors\":" << plan.tensors.size()
                      << ",\"payload_bytes\":" << plan.payload_bytes << ",\"file_bytes\":" << plan.file_bytes
                      << ",\"manifest_sha256\":\"" << adopted.manifest_sha256 << "\"}\n";
            return 0;
        }
        if (argc >= 2 && std::string_view(argv[1]) == "pack-adopt") {
            std::filesystem::path pack_path;
            std::optional<std::filesystem::path> mtp_path;
            uint32_t workers = 16;
            std::vector<std::filesystem::path> paths;
            for (int i = 2; i < argc; ++i) {
                std::string_view arg(argv[i]);
                if (arg == "--pack" && i + 1 < argc) {
                    pack_path = argv[++i];
                } else if (arg == "--mtp" && i + 1 < argc) {
                    mtp_path = argv[++i];
                } else if (arg == "--workers" && i + 1 < argc) {
                    workers = uint32_t(std::stoul(argv[++i]));
                } else if (!arg.starts_with("--")) {
                    paths.emplace_back(argv[i]);
                } else {
                    throw tokspur::FormatError("unknown pack-adopt option");
                }
            }
            if (pack_path.empty() || paths.empty()) {
                throw tokspur::FormatError("pack-adopt requires --pack and at least one model shard");
            }
            auto source = std::make_shared<tokspur::ModelFiles>(paths);
            std::shared_ptr<tokspur::ModelFiles> mtp;
            if (mtp_path) {
                mtp = std::make_shared<tokspur::ModelFiles>(std::vector<std::filesystem::path>{*mtp_path});
            }
            const auto adopted = tokspur::derived_pack::adopt_derived_pack(source, mtp, pack_path, tokspur::derived_pack::default_chunk_bytes, workers);
            std::cout << "{\"status\":\"adopted\",\"tensors\":" << adopted.plan.tensors.size()
                      << ",\"file_bytes\":" << adopted.plan.file_bytes
                      << ",\"manifest_sha256\":\"" << adopted.manifest_sha256 << "\"}\n";
            return 0;
        }
        if (argc >= 2 && std::string_view(argv[1]) == "target") {
#ifdef TOKSPUR_HAS_METAL
            return tokspur::target_cli(argc, argv);
#else
            throw tokspur::FormatError("full target requires the owned macOS Metal build");
#endif
        }
        if (argc >= 2 && std::string_view(argv[1]) == "memory-check") {
#ifdef TOKSPUR_HAS_METAL
            return tokspur::memory_check_cli(argc, argv);
#else
            throw tokspur::FormatError("memory check requires the owned macOS Metal build");
#endif
        }
        if (argc >= 2 && std::string_view(argv[1]) == "attention-check") {
#ifdef TOKSPUR_HAS_METAL
            return tokspur::attention_check_cli(argc, argv);
#else
            throw tokspur::FormatError("attention check requires the owned macOS Metal build");
#endif
        }
        if (argc >= 2 && std::string_view(argv[1]) == "recurrent-check") {
#ifdef TOKSPUR_HAS_METAL
            return tokspur::recurrent_check_cli(argc, argv);
#else
            throw tokspur::FormatError("recurrent check requires the owned macOS Metal build");
#endif
        }
        if (argc >= 2 && std::string_view(argv[1]) == "ffn-check") {
#ifdef TOKSPUR_HAS_METAL
            return tokspur::ffn_check_cli(argc, argv);
#else
            throw tokspur::FormatError("FFN check requires the owned macOS Metal build");
#endif
        }
        if (argc >= 2 && std::string_view(argv[1]) == "response") {
#ifdef TOKSPUR_HAS_METAL
            return tokspur::response_cli(argc, argv);
#else
            throw tokspur::FormatError("the response tokenizer adapter requires the owned macOS build");
#endif
        }
        if (argc >= 2 && (std::string_view(argv[1]) == "text" || std::string_view(argv[1]) == "chat" || std::string_view(argv[1]) == "http-replay")) {
#ifdef TOKSPUR_HAS_METAL
            return tokspur::text_cli(argc, argv);
#else
            throw tokspur::FormatError("the text adapter requires the owned macOS build");
#endif
        }
        if (argc < 3) {
            std::cerr << "usage: tokspur-native inspect|tensors|layout|row|verify|verify-persistent [--tensor NAME --row N --rows N --repeats N --library PATH] <GGUF shards...>\n"
                         "       tokspur-native ffn-check --router-mode softmax|sigmoid --normalize-selected true|false [--layer N --token ID --repeats N] <GGUF shards...>\n"
                         "       tokspur-native recurrent-check --router-mode softmax|sigmoid --normalize-selected true|false --query-scale unit|inverse-sqrt-key-width --l2-mode add|floor [--layer N --token ID --repeats N] <GGUF shards...>\n"
                         "       tokspur-native gdn-tile-check --query-scale unit|inverse-sqrt-key-width --l2-mode add|floor [--layer 0|2 --capacity N --steps N --delta-group 32|64|128] <GGUF shards...>\n"
                         "       tokspur-native recurrent-tile-check --router-mode softmax|sigmoid --normalize-selected true|false --query-scale unit|inverse-sqrt-key-width --l2-mode add|floor [--layer 0|2 --capacity N --steps N --delta-group 32|64|128 --block-formats 1|3 --small-gemv-splits 0|1|4|8|16|32 --reuse-residual-projections true|false] <GGUF shards...>\n"
                         "       tokspur-native recurrent-timing --layer 0|2 --delta-group 32|64|128 <four GGUF shards...>\n"
                         "       tokspur-native recurrent-replay-timing --layer 0|2 <four GGUF shards...>\n"
                         "       tokspur-native attention-check --router-mode softmax|sigmoid --normalize-selected true|false --query-packing head|flat --rotary-pairs split|adjacent [--layer N --position N --token ID --repeats N] <GGUF shards...>\n"
                         "       tokspur-native memory-check --hash-order newest|oldest --boundary literal|before|after --dilation 1|3 [--token ID --repeats N --full-output-reference true|false] <GGUF shards...>\n"
                         "       tokspur-native target --policy PATH --allow-full-model true|--preflight-only true [--stack-execution direct|indirect-compatible|recurrent-replay] [--allow-stack-experiments true] [--known-token-capacity 0|8] [--allow-prefill-experiments true] [--reuse-residual-projections true] <GGUF shards...>\n"
                         "       tokspur-native response --tokenizer PATH --sha256 PIN\n";
            return 2;
        }
        const std::string_view command(argv[1]);
        if (command != "inspect" && command != "tensors" && command != "layout" && command != "row" && command != "verify" && command != "verify-persistent") throw tokspur::FormatError("unknown command");
        const auto args = options(argc, argv);
        tokspur::ModelFiles model(args.paths);
        if (command == "row" || command == "verify" || command == "verify-persistent") return operator_check(command, args, model);
        if (!args.tensor.empty() || args.row || args.rows) throw tokspur::FormatError("operator options require row or verify");
        if (command == "layout") {
            std::cout << tokspur::inspect_flash_next(model).describe().dump() << '\n';
            return 0;
        }
        std::map<uint32_t, std::pair<uint64_t, uint64_t>> counts;
        uint64_t file_bytes = 0;
        uint64_t header_bytes = 0;
        for (const auto& shard : model.shards()) {
            file_bytes = tokspur::checked_add(file_bytes, shard.file_size());
            header_bytes = tokspur::checked_add(header_bytes, shard.data_offset());
            for (const auto& tensor : shard.tensors()) {
                auto& count = counts[tensor.type];
                ++count.first;
                count.second = tokspur::checked_add(count.second, tensor.bytes);
            }
        }
        std::cout << "{\"schema\":1,\"architecture\":";
        quote(model.architecture());
        std::cout << ",\"shards\":" << model.shards().size() << ",\"tensors\":" << model.tensors().size()
                  << ",\"file_bytes\":" << file_bytes << ",\"header_bytes\":" << header_bytes << ",\"encodings\":[";
        bool comma = false;
        for (const auto& [id, count] : counts) {
            if (comma) std::cout << ',';
            comma = true;
            std::cout << "{\"type\":" << id << ",\"name\":";
            quote(tokspur::encoding(id).name);
            std::cout << ",\"tensors\":" << count.first << ",\"bytes\":" << count.second << '}';
        }
        std::cout << ']';
        if (std::string_view(argv[1]) == "tensors") {
            std::cout << ",\"inventory\":[";
            comma = false;
            for (const auto& [name, location] : model.tensors()) {
                const auto& tensor = model.shards()[location.shard].tensors()[location.index];
                if (comma) std::cout << ',';
                comma = true;
                std::cout << "{\"name\":";
                quote(name);
                std::cout << ",\"type\":" << tensor.type << ",\"shape\":[";
                for (size_t i = 0; i < tensor.dimensions.size(); ++i) {
                    if (i) std::cout << ',';
                    std::cout << tensor.dimensions[i];
                }
                std::cout << "],\"bytes\":" << tensor.bytes << ",\"shard\":" << location.shard << '}';
            }
            std::cout << ']';
        }
        model.assert_unchanged();
        std::cout << "}\n";
    } catch (const std::exception& error) {
        std::cerr << "tokspur-native: " << error.what() << '\n';
        return 1;
    }
}
