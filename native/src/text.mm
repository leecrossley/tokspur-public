#include "tokspur/text_assets.hpp"
#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>

#include "tokspur/text.hpp"
#include "tokspur/gguf.hpp"
#include "tokspur/unicode.hpp"
#include "tokspur/json.hpp"
#include "tokspur/chat.hpp"
#include "tokspur/transcript_replay.hpp"

#include <cmath>
#include <iostream>
#include <map>
#include <set>

namespace tokspur {
namespace {

constexpr size_t text_limit = 8 * 1024 * 1024;
constexpr std::string_view supported_pattern = R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

NSString* ns(std::string_view bytes) {
    // NSString's UTF-8 initializer consumes a leading BOM as an encoding marker.
    // Here it is model text and must remain U+FEFF, so supply literal UTF-16 units.
    const auto points = unicode_scalars(bytes);
    std::vector<unichar> characters;
    characters.reserve(points.size());
    for (auto point : points) {
        if (point < 0x10000) characters.push_back(static_cast<unichar>(point));
        else {
            point -= 0x10000;
            characters.push_back(static_cast<unichar>(0xd800 + (point >> 10)));
            characters.push_back(static_cast<unichar>(0xdc00 + (point & 1023)));
        }
    }
    NSString* result = [[NSString alloc] initWithCharacters:characters.data() length:characters.size()];
    if (!result) throw FormatError("expected valid UTF-8 text");
    return result;
}

std::string utf8(id object) {
    if (![object isKindOfClass:NSString.class]) throw FormatError("expected JSON string");
    NSString* value = static_cast<NSString*>(object);
    if (value.length > 16 * 1024 * 1024) throw FormatError("JSON string exceeds limit");
    std::vector<unichar> characters(value.length);
    [value getCharacters:characters.data() range:NSMakeRange(0, value.length)];
    std::string output;
    for (size_t i = 0; i < characters.size(); ++i) {
        uint32_t point = characters[i];
        if (point >= 0xd800 && point <= 0xdbff) {
            if (++i == characters.size() || characters[i] < 0xdc00 || characters[i] > 0xdfff) throw FormatError("JSON string contains unpaired surrogate");
            point = 0x10000 + ((point - 0xd800) << 10) + characters[i] - 0xdc00;
        } else if (point >= 0xdc00 && point <= 0xdfff) throw FormatError("JSON string contains unpaired surrogate");
        output += scalar_utf8(point);
    }
    return output;
}

NSDictionary* dictionary(id value) {
    if (![value isKindOfClass:NSDictionary.class]) throw FormatError("expected JSON object");
    return static_cast<NSDictionary*>(value);
}

NSArray* array(id value) {
    if (![value isKindOfClass:NSArray.class]) throw FormatError("expected JSON array");
    return static_cast<NSArray*>(value);
}

bool boolean(id value) {
    if (!value || CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID()) throw FormatError("expected JSON boolean");
    return [value boolValue];
}

uint32_t integer(id value) {
    if (![value isKindOfClass:NSNumber.class] || CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID()) {
        throw FormatError("expected unsigned JSON integer");
    }
    const double number = [value doubleValue];
    if (!std::isfinite(number) || number < 0 || number > 0xffffffff || std::floor(number) != number) throw FormatError("JSON integer is out of range");
    return static_cast<uint32_t>(number);
}

void require(bool condition, const char* message) {
    if (!condition) throw FormatError(message);
}

void null_or_missing(id value) {
    require(!value || value == NSNull.null, "unsupported non-null tokenizer setting");
}

std::string sha256(std::span<const std::byte> bytes) {
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data());
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    for (const auto byte : digest) { result.push_back(digits[byte >> 4]); result.push_back(digits[byte & 15]); }
    return result;
}

template <typename Strings>
std::string string_fingerprint(const Strings& values) {
    CC_SHA256_CTX context;
    CC_SHA256_Init(&context);
    const auto length = [&](uint64_t value) {
        std::array<unsigned char, 8> bytes{};
        for (size_t i = 0; i < 8; ++i) bytes[i] = static_cast<unsigned char>(value >> (8 * i));
        CC_SHA256_Update(&context, bytes.data(), bytes.size());
    };
    length(values.size());
    for (const auto& value : values) {
        length(value.size());
        CC_SHA256_Update(&context, value.data(), static_cast<CC_LONG>(value.size()));
    }
    std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
    CC_SHA256_Final(digest.data(), &context);
    constexpr char digits[] = "0123456789abcdef";
    std::string result;
    for (const auto byte : digest) { result.push_back(digits[byte >> 4]); result.push_back(digits[byte & 15]); }
    return result;
}

NSDictionary* json_object(std::string_view bytes) {
    require(bytes.size() <= 64 * 1024 * 1024, "JSON document exceeds limit");
    NSData* data = [NSData dataWithBytes:bytes.data() length:bytes.size()];
    NSError* error = nil;
    id value = [NSJSONSerialization JSONObjectWithData:data options:0 error:&error];
    if (!value || error) throw FormatError("invalid JSON object");
    return dictionary(value);
}

class AppleProcessor final : public TextProcessor {
public:
    explicit AppleProcessor(std::string_view pattern) {
        require(pattern == supported_pattern, "unsupported tokenizer split pattern");
        NSError* error = nil;
        regex_ = [NSRegularExpression regularExpressionWithPattern:ns(pattern) options:0 error:&error];
        if (!regex_ || error) throw FormatError("cannot compile declared split pattern");
    }
    std::vector<std::string> pieces(std::string_view segment) const override {
        @autoreleasepool {
            require(segment.size() <= text_limit, "normalizer input exceeds limit");
            NSString* normalized = ns(normalize_nfc9(segment));
            std::vector<std::string> output;
            auto* target = &output;
            __block NSUInteger last = 0;
            __block bool exceeded = false;
            [regex_ enumerateMatchesInString:normalized options:0 range:NSMakeRange(0, normalized.length)
                usingBlock:^(NSTextCheckingResult* match, NSMatchingFlags, BOOL* stop) {
                    @autoreleasepool {
                        if (!match) return;
                        if (target->size() >= 1'000'000) { exceeded = true; *stop = YES; return; }
                        if (match.range.location > last) target->push_back(utf8([normalized substringWithRange:NSMakeRange(last, match.range.location - last)]));
                        if (match.range.length) target->push_back(utf8([normalized substringWithRange:match.range]));
                        last = NSMaxRange(match.range);
                    }
                }];
            require(!exceeded, "pretokenizer piece limit exceeded");
            if (last < normalized.length) output.push_back(utf8([normalized substringFromIndex:last]));
            return output;
        }
    }
private:
    NSRegularExpression* regex_;
};

BpeData parse_model(NSDictionary* root, std::string& pattern, std::string& vocabulary_fingerprint, std::string& merge_fingerprint) {
    require(utf8(root[@"version"]) == "1.0", "unsupported tokenizer JSON version");
    null_or_missing(root[@"truncation"]);
    null_or_missing(root[@"padding"]);
    require(utf8(dictionary(root[@"normalizer"])[@"type"]) == "NFC", "unsupported tokenizer normalizer");
    NSDictionary* sequence = dictionary(root[@"pre_tokenizer"]);
    require(utf8(sequence[@"type"]) == "Sequence", "expected declared tokenizer sequence");
    NSArray* steps = array(sequence[@"pretokenizers"]);
    require(steps.count == 2, "unsupported tokenizer sequence length");
    NSDictionary* split = dictionary(steps[0]);
    require(utf8(split[@"type"]) == "Split" && utf8(split[@"behavior"]) == "Isolated" && !boolean(split[@"invert"]), "unsupported split settings");
    pattern = utf8(dictionary(split[@"pattern"])[@"Regex"]);
    const auto byte_level = [](NSDictionary* settings, bool prefix, bool trim, bool regex) {
        require(utf8(settings[@"type"]) == "ByteLevel" && boolean(settings[@"add_prefix_space"]) == prefix &&
                boolean(settings[@"trim_offsets"]) == trim && boolean(settings[@"use_regex"]) == regex, "unsupported byte-level settings");
    };
    byte_level(dictionary(steps[1]), false, false, false);
    byte_level(dictionary(root[@"post_processor"]), false, false, false);
    byte_level(dictionary(root[@"decoder"]), true, true, true);
    NSDictionary* model = dictionary(root[@"model"]);
    require(utf8(model[@"type"]) == "BPE", "unsupported tokenizer model");
    for (NSString* name in @[@"dropout", @"unk_token", @"continuing_subword_prefix", @"end_of_word_suffix"]) null_or_missing(model[name]);
    for (NSString* name in @[@"fuse_unk", @"byte_fallback", @"ignore_merges"]) require(!boolean(model[name]), "unsupported BPE option");
    BpeData data;
    std::map<uint32_t, std::string> added;
    NSArray* markers = array(root[@"added_tokens"]);
    require(markers.count <= 4096, "added token count exceeds limit");
    for (id item in markers) {
        NSDictionary* token = dictionary(item);
        for (NSString* flag in @[@"single_word", @"lstrip", @"rstrip", @"normalized"]) require(!boolean(token[flag]), "unsupported added-token matching flags");
        const auto id = integer(token[@"id"]);
        const auto content = utf8(token[@"content"]);
        require(added.emplace(id, content).second, "duplicate added token ID");
        data.added.push_back({id, content, boolean(token[@"special"])});
    }
    NSDictionary* vocabulary = dictionary(model[@"vocab"]);
    require(vocabulary.count >= 256 && vocabulary.count <= 1'000'000, "tokenizer vocabulary count exceeds limit");
    data.vocabulary.resize(vocabulary.count);
    std::vector<std::string> spellings(vocabulary.count);
    std::vector<bool> seen(vocabulary.count, false);
    for (id spelling in vocabulary) {
        const auto id = integer(vocabulary[spelling]);
        require(id < vocabulary.count && !seen[id], "noncontiguous or duplicate vocabulary ID");
        seen[id] = true;
        const auto encoded = utf8(spelling);
        spellings[id] = encoded;
        if (const auto marker = added.find(id); marker != added.end()) {
            require(marker->second == encoded, "added token differs from vocabulary");
            data.vocabulary[id] = encoded;
        } else data.vocabulary[id] = byte_level_decode(encoded);
    }
    vocabulary_fingerprint = string_fingerprint(spellings);
    spellings.clear();
    NSArray* merges = array(model[@"merges"]);
    require(merges.count <= 1'000'000, "merge count exceeds limit");
    data.merges.reserve(merges.count);
    for (id item in merges) {
        NSArray* parts = array(item);
        require(parts.count == 2, "unsupported merge representation");
        const auto left = utf8(parts[0]);
        const auto right = utf8(parts[1]);
        data.merges.emplace_back(byte_level_decode(left), byte_level_decode(right));
        spellings.push_back(left + " " + right);
    }
    merge_fingerprint = string_fingerprint(spellings);
    return data;
}

bool line(std::string& output, bool& oversized) {
    // Drain an oversized line instead of throwing here: the caller reports it as
    // an error response and keeps serving, as the response adapter already does.
    output.clear();
    oversized = false;
    char c;
    bool any = false;
    while (std::cin.get(c)) {
        any = true;
        if (c == '\n') return true;
        if (output.size() >= text_limit) oversized = true;
        else output.push_back(c);
    }
    return any;
}

} // namespace

struct NativeText::Impl {
    std::unique_ptr<ByteBpe> bpe;
    std::unique_ptr<AppleProcessor> processor;
    std::string vocabulary_fingerprint;
    std::string merge_fingerprint;
};

NativeText::NativeText(const std::filesystem::path& tokenizer_json, std::string_view expected_sha256)
    : impl_(std::make_unique<Impl>()) {
    @autoreleasepool {
        require(expected_sha256.size() == 64 && expected_sha256.find_first_not_of("0123456789abcdef") == std::string_view::npos,
                "tokenizer asset requires a lowercase SHA256 pin");
        MappedFile file(tokenizer_json);
        require(file.size() <= 64 * 1024 * 1024, "tokenizer JSON exceeds limit");
        require(sha256(file.bytes()) == expected_sha256, "tokenizer JSON hash differs from supplied pin");
        const auto bytes = file.bytes();
        std::string pattern;
        BpeData data = parse_model(json_object({reinterpret_cast<const char*>(bytes.data()), bytes.size()}), pattern,
                                   impl_->vocabulary_fingerprint, impl_->merge_fingerprint);
        impl_->processor = std::make_unique<AppleProcessor>(pattern);
        impl_->bpe = std::make_unique<ByteBpe>(std::move(data));
        file.assert_unchanged();
    }
}

NativeText::~NativeText() = default;
std::vector<uint32_t> NativeText::encode(std::string_view text, std::span<const std::pair<size_t, size_t>> literal_ranges) const {
    return impl_->bpe->encode(text, *impl_->processor, literal_ranges);
}
std::string NativeText::decode_bytes(std::span<const uint32_t> ids, bool skip_special) const { return impl_->bpe->decode_bytes(ids, skip_special); }
std::string NativeText::decode(std::span<const uint32_t> ids, bool skip_special) const { return impl_->bpe->decode_text(ids, skip_special); }
size_t NativeText::vocabulary_size() const { return impl_->bpe->vocabulary_size(); }
size_t NativeText::merge_count() const { return impl_->bpe->merge_count(); }

void NativeText::verify_gguf(const ModelFiles& model) const {
    size_t found = 0;
    for (const auto& shard : model.shards()) {
        if (!shard.find("tokenizer.ggml.tokens")) continue;
        ++found;
        const auto tokens = shard.string_array("tokenizer.ggml.tokens");
        const auto merges = shard.string_array("tokenizer.ggml.merges");
        require(tokens.size() == vocabulary_size() && merges.size() == merge_count(), "JSON/GGUF tokenizer counts differ");
        require(string_fingerprint(tokens) == impl_->vocabulary_fingerprint, "JSON vocabulary differs from GGUF");
        require(string_fingerprint(merges) == impl_->merge_fingerprint, "JSON merge ranks differ from GGUF");
    }
    require(found > 0, "GGUF shard set lacks tokenizer metadata");
    model.assert_unchanged();
}

int text_cli(int argc, char** argv) {
    const bool replay_mode = std::string_view(argv[1]) == "http-replay";
    const bool chat_mode = std::string_view(argv[1]) == "chat" || replay_mode;
    std::filesystem::path path;
    std::string pin;
    std::filesystem::path template_path;
    std::string template_pin;
    std::vector<std::filesystem::path> shards;
    std::string replay_alias;
    ToolEnumValidation replay_enums = ToolEnumValidation::strict; bool enum_option = false;
    for (int i = 2; i < argc; i += 2) {
        if (i + 1 >= argc) throw FormatError("missing text option value");
        const std::string_view flag(argv[i]);
        if (flag == "--tokenizer" && path.empty()) path = argv[i + 1];
        else if (flag == "--sha256" && pin.empty()) pin = argv[i + 1];
        else if (flag == "--template" && template_path.empty()) template_path = argv[i + 1];
        else if (flag == "--template-sha256" && template_pin.empty()) template_pin = argv[i + 1];
        else if (flag == "--gguf") shards.emplace_back(argv[i + 1]);
        else if (replay_mode && flag == "--model-alias" && replay_alias.empty()) replay_alias = argv[i + 1];
        else if (replay_mode && flag == "--tool-enum-validation" && !enum_option) {
            replay_enums = parse_tool_enum_validation(argv[i + 1]); enum_option = true;
        }
        else throw FormatError("unknown or duplicate text option");
    }
    require(!path.empty() && !pin.empty(), "text requires --tokenizer PATH --sha256 PIN");
    require(!replay_mode || (shards.empty() && !replay_alias.empty() && replay_alias.size() <= 64 &&
        pin == text_assets::tokenizer_sha256), "offline replay requires explicit alias/pinned tokenizer and refuses GGUF assets");
    std::unique_ptr<ChatRenderer> renderer;
    if (chat_mode) {
        require(!template_path.empty() && template_pin == text_assets::template_sha256, "chat requires the supported pinned template");
        MappedFile file(template_path);
        require(file.size() <= 64 * 1024, "chat template exceeds limit");
        require(sha256(file.bytes()) == template_pin, "chat template differs from its pin");
        const auto bytes = file.bytes();
        renderer = std::make_unique<ChatRenderer>(extract_flash_next_literals({reinterpret_cast<const char*>(bytes.data()), bytes.size()}));
        file.assert_unchanged();
    } else require(template_path.empty() && template_pin.empty(), "template options require chat mode");
    NativeText tokenizer(path, pin);
    if (!shards.empty()) tokenizer.verify_gguf(ModelFiles(shards));
    std::unique_ptr<TranscriptReplay> replay;
    if (replay_mode) {
        const auto marker = [&](std::string_view bytes) {
            const auto ids = tokenizer.encode(bytes);
            require(ids.size() == 1 && tokenizer.decode_bytes(ids) == bytes, "replay marker identity differs");
            return ids[0];
        };
        replay = std::make_unique<TranscriptReplay>(ServeCodec{
            [&](const Json& value) { return renderer->render(value); },
            [&](const RenderedChat& rendered) { return tokenizer.encode(rendered.prompt, rendered.literal_ranges); },
            [&](std::span<const uint32_t> ids) { return tokenizer.decode_bytes(ids); },
            {uint32_t(tokenizer.vocabulary_size()), marker("<|im_end|>"), marker("<|endoftext|>"), marker("<|im_start|>")}}, replay_alias, replay_enums);
    }
    std::string request;
    bool failed = false;
    bool oversized = false;
    while (line(request, oversized)) {
        @autoreleasepool {
            try {
                require(!oversized, "text request line exceeds limit");
                const auto object = parse_json(request);
                if (replay) {
                    const auto result = replay->turn(object);
                    if (!result.find("passed")->boolean()) failed = true;
                    std::cout << result.dump(32768) << '\n' << std::flush;
                    continue;
                }
                if (renderer) {
                    auto rendered = renderer->render(object);
                    const auto ids = tokenizer.encode(rendered.prompt, rendered.literal_ranges);
                    Json::Array encoded;
                    encoded.reserve(ids.size());
                    for (const auto id : ids) encoded.push_back(Json::integer(id));
                    std::cout << Json(Json::Object{{"prompt", Json(std::move(rendered.prompt))}, {"ids", Json(std::move(encoded))},
                        {"normalized_messages", std::move(rendered.normalized_messages)},
                        {"parsed_tool_arguments", Json::integer(rendered.parsed_tool_arguments)},
                        {"hoisted_system_messages", Json::integer(rendered.hoisted_system_messages)}}).dump() << '\n' << std::flush;
                    continue;
                }
                for (const auto& [key, value] : object.object()) {
                    static_cast<void>(value);
                    require(key == "text" || key == "ids" || key == "skip_special_tokens", "unknown text request field");
                }
                const auto* text = object.find("text");
                const auto* input_ids = object.find("ids");
                require(bool(text) != bool(input_ids), "provide exactly one of text or ids");
                const auto* skip_option = object.find("skip_special_tokens");
                const bool skip = skip_option ? skip_option->boolean() : false;
                std::vector<uint32_t> ids;
                if (text) ids = tokenizer.encode(text->string());
                else {
                    const auto& input = input_ids->array();
                    require(input.size() <= 1'000'000, "decode token count exceeds limit");
                    ids.reserve(input.size());
                    for (const auto& value : input) {
                        const auto id = value.unsigned_integer();
                        require(id <= 0xffffffff, "token ID exceeds range");
                        ids.push_back(static_cast<uint32_t>(id));
                    }
                }
                const auto decoded = tokenizer.decode(ids, skip);
                Json::Array encoded;
                encoded.reserve(ids.size());
                for (const auto id : ids) encoded.push_back(Json::integer(id));
                std::cout << Json(Json::Object{{"ids", Json(std::move(encoded))}, {"text", Json(decoded)},
                    {"vocabulary_size", Json::integer(tokenizer.vocabulary_size())}, {"merge_count", Json::integer(tokenizer.merge_count())}}).dump() << '\n' << std::flush;
            } catch (const FormatError& error) {
                failed = true;
                std::cout << Json(Json::Object{{"error", Json(error.what())}}).dump() << '\n' << std::flush;
            }
        }
    }
    return failed ? 1 : 0;
}

} // namespace tokspur
