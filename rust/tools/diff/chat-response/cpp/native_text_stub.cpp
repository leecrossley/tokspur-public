// Fixture NativeText for linking the unchanged native/src/response_cli.cpp and native/tests/response_cli_main.cpp into a model-free differential binary. The real NativeText (native/src/text.mm) maps a pinned tokenizer JSON; this stub never reads files. The path's file name selects the fixture:
//   fixture.json  test_response_protocol.cpp's ByteBpe (markers are added tokens)
//   plain.json    the same vocabulary with the markers as ordinary tokens
//   anything else throws FormatError("stub tokenizer missing")
// text_cli is a stub: it returns 7, or throws when argv[2] is "throw".

#include "tokspur/gguf.hpp"
#include "tokspur/text.hpp"

#include <memory>

namespace tokspur {
namespace {
class Identity final : public TextProcessor {
    std::vector<std::string> pieces(std::string_view text) const override { return {std::string(text)}; }
};
const Identity identity;
} // namespace

struct NativeText::Impl {
    std::unique_ptr<ByteBpe> bpe;
};

NativeText::NativeText(const std::filesystem::path& tokenizer_json, std::string_view expected_sha256)
    : impl_(std::make_unique<Impl>()) {
    static_cast<void>(expected_sha256);
    const auto name = tokenizer_json.filename().string();
    const bool added = name == "fixture.json";
    if (!added && name != "plain.json") throw FormatError("stub tokenizer missing");
    BpeData data;
    for (unsigned byte = 0; byte < 256; ++byte) data.vocabulary.emplace_back(1, char(byte));
    for (const auto marker : {"<|im_end|>", "<|endoftext|>", "<|im_start|>"}) {
        const auto id = uint32_t(data.vocabulary.size());
        data.vocabulary.emplace_back(marker);
        if (added) data.added.push_back({id, marker, true});
    }
    impl_->bpe = std::make_unique<ByteBpe>(std::move(data));
}

NativeText::~NativeText() = default;
std::vector<uint32_t> NativeText::encode(std::string_view text) const { return impl_->bpe->encode(text, identity); }
std::string NativeText::decode_bytes(std::span<const uint32_t> ids, bool skip_special) const {
    return impl_->bpe->decode_bytes(ids, skip_special);
}
size_t NativeText::vocabulary_size() const { return impl_->bpe->vocabulary_size(); }

int text_cli(int argc, char** argv) {
    if (argc >= 3 && std::string_view(argv[2]) == "throw") throw FormatError("text stub refused");
    return 7;
}

} // namespace tokspur
