#include "tokspur/mtp.hpp"

#include <iostream>
#include <optional>

int main(int argc, char** argv) {
    try {
        std::vector<std::filesystem::path> target;
        std::filesystem::path sidecar;
        std::optional<tokspur::MtpTemplatePolicy> policy;
        if (argc > 15) throw tokspur::FormatError("too many MTP metadata arguments");
        for (int i = 1; i < argc; ++i) {
            const std::string_view key(argv[i]);
            if (i + 1 == argc) throw tokspur::FormatError("missing MTP metadata argument value");
            const std::string_view value(argv[++i]);
            if (key == "--target") target.emplace_back(value);
            else if (key == "--sidecar" && sidecar.empty()) sidecar = value;
            else if (key == "--template-policy" && !policy) {
                if (value == "require-exact-identity") policy = tokspur::MtpTemplatePolicy::require_exact_identity;
                else if (value == "target-rendered-ids-only") policy = tokspur::MtpTemplatePolicy::target_rendered_ids_only;
                else throw tokspur::FormatError("unsupported MTP template policy");
            } else throw tokspur::FormatError("unknown or duplicate MTP metadata argument");
        }
        if (target.size() != 4 || sidecar.empty() || !policy) throw tokspur::FormatError(
            "expected four --target paths, one --sidecar and explicit --template-policy");
        tokspur::MtpSharedBinding binding(std::make_shared<const tokspur::ModelFiles>(target), sidecar, *policy);
        // Exercise only registered descriptor lookup; neither call reads weights.
        static_cast<void>(binding.borrowed_tensor("token_embd.weight"));
        static_cast<void>(binding.borrowed_tensor("output.weight"));
        binding.assert_unchanged();
        std::cout << binding.layout().describe().dump() << '\n';
    } catch (const std::exception& error) {
        std::cerr << "MTP metadata admission refused: " << error.what() << '\n';
        return 1;
    }
}
