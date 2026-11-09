#pragma once
// Diagnostic row serializer only. The existing vocabulary-only target helper
// and its248320-value refusal remain untouched.
#include "tokspur/gguf.hpp"
#include "tokspur/json.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <array>
#include <bit>
#include <span>
#include <string>
#include <utility>

namespace tokspur::sampled_cost {
inline Json row_seal(std::span<const float> logits, std::span<const float> hc, std::string state) {
    static_assert(std::endian::native == std::endian::little && sizeof(float) == 4);
    if (logits.size() != 248320) throw FormatError("sampled COST vocabulary hash requires248320 values");
    if (hc.size() != 10240) throw FormatError("sampled COST HC hash requires10240 values");
    if (state.size() != 64 || state.find_first_not_of("0123456789abcdef") != std::string::npos)
        throw FormatError("sampled COST accepted-state hash differs");
    const auto hash = [](std::span<const float> values) {
        std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
        if (!CC_SHA256(values.data(), static_cast<CC_LONG>(values.size_bytes()), digest.data()))
            throw FormatError("cannot hash sampled COST row");
        constexpr char hex[] = "0123456789abcdef"; std::string out; out.reserve(64);
        for (auto byte : digest) { out += hex[byte >> 4]; out += hex[byte & 15]; } return out;
    };
    return Json(Json::Object{{"logit",Json(hash(logits))},{"hc",Json(hash(hc))},{"state",Json(std::move(state))}});
}
}
