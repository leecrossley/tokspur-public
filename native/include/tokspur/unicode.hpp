#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace tokspur {

std::vector<uint32_t> unicode_scalars(std::string_view utf8);
std::string scalar_utf8(uint32_t scalar);
std::string normalize_nfc9(std::string_view utf8);

} // namespace tokspur
