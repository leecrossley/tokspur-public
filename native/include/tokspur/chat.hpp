#pragma once

#include "tokspur/json.hpp"

#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tokspur {

// These are literal DATA read from the pinned model template, not executable
// template code. Rendering rules below are independently authored C++.
struct ChatStrings {
    std::string reasoning_xhigh;
    std::string reasoning_low;
    std::string tools_header;
    std::string tool_instructions;
};

ChatStrings extract_flash_next_literals(std::string_view pinned_template);
std::string template_json(const Json& value);
std::string chat_trim(std::string_view value);

struct RenderedChat {
    std::string prompt;
    Json normalized_messages;
    size_t parsed_tool_arguments = 0;
    size_t hoisted_system_messages = 0;
    // Renderer-owned byte ranges where client spellings must not select added tokens.
    std::vector<std::pair<size_t, size_t>> literal_ranges = {};
};

class ChatRenderer {
public:
    explicit ChatRenderer(ChatStrings strings);
    [[nodiscard]] RenderedChat render(const Json& request) const;
private:
    ChatStrings strings_;
};

} // namespace tokspur
