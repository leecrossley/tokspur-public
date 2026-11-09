#include "tokspur/text.hpp"

#include <iostream>

// Isolated Foundation-only integration executable; never initializes Metal.
int main(int argc, char** argv) {
    try {
        if (argc >= 2 && (std::string_view(argv[1]) == "text" || std::string_view(argv[1]) == "chat")) return tokspur::text_cli(argc, argv);
        return tokspur::response_cli(argc, argv);
    }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
