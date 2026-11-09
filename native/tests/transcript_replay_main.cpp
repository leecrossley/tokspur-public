#include "tokspur/text.hpp"

#include <exception>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    try {
        if (argc < 2 || std::string_view(argv[1]) != "http-replay") {
            std::cerr << "expected http-replay with pinned tokenizer/template and explicit model alias\n";
            return 2;
        }
        return tokspur::text_cli(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 2;
    }
}
