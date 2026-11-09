#include "tokspur/serve.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    try { return tokspur::server_cli(argc, argv); }
    catch (const std::exception& error) {
        std::cerr << "tokspur-server refused startup/shutdown: " << error.what() << '\n';
        return 1;
    }
}
