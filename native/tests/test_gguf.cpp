#include "tokspur/gguf.hpp"
#include "tokspur/worker_identity.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <sys/stat.h>
#include <unistd.h>

namespace {
using Bytes = std::vector<std::byte>;
size_t checks = 0;

void require(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}

void rejected(const std::function<void()>& action) {
    ++checks;
    try { action(); } catch (const tokspur::FormatError&) { return; }
    throw std::runtime_error("malformed input was accepted");
}

void number(Bytes& bytes, uint64_t value, size_t width) {
    for (size_t i = 0; i < width; ++i) bytes.push_back(std::byte((value >> (i * 8)) & 255));
}

void replace(Bytes& bytes, size_t offset, uint64_t value, size_t width) {
    for (size_t i = 0; i < width; ++i) bytes.at(offset + i) = std::byte((value >> (i * 8)) & 255);
}

void string(Bytes& bytes, std::string_view value) {
    number(bytes, value.size(), 8);
    for (const unsigned char c : value) bytes.push_back(std::byte(c));
}

Bytes integer(uint64_t value, size_t width) {
    Bytes bytes;
    number(bytes, value, width);
    return bytes;
}

Bytes text(std::string_view value) {
    Bytes bytes;
    string(bytes, value);
    return bytes;
}

struct Key {
    std::string name;
    uint32_t type;
    Bytes value;
};

struct Data {
    std::string name = "weight";
    std::vector<uint64_t> dimensions{4};
    uint32_t type = 0;
    uint64_t offset = 0;
    size_t bytes = 16;
};

Bytes fixture(std::vector<Key> keys = {}, std::vector<Data> tensors = {Data{}}, size_t alignment = 32) {
    Bytes bytes;
    number(bytes, 0x46554747, 4);
    number(bytes, 3, 4);
    number(bytes, tensors.size(), 8);
    number(bytes, keys.size(), 8);
    for (const auto& key : keys) {
        string(bytes, key.name);
        number(bytes, key.type, 4);
        bytes.insert(bytes.end(), key.value.begin(), key.value.end());
    }
    size_t payload = 0;
    for (const auto& tensor : tensors) {
        string(bytes, tensor.name);
        number(bytes, tensor.dimensions.size(), 4);
        for (auto dimension : tensor.dimensions) number(bytes, dimension, 8);
        number(bytes, tensor.type, 4);
        number(bytes, tensor.offset, 8);
        payload = std::max(payload, static_cast<size_t>(tensor.offset) + tensor.bytes);
    }
    bytes.resize((bytes.size() + alignment - 1) / alignment * alignment);
    bytes.resize(bytes.size() + payload);
    return bytes;
}

std::vector<Key> split(unsigned part, unsigned count = 2, unsigned total = 2) {
    return {{"general.architecture", 8, text("test")}, {"split.no", 2, integer(part, 2)},
            {"split.count", 2, integer(count, 2)}, {"split.tensors.count", 5, integer(total, 4)}};
}

class Temporary {
public:
    Temporary() {
        const std::string prefix = (std::filesystem::temp_directory_path() / "tokspur-test-XXXXXX").string();
        std::vector<char> pattern(prefix.begin(), prefix.end());
        pattern.push_back('\0');
        const auto* made = mkdtemp(pattern.data());
        if (!made) throw std::runtime_error("cannot create temporary directory");
        root = made;
    }
    std::filesystem::path write(const char* name, const Bytes& bytes) {
        const auto path = root / name;
        std::ofstream stream(path, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!stream) throw std::runtime_error("cannot write fixture");
        files.push_back(path);
        return path;
    }
    ~Temporary() {
        for (const auto& path : files) std::filesystem::remove(path);
        std::filesystem::remove(root);
    }
    std::filesystem::path root;
    std::vector<std::filesystem::path> files;
};

void parsing() {
    const auto bytes = fixture();
    tokspur::Gguf valid{std::span(bytes)};
    require(valid.tensors().size() == 1, "missing tensor");
    require(valid.tensors()[0].elements == 4, "element count");
    require(valid.tensor_bytes(valid.tensors()[0]).size() == 16, "tensor span");
    const auto copied_tensor = valid.tensors()[0];
    rejected([&] { static_cast<void>(valid.tensor_bytes(copied_tensor)); });
    const auto header = std::span(bytes).first(static_cast<size_t>(valid.data_offset()));
    tokspur::Gguf catalogue(header, bytes.size());
    require(catalogue.header_only() && catalogue.file_size() == bytes.size(), "header-only extent");
    require(catalogue.tensors()[0].bytes == 16, "header-only descriptor");
    rejected([&] { static_cast<void>(catalogue.tensor_bytes(catalogue.tensors()[0])); });
    // Payload access is forbidden even when the supplied span contains it.
    tokspur::Gguf full_span_catalogue(std::span(bytes), bytes.size());
    rejected([&] { static_cast<void>(full_span_catalogue.tensor_bytes(full_span_catalogue.tensors()[0])); });
    rejected([&] { tokspur::Gguf short_extent(header, bytes.size() - 1); });
    rejected([&] { tokspur::Gguf impossible_extent(header, header.size() - 1); });
    rejected([&] { tokspur::Gguf truncated_header(header.first(header.size() - 1), bytes.size()); });
    for (size_t length = 0; length < bytes.size(); ++length) {
        rejected([&] { tokspur::Gguf truncated{std::span(bytes.data(), length)}; });
    }
    auto corrupt = bytes;
    corrupt[0] = std::byte{0};
    rejected([&] { tokspur::Gguf file{std::span(corrupt)}; });
    corrupt = bytes;
    replace(corrupt, 4, 2, 4);
    rejected([&] { tokspur::Gguf file{std::span(corrupt)}; });
    replace(corrupt, 4, 0x03000000, 4);
    rejected([&] { tokspur::Gguf file{std::span(corrupt)}; });
    corrupt = bytes;
    replace(corrupt, 8, 100001, 8);
    rejected([&] { tokspur::Gguf file{std::span(corrupt)}; });
    replace(corrupt, 8, 1, 8);
    replace(corrupt, 16, std::numeric_limits<uint64_t>::max(), 8);
    rejected([&] { tokspur::Gguf file{std::span(corrupt)}; });
}

void metadata_only_shards() {
    Temporary temporary;
    for (const auto alignment : {32u, 64u}) {
        auto keys = split(0, 2, 1);
        keys.push_back({"general.alignment", 4, integer(alignment, 4)});
        const auto padded = fixture(keys, {}, alignment);
        size_t metadata_end = 24;
        for (const auto& key : keys) metadata_end += 8 + key.name.size() + 4 + key.value.size();
        require(metadata_end < padded.size(), "fixture needs optional padding");
        for (size_t length = metadata_end; length <= padded.size(); ++length) {
            const auto bytes = std::span(padded).first(length);
            tokspur::Gguf borrowed(bytes);
            tokspur::Gguf catalogue(bytes, length);
            require(borrowed.tensors().empty() && catalogue.tensors().empty(), "metadata-only tensor count");
            require(borrowed.data_offset() == length && catalogue.data_offset() == length, "metadata-only extent");
            require(borrowed.unsigned_value("split.tensors.count") == 1, "metadata-only split total");
        }
        for (size_t length = 0; length < metadata_end; ++length) {
            rejected([&] { tokspur::Gguf truncated(std::span(padded).first(length)); });
            rejected([&] { tokspur::Gguf truncated(std::span(padded).first(length), padded.size()); });
        }
        const auto name = std::to_string(alignment);
        const auto unpadded = temporary.write(("metadata-" + name).c_str(), Bytes(padded.begin(), padded.begin() + metadata_end));
        const auto payload = temporary.write(("tensor-" + name).c_str(), fixture(split(1, 2, 1)));
        tokspur::Gguf file(unpadded);
        require(file.data_offset() == metadata_end && file.tensors().empty(), "metadata-only file");
        tokspur::ModelFiles model({payload, unpadded});
        require(model.tensors().size() == 1 && model.tensor_bytes("weight").size() == 16, "metadata-only shard set");
        const auto tensor_bytes = fixture(keys, {Data{}}, alignment);
        tokspur::Gguf tensor_file{std::span(tensor_bytes)};
        const auto short_header = std::span(tensor_bytes).first(tensor_file.data_offset() - 1);
        rejected([&] { tokspur::Gguf truncated(short_header); });
        rejected([&] { tokspur::Gguf truncated(short_header, tensor_bytes.size()); });
        const auto short_file = temporary.write(("short-" + name).c_str(), Bytes(short_header.begin(), short_header.end()));
        rejected([&] { tokspur::Gguf truncated(short_file); });
    }
    for (const auto alignment : {0u, 3u, 65537u}) {
        const auto bytes = fixture({{"general.alignment", 4, integer(alignment, 4)}}, {});
        rejected([&] { tokspur::Gguf invalid{std::span(bytes)}; });
    }
    auto bare = fixture({}, {});
    bare.resize(24);
    tokspur::Gguf empty{std::span(bare)};
    require(empty.tensors().empty() && empty.data_offset() == 24, "empty GGUF header");
}

void metadata() {
    const std::vector<Key> keys{{"general.architecture", 8, text("toy")}, {"test.count", 2, integer(42, 2)},
                                {"test.negative", 5, integer(0xffffffff, 4)}, {"test.bool", 7, integer(1, 1)}};
    auto bytes = fixture(keys);
    tokspur::Gguf model{std::span(bytes)};
    require(model.string_value("general.architecture") == "toy", "string metadata");
    require(model.unsigned_value("test.count") == 42, "unsigned metadata");
    require(model.boolean_value("test.bool"), "boolean metadata");
    require(model.metadata_bytes("test.count").size() == 2, "exact typed metadata span");
    rejected([&] { static_cast<void>(model.boolean_value("test.count")); });
    rejected([&] { static_cast<void>(model.boolean_value("missing")); });
    rejected([&] { static_cast<void>(model.metadata_bytes("missing")); });
    rejected([&] { static_cast<void>(model.unsigned_value("test.negative")); });
    rejected([&] { static_cast<void>(model.unsigned_value("test.bool")); });
    rejected([&] { static_cast<void>(model.unsigned_value("test.missing")); });
    rejected([&] { static_cast<void>(model.string_value("test.count")); });
    require(model.find("not.found") == nullptr, "missing metadata lookup");
    for (const auto& key : {"", "Bad", ".bad", "bad.", "bad..key", "bad-key"}) {
        const auto data = fixture({{key, 0, integer(0, 1)}});
        rejected([&] { tokspur::Gguf file{std::span(data)}; });
    }
    bytes = fixture({{"a", 0, integer(0, 1)}, {"a", 0, integer(1, 1)}});
    rejected([&] { tokspur::Gguf file{std::span(bytes)}; });
    bytes = fixture({{"a", 7, integer(2, 1)}});
    rejected([&] { tokspur::Gguf file{std::span(bytes)}; });
    bytes = fixture({{"a", 13, {}}});
    rejected([&] { tokspur::Gguf file{std::span(bytes)}; });
    for (const auto& invalid : {std::string("\xc0\x80", 2), std::string("\xed\xa0\x80", 3),
                               std::string("\xf4\x90\x80\x80", 4), std::string("\xe0\x80\x80", 3),
                               std::string("\xc2", 1), std::string("\xf5\x80\x80\x80", 4)}) {
        bytes = fixture({{"a", 8, text(invalid)}});
        rejected([&] { tokspur::Gguf file{std::span(bytes)}; });
    }
    bytes = fixture({{"a", 8, text("α🦉")}});
    tokspur::Gguf utf8{std::span(bytes)};
    require(utf8.string_value("a") == "α🦉", "valid Unicode rejected");
    Bytes array;
    number(array, 8, 4);
    number(array, 2, 8);
    string(array, "first");
    string(array, "second");
    bytes = fixture({{"a", 9, array}});
    tokspur::Gguf strings{std::span(bytes)};
    require(strings.find("a")->elements == 2 && strings.find("a")->element_type == 8, "array descriptor");
    require(strings.string_array("a") == std::vector<std::string_view>({"first", "second"}), "borrowed string array");
    rejected([&] { static_cast<void>(model.string_array("test.count")); });
    rejected([&] { static_cast<void>(model.string_array("missing")); });
    replace(array, 4, std::numeric_limits<uint64_t>::max(), 8);
    bytes = fixture({{"a", 9, array}});
    rejected([&] { tokspur::Gguf file{std::span(bytes)}; });
    array.clear();
    number(array, 7, 4);
    number(array, 1, 8);
    number(array, 2, 1);
    bytes = fixture({{"a", 9, array}});
    rejected([&] { tokspur::Gguf file{std::span(bytes)}; });
    array.clear();
    for (size_t i = 0; i < 10; ++i) { number(array, 9, 4); number(array, 1, 8); }
    number(array, 0, 4);
    number(array, 0, 8);
    bytes = fixture({{"a", 9, array}});
    rejected([&] { tokspur::Gguf file{std::span(bytes)}; });
    for (auto alignment : {0, 3, 65537}) {
        bytes = fixture({{"general.alignment", 4, integer(alignment, 4)}});
        rejected([&] { tokspur::Gguf file{std::span(bytes)}; });
    }
    bytes = fixture({{"general.alignment", 4, integer(64, 4)}}, {Data{}}, 64);
    tokspur::Gguf aligned{std::span(bytes)};
    require(aligned.data_offset() % 64 == 0, "explicit alignment");
}

void tensors() {
    auto huge_offset = fixture();
    replace(huge_offset, 54, uint64_t(-32), 8);
    rejected([&] { tokspur::Gguf file{std::span(huge_offset)}; });
    for (const Data& bad : {Data{"", {4}, 0, 0, 16}, Data{"a", {}, 0, 0, 16},
                            Data{"a", {1, 1, 1, 1, 1}, 0, 0, 16}, Data{"a", {0}, 0, 0, 16},
                            Data{"a", {3}, 12, 0, 16}, Data{"a", {1}, 999, 0, 16},
                            Data{"a", {4}, 0, 1, 16}, Data{"a", {5}, 0, 0, 16},
                            Data{"a", {uint64_t(-1), 2}, 0, 0, 0}, Data{"a b", {4}, 0, 0, 16}}) {
        const auto bytes = fixture({}, {bad});
        rejected([&] { tokspur::Gguf file{std::span(bytes)}; });
    }
    auto bytes = fixture({}, {Data{}, Data{}});
    rejected([&] { tokspur::Gguf file{std::span(bytes)}; });
    bytes = fixture({}, {Data{}, Data{"second", {4}, 0, 0, 16}});
    rejected([&] { tokspur::Gguf file{std::span(bytes)}; });
    bytes = fixture({}, {Data{}, Data{"second", {4}, 0, 32, 16}});
    tokspur::Gguf separated{std::span(bytes)};
    require(separated.tensors().size() == 2, "separated tensors");
    constexpr std::array<std::array<unsigned, 3>, 9> blocks{{{0, 1, 4}, {1, 1, 2}, {8, 32, 34},
        {12, 256, 144}, {13, 256, 176}, {14, 256, 210}, {20, 32, 18}, {23, 256, 136}, {30, 1, 2}}};
    for (const auto& block : blocks) {
        bytes = fixture({}, {Data{"weight", {block[1], 3, 2}, block[0], 0, block[2] * 6}});
        tokspur::Gguf file{std::span(bytes)};
        require(file.tensors()[0].bytes == block[2] * 6, "block byte count");
    }
    rejected([] { tokspur::checked_add(uint64_t(-1), 1); });
    rejected([] { tokspur::checked_mul(uint64_t(-1), 2); });
    require(tokspur::checked_mul(0, uint64_t(-1)) == 0, "zero multiplication");
}

void typed_metadata() {
    const auto f32 = std::bit_cast<uint32_t>(1e-6f);
    const auto f64 = std::bit_cast<uint64_t>(10000000.0);
    auto bytes = fixture({{"eps", 6, integer(f32, 4)}, {"base", 12, integer(f64, 8)},
                          {"nan", 6, integer(0x7fc00000, 4)}, {"inf", 12, integer(0x7ff0000000000000, 8)}});
    tokspur::Gguf file{std::span(bytes)};
    require(file.floating_value("eps") == double(1e-6f), "float metadata precision");
    require(file.floating_value("base") == 10000000.0, "double metadata precision");
    rejected([&] { static_cast<void>(file.floating_value("nan")); });
    rejected([&] { static_cast<void>(file.floating_value("inf")); });
    rejected([&] { static_cast<void>(file.floating_value("missing")); });
    rejected([&] { static_cast<void>(file.unsigned_array("eps")); });
    for (const auto& [type, width] : std::array<std::pair<unsigned, unsigned>, 8>{
            {{0, 1}, {1, 1}, {2, 2}, {3, 2}, {4, 4}, {5, 4}, {10, 8}, {11, 8}}}) {
        Bytes values;
        number(values, type, 4); number(values, 3, 8);
        number(values, 0, width); number(values, 1, width); number(values, 42, width);
        bytes = fixture({{"a", 9, values}});
        tokspur::Gguf array{std::span(bytes)};
        require(array.unsigned_array("a") == std::vector<uint64_t>({0, 1, 42}), "typed integer array");
        rejected([&] { static_cast<void>(array.floating_value("a")); });
        rejected([&] { static_cast<void>(array.unsigned_array("missing")); });
        if (type == 1 || type == 3 || type == 5 || type == 11) {
            replace(values, 12, uint64_t(-1), width);
            bytes = fixture({{"a", 9, values}});
            tokspur::Gguf negative{std::span(bytes)};
            rejected([&] { static_cast<void>(negative.unsigned_array("a")); });
        }
    }
    Bytes values;
    number(values, 10, 4); number(values, 1, 8); number(values, uint64_t(-1), 8);
    bytes = fixture({{"a", 9, values}});
    tokspur::Gguf maximum{std::span(bytes)};
    require(maximum.unsigned_array("a")[0] == uint64_t(-1), "full-width unsigned array");
    values.clear(); number(values, 6, 4); number(values, 0, 8);
    bytes = fixture({{"a", 9, values}});
    tokspur::Gguf noninteger{std::span(bytes)};
    rejected([&] { static_cast<void>(noninteger.unsigned_array("a")); });
    values.clear(); number(values, 5, 4); number(values, 0, 8);
    bytes = fixture({{"a", 9, values}});
    tokspur::Gguf empty{std::span(bytes)};
    require(empty.unsigned_array("a").empty(), "empty integer array");
}

void files() {
    Temporary temporary;
    const auto p0 = temporary.write("part0.gguf", fixture(split(0), {Data{"first", {4}, 0, 0, 16}}));
    const auto p1 = temporary.write("part1.gguf", fixture(split(1), {Data{"second", {4}, 0, 0, 16}}));
    tokspur::ModelFiles valid({p1, p0});
    require(valid.architecture() == "test", "split architecture");
    require(valid.tensors().size() == 2, "split tensors");
    require(valid.tensor_bytes("first").size() == 16, "split tensor lookup");
    rejected([&] { static_cast<void>(valid.tensor("missing")); });
    rejected([&] { static_cast<void>(valid.tensor_bytes("missing")); });
    rejected([&] { tokspur::ModelFiles empty({}); });
    rejected([&] { tokspur::ModelFiles duplicate({p0, p0}); });
    rejected([&] { tokspur::ModelFiles incomplete({p0}); });
    const auto plain = temporary.write("plain.gguf", fixture({{"general.architecture", 8, text("toy")}}));
    rejected([&] { tokspur::ModelFiles mixed({p0, plain}); });
    const auto bad_keys = split(1, 2, 3);
    const auto bad = temporary.write("bad.gguf", fixture(bad_keys, {Data{"second", {4}, 0, 0, 16}}));
    rejected([&] { tokspur::ModelFiles mismatched({p0, bad}); });
    auto architecture_keys = split(1);
    architecture_keys[0].value = text("other");
    const auto other = temporary.write("other.gguf", fixture(architecture_keys, {Data{"second", {4}, 0, 0, 16}}));
    rejected([&] { tokspur::ModelFiles mismatched({p0, other}); });
    const auto duplicate_name = temporary.write("duplicate.gguf", fixture(split(1), {Data{"first", {4}, 0, 0, 16}}));
    rejected([&] { tokspur::ModelFiles duplicate({p0, duplicate_name}); });
    const auto empty = temporary.write("empty.gguf", {});
    rejected([&] { tokspur::MappedFile invalid(empty); });
    rejected([&] { tokspur::MappedFile invalid(temporary.root); });
    rejected([&] { tokspur::MappedFile invalid(temporary.root / "missing"); });
    tokspur::MappedFile mapped(plain);
    require(mapped.mapping_mode() == tokspur::ReadOnlyMapMode::private_copy_on_write, "default private mapping");
    rejected([&] { tokspur::MappedFile shared(plain, tokspur::ReadOnlyMapMode::shared_file); });
    tokspur::MappedFile moved(std::move(mapped));
    require(moved.bytes().size() > 16, "move mapping");
    moved.assert_unchanged();
    { std::ofstream changed(plain, std::ios::binary | std::ios::app); changed.put('x'); }
    rejected([&] { moved.assert_unchanged(); });

    const Bytes artifact_bytes{std::byte{1}, std::byte{2}, std::byte{3}};
    for (const auto mode : {0644, 0755}) {
        const auto name = "artifact-" + std::to_string(mode);
        const auto artifact = temporary.write(name.c_str(), artifact_bytes);
        require(chmod(artifact.c_str(), mode) == 0, "set build artifact permissions");
        auto identity = tokspur::execution_identity_artifact(artifact);
        require(identity.mapping_mode() == tokspur::ReadOnlyMapMode::private_copy_on_write,
                "identity hashing maps writable build artifacts privately");
        require(std::equal(identity.bytes().begin(), identity.bytes().end(), artifact_bytes.begin(), artifact_bytes.end()),
                "identity mapping preserves all artifact bytes");
        identity.assert_unchanged();
        { std::ofstream changed(artifact, std::ios::binary | std::ios::app); changed.put('x'); }
        rejected([&] { identity.assert_unchanged(); });
    }

    const auto immutable_bytes = fixture({{"general.architecture", 8, text("immutable")}},
                                        {Data{"tensor", {4}, 0, 0, 16}});
    const auto immutable = temporary.write("immutable.gguf", immutable_bytes);
    require(chmod(immutable.c_str(), 0444) == 0, "make generated pack read-only");
    tokspur::MappedFile shared(immutable, tokspur::ReadOnlyMapMode::shared_file);
    tokspur::MappedFile shared_moved(std::move(shared));
    require(shared_moved.mapping_mode() == tokspur::ReadOnlyMapMode::shared_file, "move shared mapping");
    require(std::equal(shared_moved.bytes().begin(), shared_moved.bytes().end(), immutable_bytes.begin(), immutable_bytes.end()),
            "shared mapping preserves all bytes");
    tokspur::ModelFiles shared_model({immutable}, tokspur::ReadOnlyMapMode::shared_file);
    tokspur::ModelFiles private_model({immutable});
    require(shared_model.architecture() == private_model.architecture() && shared_model.tensors().size() == 1 &&
            std::equal(shared_model.tensor_bytes("tensor").begin(), shared_model.tensor_bytes("tensor").end(),
                       private_model.tensor_bytes("tensor").begin(), private_model.tensor_bytes("tensor").end()),
            "shared and private model parsing preserve operands");
    shared_moved.assert_unchanged();
    require(chmod(immutable.c_str(), 0644) == 0, "restore generated file mode");
    rejected([&] { shared_moved.assert_unchanged(); });
}

} // namespace

int main(int argc, char** argv) {
    try {
        parsing();
        metadata_only_shards();
        metadata();
        tensors();
        typed_metadata();
        files();
        if (argc == 3 && std::string_view(argv[1]) == "--write-fuzz-seeds") {
            const std::filesystem::path directory(argv[2]);
            std::filesystem::create_directories(directory);
            const std::array seeds{
                fixture(),
                fixture({{"general.architecture", 8, text("test")}, {"a", 7, integer(1, 1)}}),
                fixture({}, {Data{"q4", {256}, 12, 0, 144}}),
                fixture(split(0), {Data{"a", {4}, 0, 0, 16}, Data{"b", {4}, 0, 32, 16}}),
            };
            for (size_t i = 0; i < seeds.size(); ++i) {
                std::ofstream file(directory / ("seed-" + std::to_string(i)), std::ios::binary);
                file.write(reinterpret_cast<const char*>(seeds[i].data()), static_cast<std::streamsize>(seeds[i].size()));
                if (!file) throw std::runtime_error("cannot write generated fuzz seed");
            }
        } else if (argc != 1) throw std::runtime_error("unknown test argument");
        std::cout << "GGUF checks passed: " << checks << '\n';
    } catch (const std::exception& error) {
        std::cerr << "GGUF check failed: " << error.what() << '\n';
        return 1;
    }
}
