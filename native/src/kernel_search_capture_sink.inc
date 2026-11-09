#ifdef TOKSPUR_PRIVATE_KERNEL_SEARCH_CAPTURE
// Private raw slices stay local. No evaluator or LLM endpoint receives these files.
struct TargetKernelSearchCaptureSink {
    int output = -1, manifest = -1;
    uint64_t bytes = 0;
    uint32_t records = 0, grouped = 0, qsa = 0;
    ~TargetKernelSearchCaptureSink() { if (manifest >= 0) close(manifest); if (output >= 0) close(output); }
    void open_directory(const std::filesystem::path& path) {
        output = TargetRouteLockSink::directory(path, true);
        manifest = openat(output, "manifest.jsonl", O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600); require(manifest >= 0, "kernel capture manifest exists");
    }
    void write_file(const std::string& name, std::span<const std::byte> data) {
        require(data.size() <= (640ull << 20) && bytes + data.size() <= (4ull << 30), "kernel capture disk budget");
        const int fd = openat(output, name.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600); require(fd >= 0, "kernel capture refuses existing payload");
        try { TargetRouteAttributionSink::write_all(fd, data); } catch (...) { close(fd); throw; }
        require(close(fd) == 0, "kernel capture close failed"); bytes += data.size();
    }
    static bool checkpoint(void*, const OperatorCheckpointRecord&, const BatchResult*, const BatchResult&) noexcept { return true; }
    static bool emit(void* context, MetalGraph& g, const KernelSearchCaptureRecord& record) noexcept {
        try { static_cast<TargetKernelSearchCaptureSink*>(context)->record(g, record); return true; } catch (...) { return false; }
    }
    void record(MetalGraph& g, const KernelSearchCaptureRecord& r) {
        require(records < 8 && (r.kind == 1 || r.kind == 2), "kernel capture record bound");
        if (r.kind == 1) { require(r.dimensions[2] == 512 && r.dimensions[3] == 10 && r.dimensions[4] == 512 &&
            ((r.encoding == 12 && r.dimensions[0] == 2560 && r.dimensions[1] == 640) || (r.encoding == 7 && r.dimensions[0] == 640 && r.dimensions[1] == 2560)), "actual grouped capture shape"); ++grouped; }
        else { require(r.dimensions[0] == 24 && r.dimensions[1] == 2 && r.dimensions[2] == 256 && r.dimensions[3] == 4 && r.dimensions[5] == 512, "actual QSA capture shape"); ++qsa; }
        Json::Array dimensions, views; for (auto value : r.dimensions) dimensions.push_back(Json::integer(value));
        const auto stem = std::to_string(records); const auto params = std::as_bytes(std::span(r.parameters)).first(r.parameter_bytes);
        write_file(stem + "-parameters.bin", params);
        for (size_t i = 0; i < r.views.size(); ++i) {
            if (!r.views[i].owner) { views.emplace_back(); continue; }
            const auto data = g.read(r.views[i]); const auto name = stem + "-" + std::to_string(i) + ".bin"; write_file(name, data);
            views.push_back(Json(Json::Object{{"file", Json(name)}, {"bytes", Json::integer(data.size())}, {"sha256", Json(TargetRouteLockSink::hash(data))}}));
        }
        const auto line = Json(Json::Object{{"schema", Json::integer(1)}, {"record", Json::integer(records)}, {"kind", Json(r.kind == 1 ? "grouped" : "qsa_channels")},
            {"dataset", Json(r.context.layer == 0 || r.context.layer == 3 ? "search" : "holdout")}, {"layer", number(r.context.layer)}, {"prefix", Json::integer(r.context.prefix)},
            {"encoding", Json::integer(r.encoding)}, {"dimensions", Json(std::move(dimensions))}, {"parameters_file", Json(stem + "-parameters.bin")}, {"parameters_sha256", Json(TargetRouteLockSink::hash(params))}, {"views", Json(std::move(views))}}).dump() + "\n";
        TargetRouteAttributionSink::write_all(manifest, std::as_bytes(std::span(line))); ++records;
    }
    void finish() {
        require(records == 8 && grouped == 4 && qsa == 4, "incomplete kernel slice capture");
        const auto data = TargetRouteLockSink::read_file(output, "manifest.jsonl", 65536);
        const auto line = Json(Json::Object{{"schema", Json::integer(1)}, {"passed", Json(true)}, {"records", Json::integer(records)}, {"grouped", Json::integer(grouped)}, {"qsa", Json::integer(qsa)}, {"bytes", Json::integer(bytes)}, {"manifest_sha256", Json(TargetRouteLockSink::hash(data))}}).dump() + "\n";
        write_file("complete.json", std::as_bytes(std::span(line)));
    }
};
#endif
