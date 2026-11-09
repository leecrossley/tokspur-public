#ifdef TOKSPUR_PRIVATE_ROUTE_ATTRIBUTION
// Private bounded capture. Payloads contain model activations and stay outside tracked source and publication.
struct TargetRouteAttributionSink {
    int directory = -1, manifest = -1;
    uint32_t records = 0;
    uint64_t bytes = 0;
    ~TargetRouteAttributionSink() { if (manifest >= 0) close(manifest); if (directory >= 0) close(directory); }
    static void write_all(int fd, std::span<const std::byte> data) {
        while (!data.empty()) { const auto n = write(fd,data.data(),data.size()); if (n < 0 && errno == EINTR) continue; require(n > 0,"route capture write failed"); data=data.subspan(size_t(n)); }
    }
    void open_directory(const std::filesystem::path& path) {
        require(path.is_absolute() && std::filesystem::canonical(path)==path.lexically_normal() && std::filesystem::is_empty(path),"route capture needs a fresh canonical private directory");
        directory=open(path.c_str(),O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);struct stat st{};
        require(directory>=0 && fstat(directory,&st)==0 && st.st_uid==geteuid() && (st.st_mode&07777)==0700,"route capture directory must be owner-only0700");
        manifest=openat(directory,"manifest.jsonl",O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);require(manifest>=0,"route capture manifest already exists");
    }
    static bool emit(void* context, MetalGraph& g, const RouteAttributionRecord& r) noexcept {
        try { static_cast<TargetRouteAttributionSink*>(context)->record(g,r);return true; } catch (...) { return false; }
    }
    void record(MetalGraph& g,const RouteAttributionRecord& r) {
        constexpr std::array<uint32_t,5> sequence{1,2,5,3,4};
        require(records<240 && r.layer==int32_t(records/5) && r.phase==sequence[records%5],"route capture layer/phase sequence");
        const bool common=r.phase==1||r.phase==5;Json::Array views;
        for(size_t i=0;i<r.views.size();++i) {
            if(!r.views[i].owner){views.emplace_back();continue;}
            const auto data=g.read(r.views[i]);require(data.size()<(256ull<<20),"route capture view bound");
            std::array<unsigned char,CC_SHA256_DIGEST_LENGTH> digest{};CC_SHA256(data.data(),CC_LONG(data.size()),digest.data());
            constexpr char hex[]="0123456789abcdef";std::string hash;for(auto c:digest){hash+=hex[c>>4];hash+=hex[c&15];}
            Json::Object view{{"bytes",Json::integer(data.size())},{"sha256",Json(hash)}};
            if(!common) {
                require(bytes+data.size()<=(2ull<<30),"route capture disk budget");
                const auto name=std::to_string(r.layer)+"-"+std::to_string(r.phase)+"-"+std::to_string(i)+".bin";
                const int fd=openat(directory,name.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);require(fd>=0,"route capture refuses reused output");
                try{write_all(fd,data);}catch(...){close(fd);throw;}require(close(fd)==0,"route capture close failed");
                bytes+=data.size();view.emplace_back("file",Json(name));
            }
            views.emplace_back(std::move(view));
        }
        const auto line=Json(Json::Object{{"schema",Json::integer(1)},{"record",Json::integer(records)},{"layer",Json::integer(r.layer)},{"phase",Json::integer(r.phase)},
            {"rows",Json::integer(r.rows)},{"channels",Json::integer(r.channels)},{"experts",Json::integer(r.experts)},{"selected",Json::integer(r.selected)},
            {"common_layer_input_and_accepted_state",Json(common)},{"views",Json(std::move(views))}}).dump()+"\n";
        write_all(manifest,std::as_bytes(std::span(line)));++records;
    }
};
#endif
