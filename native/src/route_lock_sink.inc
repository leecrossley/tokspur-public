#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
// Private diagnostic tape. Exact expert IDs and weights never leave the local receipt directory.
struct TargetRouteLockSink {
    int output=-1, manifest=-1, reference=-1;
    uint32_t records=0, rows=0, changed_ids=0, changed_weights=0;
    uint64_t bytes=0;
    bool census_only=false;
    uint32_t request_index=0;
    std::vector<Json> tape;
    ~TargetRouteLockSink(){if(manifest>=0)close(manifest);if(output>=0)close(output);if(reference>=0)close(reference);}
    static std::string hash(std::span<const std::byte> data) {
        std::array<unsigned char,CC_SHA256_DIGEST_LENGTH> digest{};CC_SHA256(data.data(),CC_LONG(data.size()),digest.data());
        constexpr char hex[]="0123456789abcdef";std::string result;for(auto c:digest){result+=hex[c>>4];result+=hex[c&15];}return result;
    }
    static int directory(const std::filesystem::path& path,bool fresh) {
        require(path.is_absolute()&&std::filesystem::canonical(path)==path.lexically_normal()&&(!fresh||std::filesystem::is_empty(path)),"route lock needs a canonical private directory, fresh for output");
        const int fd=open(path.c_str(),O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);struct stat st{};
        require(fd>=0&&fstat(fd,&st)==0&&st.st_uid==geteuid()&&(st.st_mode&07777)==0700,"route lock directory must be owner-only0700");return fd;
    }
    static std::vector<std::byte> read_file(int dir,std::string_view name,uint64_t limit) {
        require(name.find('/')==std::string::npos&&name!="."&&name!="..","route tape filename");
        const std::string filename(name);const int fd=openat(dir,filename.c_str(),O_RDONLY|O_NOFOLLOW|O_CLOEXEC);struct stat st{};
        require(fd>=0,"missing route tape payload");
        try {require(fstat(fd,&st)==0&&S_ISREG(st.st_mode)&&st.st_uid==geteuid()&&(st.st_mode&07777)==0600&&st.st_size>=0&&uint64_t(st.st_size)<=limit,"route tape file extent or ownership");
            std::vector<std::byte> data(size_t(st.st_size));size_t at=0;while(at<data.size()){const auto n=read(fd,data.data()+at,data.size()-at);if(n<0&&errno==EINTR)continue;require(n>0,"route tape short read");at+=size_t(n);}close(fd);return data;
        }catch(...){close(fd);throw;}
    }
    void write_file(const std::string& name,std::span<const std::byte> data) {
        require(bytes+data.size()<=32ull<<20,"route lock output budget");const int fd=openat(output,name.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);require(fd>=0,"route lock refuses existing payload");
        try{TargetRouteAttributionSink::write_all(fd,data);}catch(...){close(fd);throw;}require(close(fd)==0,"route lock close failed");bytes+=data.size();
    }
    void open_directory(const std::filesystem::path& path,const std::optional<std::filesystem::path>& source,bool census=false) {
        census_only=census;require(!census_only||!source,"census cannot replay routes");
        output=directory(path,true);manifest=openat(output,"manifest.jsonl",O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);require(manifest>=0,"route lock manifest already exists");
        if(source){reference=directory(*source,false);const auto data=read_file(reference,"manifest.jsonl",2ull<<20);std::istringstream in(std::string(reinterpret_cast<const char*>(data.data()),data.size()));std::string line;
            while(std::getline(in,line)){require(tape.size()<1024,"route lock tape record bound");tape.push_back(parse_json(line));}require(!tape.empty(),"empty route lock tape");
            const auto done=read_file(reference,"complete.json",4096);const auto sealed=parse_json(std::string(reinterpret_cast<const char*>(done.data()),done.size()));
            require(field(sealed,"records").unsigned_integer()==tape.size()&&field(sealed,"manifest_sha256").string()==hash(data),"route lock reference is not complete or manifest changed");
        }
    }
    static bool emit(void* context,MetalGraph& g,const RouteLockRecord& r) noexcept {try{static_cast<TargetRouteLockSink*>(context)->record(g,r);return true;}catch(...){return false;}}
    void record(MetalGraph& g,const RouteLockRecord& r) {
        if(census_only){
            require(records<65536,"occupancy census record bound");
            const auto ids=g.read(r.ids);require(ids.size()==uint64_t(r.rows)*r.selected*4,"occupancy census payload extent");
            std::array<uint32_t,512> histogram{};
            for(uint32_t row=0;row<r.rows;++row){std::set<uint32_t> unique;for(uint32_t k=0;k<r.selected;++k){
                uint32_t id;std::memcpy(&id,ids.data()+(uint64_t(row)*r.selected+k)*4,4);
                require(id<512&&unique.insert(id).second,"occupancy census invalid selected ID");++histogram[id];}}
            Json::Array counts;for(uint32_t id=0;id<512;++id)if(histogram[id])counts.push_back(Json(Json::Array{Json::integer(id),Json::integer(histogram[id])}));
            const auto line=Json(Json::Object{{"schema",Json::integer(1)},{"record",Json::integer(records)},{"request",Json::integer(request_index)},
                {"phase",Json(r.context.phase)},{"layer",number(r.context.layer)},{"prefix",Json::integer(r.context.prefix)},
                {"context_rows",Json::integer(r.context.rows)},{"rows",Json::integer(r.rows)},{"row_offset",Json::integer(r.row_offset)},
                {"experts",Json::integer(r.experts)},{"selected",Json::integer(r.selected)},{"occupancy",Json(std::move(counts))},
                {"ids_sha256",Json(hash(ids))},{"replayed",Json(false)}}).dump()+"\n";
            require(bytes+line.size()<=64ull<<20,"occupancy census 64 MiB output budget");
            TargetRouteAttributionSink::write_all(manifest,std::as_bytes(std::span(line)));bytes+=line.size();++records;rows+=r.rows;return;
        }
        require(records<1024,"route lock record bound");
        const auto logits=g.read(r.logits),ids=g.read(r.ids),weights=g.read(r.weights);
        const auto check=[&](std::span<const std::byte> id_bytes,std::span<const std::byte> weight_bytes){
            require(id_bytes.size()==uint64_t(r.rows)*10*4&&weight_bytes.size()==id_bytes.size(),"route lock payload extent");
            for(uint32_t t=0;t<r.rows;++t){std::set<uint32_t> unique;double sum=0;for(uint32_t k=0;k<10;++k){uint32_t id;float weight;const auto at=(uint64_t(t)*10+k)*4;std::memcpy(&id,id_bytes.data()+at,4);std::memcpy(&weight,weight_bytes.data()+at,4);require(id<512&&unique.insert(id).second&&std::isfinite(weight)&&weight>=0&&weight<=1,"route lock invalid selection");sum+=weight;}require(std::abs(sum-1)<1e-5,"route lock weights are not normalised");}
        };
        check(ids,weights);for(size_t at=0;at<logits.size();at+=4){float x;std::memcpy(&x,logits.data()+at,4);require(std::isfinite(x),"route lock candidate router is nonfinite");}
        const auto stem=std::to_string(records);const auto id_name=stem+"-ids.bin",weight_name=stem+"-weights.bin";
        Json::Object key{{"record",Json::integer(records)},{"phase",Json(r.context.phase)},{"layer",number(r.context.layer)},
            {"prefix",Json::integer(r.context.prefix)},{"context_rows",Json::integer(r.context.rows)},{"rows",Json::integer(r.rows)},{"row_offset",Json::integer(r.row_offset)},
            {"channels",Json::integer(r.channels)},{"experts",Json::integer(r.experts)},{"selected",Json::integer(r.selected)}};
        auto chosen_ids=ids,chosen_weights=weights;
        if(reference>=0){require(records<tape.size(),"route lock tape exhausted");const auto& expected=tape[records];for(const auto& [name,value]:key)require(field(expected,name).dump()==value.dump(),"route lock call sequence differs");
            chosen_ids=read_file(reference,field(expected,"ids_file").string(),512*10*4);chosen_weights=read_file(reference,field(expected,"weights_file").string(),512*10*4);
            require(hash(chosen_ids)==field(expected,"ids_sha256").string()&&hash(chosen_weights)==field(expected,"weights_sha256").string(),"route lock reference payload changed");check(chosen_ids,chosen_weights);
            changed_ids+=ids!=chosen_ids;changed_weights+=weights!=chosen_weights;g.write(r.ids,chosen_ids);g.write(r.weights,chosen_weights);
            require(g.read(r.ids)==chosen_ids&&g.read(r.weights)==chosen_weights,"route lock replay did not install exact IDs and weights");
        }
        write_file(id_name,chosen_ids);write_file(weight_name,chosen_weights);
        key.emplace_back("schema",Json::integer(1));key.emplace_back("replayed",Json(reference>=0));key.emplace_back("candidate_router_sha256",Json(hash(logits)));
        key.emplace_back("candidate_ids_sha256",Json(hash(ids)));key.emplace_back("candidate_weights_sha256",Json(hash(weights)));
        key.emplace_back("ids_file",Json(id_name));key.emplace_back("weights_file",Json(weight_name));key.emplace_back("ids_sha256",Json(hash(chosen_ids)));key.emplace_back("weights_sha256",Json(hash(chosen_weights)));
        const auto line=Json(std::move(key)).dump()+"\n";TargetRouteAttributionSink::write_all(manifest,std::as_bytes(std::span(line)));++records;rows+=r.rows;
    }
    void finish() {
        require(records&&rows&&(!tape.size()||records==tape.size()),"route lock did not consume its complete reference");
        const auto data=read_file(output,"manifest.jsonl",census_only?64ull<<20:2ull<<20);
        const auto line=Json(Json::Object{{"schema",Json::integer(1)},{"records",Json::integer(records)},{"rows",Json::integer(rows)},
            {"replayed",Json(reference>=0)},{"changed_id_records",Json::integer(changed_ids)},{"changed_weight_records",Json::integer(changed_weights)},{"manifest_sha256",Json(hash(data))}}).dump()+"\n";
        if(census_only)bytes=0;
        write_file("complete.json",std::as_bytes(std::span(line)));
    }
};
#endif
