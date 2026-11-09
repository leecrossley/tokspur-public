#ifdef TOKSPUR_PRIVATE_ROUTE_LOCK
struct RouteLockControl {
    struct Row {
        uint32_t rows, prefix, offset, context_rows;
        int32_t layer;
        std::string phase;
        std::vector<std::byte> ids, weights;
    };
    std::vector<Row> tape;
    const std::vector<Row>* reference=nullptr;
    size_t cursor=0;
    bool refuse=false;
    uint32_t refused_calls=0;
    static bool emit(void* context,MetalGraph& g,const RouteLockRecord& r) noexcept {
        try {
            auto& s=*static_cast<RouteLockControl*>(context);if(s.refuse)return ++s.refused_calls<2;
            auto ids=g.read(r.ids),weights=g.read(r.weights);
            if(s.reference){require(s.cursor<s.reference->size(),"generated route tape exhausted");const auto& expected=s.reference->at(s.cursor++);
                require(expected.rows==r.rows&&expected.prefix==r.context.prefix&&expected.offset==r.row_offset&&expected.context_rows==r.context.rows&&expected.layer==r.context.layer&&expected.phase==r.context.phase,"generated route tape sequence differs");
                g.write(r.ids,expected.ids);g.write(r.weights,expected.weights);
                require(g.read(r.ids)==expected.ids&&g.read(r.weights)==expected.weights,"generated route replay did not install exact IDs and weights");
            }else s.tape.push_back({r.rows,r.context.prefix,r.row_offset,r.context.rows,r.context.layer,r.context.phase,std::move(ids),std::move(weights)});
            return true;
        }catch(...){return false;}
    }
};
#endif
