// Reuse the complete sampled commit/reject/recovery controls with generated shared Q8 weights and explicit genuine N3 selection.
#define TOKSPUR_TEST_Q8_THREE_ROWS 1
#include "test_sampled_stack.cpp"
#undef main
int main(int argc,char** argv){try{
    require(argc==3 && std::string_view(argv[1])=="--execute-generated","usage: test-q8_three_rows_stack --execute-generated OWNED_METALLIB; root GPU owner only");
    for(const bool views:{false,true})for(const bool concurrent:{false,true})for(const bool reuse:{false,true})sampled_stack(argv[2],reuse,concurrent,views);
    require(cost_width3_cases==96 && production_three_refusals==8 && cost_three_refusals==16,"N3 sampled state coverage differs");
    require(nonfinite_output_refusals==16,"N3 copy/shared-view NaN and infinity coverage differs");
    std::cout<<"N3 sampled state controls checks="<<checks<<" batches="<<batches<<" nonfinite-output-refusals="<<nonfinite_output_refusals<<" full/partial commits, cancellation, failure/reuse and append parity pass; no model or speed admission\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
