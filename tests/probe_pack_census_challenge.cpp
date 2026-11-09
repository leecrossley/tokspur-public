#include "tokspur/derived_pack.hpp"
#include <iostream>
#include <cassert>
#include <string>
#include <vector>

using namespace tokspur::derived_pack;

int main() {
    std::cout << "Testing verify_gpu_pack_census..." << std::endl;

    // Test 1: Empty plan -> fails
    PackPlan empty_plan;
    assert(!verify_gpu_pack_census(empty_plan));

    // Test 2: Build a plan matching build_gpu_pack_census()
    // Exactly 1,254 tensors
    PackPlan valid_plan;
    for (int layer = 0; layer < 48; ++layer) {
        std::string p = "blk." + std::to_string(layer) + ".";
        // 8 HC tensors
        valid_plan.tensors.push_back({p + "hc_attn_norm.weight", 0, "plain", {10240}, 40960, 0, "", ""});
        valid_plan.tensors.push_back({p + "hc_attn_down.weight", 8, "plain", {10240, 320}, 3481600, 0, "", ""});
        valid_plan.tensors.push_back({p + "hc_attn_up.weight", 8, "plain", {320, 10240}, 3481600, 0, "", ""});
        valid_plan.tensors.push_back({p + "hc_attn_inject.weight", 0, "plain", {10240, 4}, 163840, 0, "", ""});
        valid_plan.tensors.push_back({p + "hc_ffn_norm.weight", 0, "plain", {10240}, 40960, 0, "", ""});
        valid_plan.tensors.push_back({p + "hc_ffn_down.weight", 8, "plain", {10240, 320}, 3481600, 0, "", ""});
        valid_plan.tensors.push_back({p + "hc_ffn_up.weight", 8, "plain", {320, 10240}, 3481600, 0, "", ""});
        valid_plan.tensors.push_back({p + "hc_ffn_inject.weight", 0, "plain", {10240, 4}, 163840, 0, "", ""});

        // MoE FFN (8 tensors)
        valid_plan.tensors.push_back({p + "ffn_gate_inp.weight", 0, "plain", {2560, 512}, 5242880, 0, "", ""});
        valid_plan.tensors.push_back({p + "ffn_gate_inp_shexp.weight", 0, "plain", {2560}, 10240, 0, "", ""});
        valid_plan.tensors.push_back({p + "ffn_gate_exps.weight", 12, "plain", {2560, 640, 512}, 474542080, 0, "", ""});
        valid_plan.tensors.push_back({p + "ffn_up_exps.weight", 12, "plain", {2560, 640, 512}, 474542080, 0, "", ""});
        valid_plan.tensors.push_back({p + "ffn_down_exps.weight", 7, "plain", {640, 2560, 512}, 629145600, 0, "", ""});
        valid_plan.tensors.push_back({p + "ffn_gate_shexp.weight", 8, "plain", {2560, 640}, 1740800, 0, "", ""});
        valid_plan.tensors.push_back({p + "ffn_up_shexp.weight", 8, "plain", {2560, 640}, 1740800, 0, "", ""});
        valid_plan.tensors.push_back({p + "ffn_down_shexp.weight", 8, "plain", {640, 2560}, 1740800, 0, "", ""});

        // Layer-specific
        if (layer % 4 == 3) {
            // QSA (10 tensors)
            valid_plan.tensors.push_back({p + "attn_q.weight", 8, "plain", {2560, 12288}, 33423360, 0, "", ""});
            valid_plan.tensors.push_back({p + "attn_k.weight", 8, "plain", {2560, 512}, 1392640, 0, "", ""});
            valid_plan.tensors.push_back({p + "attn_v.weight", 8, "plain", {2560, 512}, 1392640, 0, "", ""});
            valid_plan.tensors.push_back({p + "attn_output.weight", 8, "plain", {6144, 2560}, 16711680, 0, "", ""});
            valid_plan.tensors.push_back({p + "attn_q_norm.weight", 0, "plain", {256}, 1024, 0, "", ""});
            valid_plan.tensors.push_back({p + "attn_k_norm.weight", 0, "plain", {256}, 1024, 0, "", ""});
            valid_plan.tensors.push_back({p + "indexer.q_proj.weight", 30, "plain", {2560, 512}, 2621440, 0, "", ""});
            valid_plan.tensors.push_back({p + "indexer.k_proj.weight", 30, "plain", {2560, 128}, 655360, 0, "", ""});
            valid_plan.tensors.push_back({p + "indexer.q_norm.weight", 0, "plain", {128}, 512, 0, "", ""});
            valid_plan.tensors.push_back({p + "indexer.k_norm.weight", 0, "plain", {128}, 512, 0, "", ""});
        } else {
            // GDN (9 tensors)
            valid_plan.tensors.push_back({p + "attn_qkv.weight", 8, "plain", {2560, 10240}, 27852800, 0, "", ""});
            valid_plan.tensors.push_back({p + "attn_gate.weight", 8, "plain", {2560, 6144}, 16711680, 0, "", ""});
            valid_plan.tensors.push_back({p + "ssm_a", 0, "plain", {48}, 192, 0, "", ""});
            valid_plan.tensors.push_back({p + "ssm_dt.bias", 0, "plain", {48}, 192, 0, "", ""});
            valid_plan.tensors.push_back({p + "ssm_alpha.weight", 0, "plain", {2560, 48}, 491520, 0, "", ""});
            valid_plan.tensors.push_back({p + "ssm_beta.weight", 0, "plain", {2560, 48}, 491520, 0, "", ""});
            valid_plan.tensors.push_back({p + "ssm_conv1d.weight", 0, "plain", {4, 10240}, 163840, 0, "", ""});
            valid_plan.tensors.push_back({p + "ssm_norm.weight", 0, "plain", {128}, 512, 0, "", ""});
            valid_plan.tensors.push_back({p + "ssm_out.weight", 8, "plain", {6144, 2560}, 16711680, 0, "", ""});
        }

        // Layer 1 PLE (6 tensors)
        if (layer == 1) {
            valid_plan.tensors.push_back({p + "ple_key.weight", 8, "plain", {2560, 10240}, 27852800, 0, "", ""});
            valid_plan.tensors.push_back({p + "ple_value.weight", 8, "plain", {2560, 2560}, 6963200, 0, "", ""});
            valid_plan.tensors.push_back({p + "ple_norm_key.weight", 0, "plain", {10240}, 40960, 0, "", ""});
            valid_plan.tensors.push_back({p + "ple_norm_query.weight", 0, "plain", {10240}, 40960, 0, "", ""});
            valid_plan.tensors.push_back({p + "ple_norm_conv.weight", 0, "plain", {10240}, 40960, 0, "", ""});
            valid_plan.tensors.push_back({p + "ple_conv1d.weight", 0, "plain", {4, 10240}, 163840, 0, "", ""});
        }
    }

    // Global output HC (3 tensors)
    valid_plan.tensors.push_back({"output_hc_norm.weight", 0, "plain", {10240}, 40960, 0, "", ""});
    valid_plan.tensors.push_back({"output_hc_down.weight", 8, "plain", {10240, 320}, 3481600, 0, "", ""});
    valid_plan.tensors.push_back({"output_hc_up.weight", 8, "plain", {320, 10240}, 3481600, 0, "", ""});

    // Global output.weight (1 tensor)
    valid_plan.tensors.push_back({"output.weight", 8, "plain", {2560, 248320}, 675430400, 0, "", ""});

    // Native MTP blk.48 sidecar (32 tensors)
    std::vector<std::string> mtp_names = {
        "attn_k.weight", "attn_k_norm.weight", "attn_output.weight", "attn_q.weight", "attn_q_norm.weight", "attn_v.weight",
        "ffn_down_exps.weight", "ffn_down_shexp.weight", "ffn_gate_exps.weight", "ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight", "ffn_gate_shexp.weight",
        "ffn_up_exps.weight", "ffn_up_shexp.weight", "hc_attn_down.weight", "hc_attn_inject.weight", "hc_attn_norm.weight", "hc_attn_up.weight",
        "hc_ffn_down.weight", "hc_ffn_inject.weight", "hc_ffn_norm.weight", "hc_ffn_up.weight", "indexer.k_norm.weight", "indexer.k_proj.weight",
        "indexer.q_norm.weight", "indexer.q_proj.weight", "nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hc_head_down.weight",
        "nextn.hc_head_norm.weight", "nextn.hc_head_up.weight", "nextn.hnorm.weight"
    };
    for (const auto& name : mtp_names) {
        valid_plan.tensors.push_back({"blk.48." + name, 8, "plain", {1}, 100, 0, "", ""});
    }

    assert(valid_plan.tensors.size() == 1254);
    assert(verify_gpu_pack_census(valid_plan));
    std::cout << "Census check passed for 1,254 tensors!" << std::endl;

    // Test 3: CPU embedding exclusion: token_embd.weight
    PackPlan invalid_plan_cpu = valid_plan;
    invalid_plan_cpu.tensors[0].name = "token_embd.weight";
    assert(!is_gpu_pack_tensor("token_embd.weight"));
    assert(!verify_gpu_pack_census(invalid_plan_cpu));
    std::cout << "Strict exclusion of token_embd.weight verified!" << std::endl;

    // Test 4: CPU embedding exclusion: per_layer_token_embd
    PackPlan invalid_plan_ple = valid_plan;
    invalid_plan_ple.tensors[0].name = "per_layer_token_embd.weight";
    assert(!is_gpu_pack_tensor("per_layer_token_embd.weight"));
    assert(!verify_gpu_pack_census(invalid_plan_ple));
    std::cout << "Strict exclusion of per_layer_token_embd.weight verified!" << std::endl;

    // Test 5: Missing 1 tensor (1,253 tensors)
    PackPlan plan_1253 = valid_plan;
    plan_1253.tensors.pop_back();
    assert(plan_1253.tensors.size() == 1253);
    assert(!verify_gpu_pack_census(plan_1253));
    std::cout << "Refusal on 1,253 tensors verified!" << std::endl;

    // Test 6: Extra 1 tensor (1,255 tensors)
    PackPlan plan_1255 = valid_plan;
    plan_1255.tensors.push_back({"extra_tensor", 0, "plain", {1}, 4, 0, "", ""});
    assert(plan_1255.tensors.size() == 1255);
    assert(!verify_gpu_pack_census(plan_1255));
    std::cout << "Refusal on 1,255 tensors verified!" << std::endl;

    // Test 7: Duplicate tensor
    PackPlan plan_dup = valid_plan;
    plan_dup.tensors[1] = plan_dup.tensors[0]; // duplicate name
    assert(!verify_gpu_pack_census(plan_dup));
    std::cout << "Refusal on duplicate tensor verified!" << std::endl;

    // Test 8: Missing output.weight anchor
    PackPlan plan_no_out = valid_plan;
    for (auto& t : plan_no_out.tensors) {
        if (t.name == "output.weight") t.name = "output_replacement.weight";
    }
    assert(!verify_gpu_pack_census(plan_no_out));
    std::cout << "Refusal on missing output.weight anchor verified!" << std::endl;

    // Test 9: Missing MTP block count (e.g. 31 MTP tensors)
    PackPlan plan_bad_mtp = valid_plan;
    for (auto& t : plan_bad_mtp.tensors) {
        if (t.name == "blk.48.attn_k.weight") t.name = "blk.47.extra_attn_k.weight";
    }
    assert(!verify_gpu_pack_census(plan_bad_mtp));
    std::cout << "Refusal on invalid MTP count verified!" << std::endl;

    std::cout << "ALL EMPIRICAL CENSUS CHECKS PASSED!" << std::endl;
    return 0;
}
