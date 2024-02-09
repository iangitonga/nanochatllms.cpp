#pragma once

#include "gten/gten.h"


using namespace gten;


struct MiniCPMConfig {
    const int n_vocab = 122753;
    const int max_ctx = 2048;
    const int n_embd = 2304;
    const int n_ffn = 5760;
    const int n_layers = 40;
    const int n_heads = 36;
    const int n_query_groups = 36;
    const float scale_emb = 12.0f;
    const int dim_model_base = 256;
    const float scale_depth = 1.4f;
    const int eos = 2;
};

static const MiniCPMConfig minicpm_cfg = MiniCPMConfig{};


class MiniCPMAttentionBlock {
public:
    MiniCPMAttentionBlock(int n_heads, int d_embed, int n_query_groups, int n_mlp, int max_ctx, ModuleDtype dtype);
    Tensor forward(Tensor& inp, Tensor& scratch, const int start_pos);
    Tensor mlp_forward(const Tensor& inp, const int start_pos=0);

public:
    RMSNorm input_norm;
    SelfAttention self_attn;
    Residual inp_residual;
    RMSNorm post_attn_norm;
    Linear ffn_gate_proj;
    Linear ffn_up_proj;
    SiLU ffn_silu;
    Multiply ffn_mul;
    Linear ffn_down_proj;
    Residual attn_res;
};


class MiniCPM : public Model {
public:
    ModuleDtype dtype_;

public:
    MiniCPM(const int n_ctx, ModuleDtype dtype);

    Tensor logits(const Tensor& tokens, const int start_pos=0);
    void load_from_ckpt(std::ifstream& ckpt);
    void print_perf(const int n_pred_tokens);

private:
    TiedEmbedding tok_emb_;
    RMSNorm norm_;
    std::vector<MiniCPMAttentionBlock> blocks_;
    Tensor res_scratch; // Used to hold a tensor copy for residual ops.

public:
    int64_t load_time = 0;
    int64_t sample_time = 0;
};
