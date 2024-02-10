#pragma once


#include "gten/gten.h"


using namespace gten;


struct ZephyrParams {
    const int n_vocab = 100352;
    const int max_ctx = 4096;
    const int n_embd = 2048;
    const int n_ffn = 5632;
    const int n_layers = 24;
    const int n_heads = 32;
    const int n_query_groups = 32;
    const float rope_pct = 0.25f;
    const int eos = 100257;
    const int fp16_size_mb = 3290;
    const int q8_size_mb = 1750;
    const int q4_size_mb = 926;
};

static const ZephyrParams zephyr_cfg = ZephyrParams{};


class ZephyrBlock {
public:
    ZephyrBlock(int n_heads, int d_embed, int n_query_groups, int n_mlp, int max_ctx, ModuleDtype dtype, float rope_pct);
    Tensor forward(Tensor& inp, const int start_pos);
    Tensor ffn_forward(const Tensor& inp, const int start_pos=0);

public:
    LayerNorm attn_norm;
    SelfAttention attn;
    Residual inp_res;
    LayerNorm ffn_norm;
    Linear ffn_gate_proj;
    Linear ffn_up_proj;
    SiLU ffn_silu;
    Multiply ffn_mul;
    Linear ffn_down_proj;
    Residual attn_res;
};


class Zephyr : public Model {
public:
    ModuleDtype m_dtype;

public:
    Zephyr(const int n_ctx, ModuleDtype dtype);
    Tensor logits(const Tensor& tokens, const int start_pos=0);
    void load_from_ckpt(std::ifstream& ckpt);
    void print_perf(const int n_pred_tokens);

private:
    Embedding tok_emb_;
    LayerNorm norm_;
    EmbeddingLinear lm_head_;
    std::vector<ZephyrBlock> blocks_;
};
