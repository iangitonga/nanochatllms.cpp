#pragma once


#include <iomanip>

#include "gten/gten.h"


using namespace gten;


struct TinyLLamaConfig {
    const int n_vocab = 32003;
    const int max_ctx = 2048;
    const int n_embd = 2048;
    const int n_ffn = 5632;
    const int n_layers = 22;
    const int n_heads = 32;
    const int n_query_groups = 4;
    const int eos = 32002;
    const int fp16_size_mb = 2200;
    const int q8_size_mb = 1170;
    const int q4_size_mb = 619;
};

static const TinyLLamaConfig tinyllama_cfg = TinyLLamaConfig{};


class TinyLLamaBlock {
public:
    TinyLLamaBlock(int n_heads, int d_embed, int n_query_groups, int n_mlp, int max_ctx, ModuleDtype dtype);
    Tensor forward(Tensor& inp, const int start_pos);
    Tensor ffn_forward(const Tensor& inp, const int start_pos=0);

public:
    RMSNorm attn_norm;
    SelfAttention attn;
    Residual inp_res;
    RMSNorm ffn_norm;
    Linear ffn_gate_proj;
    Linear ffn_up_proj;
    SiLU ffn_silu;
    Multiply ffn_mul;
    Linear ffn_down_proj;
    Residual attn_res;
};



class TinyLLama : public Model {
public:
    TinyLLama(const int n_ctx, ModuleDtype dtype);

    Tensor logits(const Tensor& tokens, const int start_pos=0);
    void load_from_ckpt(std::ifstream& ckpt);
    void print_perf(const int n_pred_tokens);

private:
    ModuleDtype dtype_;
    Embedding tok_emb_;
    RMSNorm norm_;
    EmbeddingLinear lm_head_;
    std::vector<TinyLLamaBlock> blocks_;
};
