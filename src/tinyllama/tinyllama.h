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
    RMSNorm m_attn_norm;
    SelfAttention m_self_attn;
    Residual m_inp_res;
    RMSNorm m_mlp_norm;
    Linear m_mlp_gate_proj;
    Linear m_mlp_up_proj;
    SiLU m_mlp_silu;
    Multiply m_mlp_mul;
    Linear m_mlp_down_proj;
    Residual m_attn_res;
};



class TinyLLama : public Model {
public:
    TinyLLama(const int n_ctx, ModuleDtype dtype);

    Tensor logits(const Tensor& tokens, const int start_pos=0);
    void load_from_ckpt(std::ifstream& ckpt);
    void print_perf(const int n_pred_tokens);

private:
    ModuleDtype m_dtype;
    Embedding m_tok_emb;
    RMSNorm m_norm;
    EmbeddingLinear m_lm_head;
    std::vector<TinyLLamaBlock> m_blocks;
};
