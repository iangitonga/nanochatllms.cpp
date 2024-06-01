#pragma once


#include <iomanip>

#include "gten/gten.h"


using namespace gten;

const int max_n_layers = 36;

struct OpenELMConfig {
public:
    int n_vocab = 32000;
    int max_ctx = 2048;
    int d_head = 64;
    int n_layers;
    int n_embd;
    int num_query_heads[max_n_layers];
    int num_kv_heads[max_n_layers];
    int ffn_intermediate_dim[max_n_layers];
    int fp16_size_mb = 0;
};

static const OpenELMConfig openelm_sm_cfg = {
    .n_layers = 16,
    .n_embd = 1280,
    .num_query_heads = {12, 12, 12, 12, 12, 16, 16, 16, 16, 16, 16, 16, 20, 20, 20, 20},
    .num_kv_heads = {3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5},
    .ffn_intermediate_dim = {768, 1024, 1280, 1536, 1792, 2048, 2560, 2816, 3072, 3328, 3584, 3840, 4352, 4608, 4864, 5120},
    .fp16_size_mb = 544
};

static const OpenELMConfig openelm_md_cfg = {
    .n_layers = 20,
    .n_embd = 1536,
    .num_query_heads = {12, 12, 12, 16, 16, 16, 16, 16, 16, 16, 20, 20, 20, 20, 20, 20, 24, 24, 24, 24},
    .num_kv_heads = {3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6},
    .ffn_intermediate_dim = {768, 1024, 1280, 1536, 1792, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4352, 4608, 5120, 5376, 5632, 5888, 6144},
    .fp16_size_mb = 914
};

static const OpenELMConfig openelm_lg_cfg = {
    .n_layers = 28,
    .n_embd = 2048,
    .num_query_heads = {16, 16, 16, 20, 20, 20, 20, 20, 20, 20, 24, 24, 24, 24, 24, 24, 24, 24, 28, 28, 28, 28, 28, 28, 32, 32, 32, 32},
    .num_kv_heads = {4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8},
    .ffn_intermediate_dim = {1024, 1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608, 4608, 5120, 5376, 5632, 5888, 6144, 6400, 6656, 6912, 7168, 7424, 7680, 7936, 8192},
    .fp16_size_mb = 2160
};

static const OpenELMConfig openelm_xl_cfg = {
    .n_layers = 36,
    .n_embd = 3072,
    .num_query_heads = {12, 12, 12, 12, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 24, 24, 24, 24, 24, 24},
    .num_kv_heads = {3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6},
    .ffn_intermediate_dim = {1536, 1792, 2048, 2560, 2816, 3072, 3328, 3584, 4096, 4352, 4608, 4864, 5120, 5632, 5888, 6144, 6400, 6656, 7168, 7424, 7680, 7936, 8192, 8704, 8960, 9216, 9472, 9728, 10240, 10496, 10752, 11008, 11264, 11776, 12032, 12288},
    .fp16_size_mb = 0
};


class SelfAttention_Elm {
public:
    SelfAttention_Elm(int n_heads, int d_head, int kv_dim, int n_embed, int max_ctx, ModuleDtype dtype, float rope_pct=1.0f, bool qkv_bias=false);
    Tensor forward(const Tensor& inp, const int start_pos);

public:
    Linear m_query;
    Linear m_key;
    Linear m_value;
    Linear m_qkv_proj;
    Tensor m_qk_acv;
    Tensor m_qkv_acv;
    RotaryEmbedding m_q_rope;
    RotaryEmbedding m_k_rope;
    RMSNorm3D m_q_norm;
    RMSNorm3D m_k_norm;
    int m_exec_time_attn_ms{0};

private:
    int32_t m_n_heads;
    int m_max_ctx;

private:
    Tensor masked_qkv_attn(const Tensor& q, const Tensor& k, const Tensor& v, const int start_pos);
};


class OpenELMBlock {
public:
    OpenELMBlock(int n_heads, int d_head, int kv_dim, int d_embed, int n_mlp, int max_ctx, ModuleDtype dtype);
    Tensor forward(Tensor& inp, const int start_pos);
    Tensor ffn_forward(const Tensor& inp, const int start_pos=0);

public:
    RMSNorm m_attn_norm;
    SelfAttention_Elm m_self_attn;
    Residual m_inp_res;
    RMSNorm m_mlp_norm;
    Linear m_mlp_gate_proj;
    Linear m_mlp_up_proj;
    SiLU m_mlp_silu;
    Multiply m_mlp_mul;
    Linear m_mlp_down_proj;
    Residual m_attn_res;
};



class OpenELM : public Model {
public:
    OpenELM(const int n_ctx, ModuleDtype dtype, const OpenELMConfig& cofig);

    Tensor logits(const Tensor& tokens, const int start_pos=0);
    void load_from_ckpt(std::ifstream& ckpt);
    void print_perf(const int n_pred_tokens);

private:
    ModuleDtype m_dtype;
    OpenELMConfig m_config;
    TiedEmbedding m_tok_emb;
    RMSNorm m_norm;
    std::vector<OpenELMBlock> m_blocks;
};
