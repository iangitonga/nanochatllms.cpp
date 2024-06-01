#pragma once

#include <iostream>

#include "gten_types.h"
#include "tensor.h"


namespace gten {


/// Provides an embedding table lookup for tokens.
class Embedding {
public:
    Tensor m_weight;
    Tensor m_emb_acv;
    int m_exec_time_ms{0};

public:
    Embedding() = default;
    Embedding(int n_vocab, int d_embed, int max_ctx, ModuleDtype dtype);

    /// Returns the embeddings of the given tokens. The input tensor must be of shape
    /// (n_ctx,) and the output tensor is of shape (n_ctx, d_embed).
    Tensor forward(const Tensor& tokens, const int start_pos = 0);
};


/// Provides an embedding table lookup for tokens.
class TiedEmbedding {
public:
    Tensor m_weight;
    Tensor m_emb_acv;
    Tensor m_proj_acv;
    int m_emb_exec_time_ms{0};
    int m_proj_exec_time_ms{0};

public:
    TiedEmbedding() = default;
    TiedEmbedding(int n_vocab, int d_embed, int max_ctx, ModuleDtype dtype);
 
    /// Returns the embeddings of the given tokens. The input tensor must be of shape
    /// (n_ctx,) and the output tensor is of shape (n_ctx, d_embed).
    Tensor forward_embed(const Tensor& tokens, const int start_pos = 0);
    Tensor forward_proj(const Tensor& tokens);
};


class RMSNorm {
public:
    Tensor m_weight;
    Tensor m_acv;
    int m_exec_time_ms{0};

public:
    RMSNorm(int d_in, int max_ctx, ModuleDtype dtype);
    Tensor forward(const Tensor& inp, const int start_pos = 0);
};

class RMSNorm3D {
public:
    Tensor m_weight;
    Tensor m_acv;
    int m_exec_time_ms{0};

public:
    RMSNorm3D(int d_in, int d_norm, int max_ctx, ModuleDtype dtype);
    Tensor forward(const Tensor& inp, const int start_pos = 0);
};


class LayerNorm {
public:
    Tensor m_weight;
    Tensor m_bias;
    Tensor m_acv;
    int m_exec_time_ms{0};

public:
    LayerNorm() = default;
    LayerNorm(int d_in, int max_ctx, ModuleDtype dtype);
    Tensor forward(const Tensor& inp, const int start_pos = 0);

private:
    int m_max_ctx;
};

class Residual {
public:
    Tensor m_acv;
    int ms_exec_time_ms{0};

public:
    Residual() = default;
    Residual(int max_ctx, int d_out, Dtype dtype);
    Tensor forward(const Tensor& inp0, const Tensor& inp1, const int start_pos = 0);
};


/// Applies an affine linear transformation on the input.
class Linear {
public:
    Tensor m_weight;
    Tensor m_bias;
    Tensor m_acv;
    int m_exec_time_ms{0};

public:
    Linear() = default;
    Linear(int d_in, int d_out, int max_ctx, ModuleDtype dtype, bool has_bias=false);
    Tensor forward(const Tensor& inp, const int start_pos = 0);

private:
    int m_max_ctx;
    bool m_has_bias;
};

class EmbeddingLinear {
public:
    Tensor m_weight;
    Tensor m_acv;
    int m_exec_time_ms{0};

public:
    EmbeddingLinear() = default;
    EmbeddingLinear(int n_embd, int n_vocab, int max_ctx, ModuleDtype dtype);
    Tensor forward(const Tensor& inp);
};

class Multiply {
public:
    Tensor m_acv;
    int m_exec_time_ms{0};

public:
    Multiply() = default;
    Multiply(int max_ctx, int d_out, Dtype dtype, const bool inplace = false);
    Tensor forward(Tensor& inp0, const Tensor& inp1, const int start_pos=0);

private:
    bool m_inplace{false};
};

class SiLU {
public:
    Tensor m_acv;
    int m_exec_time_ms{0};

public:
    SiLU() = default;
    SiLU(int max_ctx, int d_out, Dtype dtype, const bool inplace=false);
    Tensor forward(Tensor& inp, const int start_pos=0);

private:
    bool m_inplace{false};
};


class RotaryEmbedding {
public:
    int m_exec_time_ms{0};

public:
    // `rope_pct` is the percentage (in range [0.0, 1.0]) of the head_dim we should apply rope. 
    RotaryEmbedding(const int n_embed, const int d_head, const int max_ctx, const bool inplace=true, const float rope_pct=1.0f);
    Tensor forward(Tensor& inp, const int start_pos=0);

private:
    int m_d_head;
    float m_rope_pct;
    bool m_inplace;
    Tensor m_acv;
};


class SelfAttention {
public:
    SelfAttention(int n_heads, int n_embed, int n_query_groups, int max_ctx, ModuleDtype dtype, float rope_pct=1.0f, bool qkv_bias=false);
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
    int m_exec_time_attn_ms{0};

private:
    int32_t m_n_heads;
    int m_max_ctx;

private:
    Tensor masked_qkv_attn(const Tensor& q, const Tensor& k, const Tensor& v, const int start_pos);
};

} // namespace gten
