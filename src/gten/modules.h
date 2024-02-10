#pragma once

#include <iostream>

#include "gten_types.h"
#include "tensor.h"


namespace gten {


/// Provides an embedding table lookup for tokens.
class Embedding {
public:
    Embedding() = default;
    Embedding(int n_vocab, int d_embed, int max_ctx, ModuleDtype dtype);

    /// Returns the embeddings of the given tokens. The input tensor must be of shape
    /// (n_ctx,) and the output tensor is of shape (n_ctx, d_embed).
    Tensor forward(const Tensor& tokens, const int start_pos = 0);

public:
    Tensor weight;
    Tensor emb_acv;
    int exec_time{0};
};

/// Provides an embedding table lookup for tokens.
class TiedEmbedding {
public:
    TiedEmbedding() = default;
    TiedEmbedding(int n_vocab, int d_embed, int max_ctx, ModuleDtype dtype);
 
    /// Returns the embeddings of the given tokens. The input tensor must be of shape
    /// (n_ctx,) and the output tensor is of shape (n_ctx, d_embed).
    Tensor forward_embed(const Tensor& tokens, const int start_pos = 0);
    Tensor forward_proj(const Tensor& tokens);

public:
    Tensor weight;
    Tensor emb_acv;
    Tensor proj_acv;
    int emb_exec_time{0};
    int proj_exec_time{0};
};


class RMSNorm {
public:
    RMSNorm(int d_in, int max_ctx, ModuleDtype dtype);
    Tensor forward(const Tensor& inp, const int start_pos = 0);

public:
    Tensor weight;
    Tensor acv;
    int exec_time{0};
};


class LayerNorm {
public:
    LayerNorm() = default;
    LayerNorm(int d_in, int max_ctx, ModuleDtype dtype);
    Tensor forward(const Tensor& inp, const int start_pos = 0);

public:
    Tensor weight;
    Tensor bias;
    Tensor acv;
    int exec_time{0};

private:
    int max_ctx_;
};

class Residual {
public:
    Residual() = default;
    Residual(int max_ctx, int d_out, Dtype dtype);
    Tensor forward(const Tensor& inp0, const Tensor& inp1, const int start_pos = 0);

public:
    Tensor acv;
    int exec_time{0};
};


/// Applies an affine linear transformation on the input.
class Linear {
public:
    Linear() = default;
    Linear(int d_in, int d_out, int max_ctx, ModuleDtype dtype, bool has_bias=false);
    Tensor forward(const Tensor& inp, const int start_pos = 0);

public:
    Tensor weight;
    Tensor bias;
    Tensor acv;
    int exec_time{0};

private:
    int max_ctx_;
    bool has_bias_;
};

class EmbeddingLinear {
public:
    EmbeddingLinear() = default;
    EmbeddingLinear(int n_embd, int n_vocab, int max_ctx, ModuleDtype dtype);
    Tensor forward(const Tensor& inp);

public:
    Tensor weight;
    Tensor acv;
    int exec_time{0};
};

class Multiply {
public:
    Multiply() = default;
    Multiply(int max_ctx, int d_out, Dtype dtype, const bool inplace = false);
    Tensor forward(Tensor& inp0, const Tensor& inp1, const int start_pos=0);

public:
    Tensor acv;
    int exec_time{0};

private:
    bool inplace_{false};
};

class SiLU {
public:
    SiLU() = default;
    SiLU(int max_ctx, int d_out, Dtype dtype, const bool inplace=false);
    Tensor forward(Tensor& inp, const int start_pos=0);

public:
    Tensor acv;
    bool inplace_{false};
    int exec_time{0};
};


class RotaryEmbedding {
public:
    // `rope_pct` is the percentage (in range [0.0, 1.0]) of the head_dim we should apply rope. 
    RotaryEmbedding(const int d_head, const bool inplace=true, const float rope_pct=1.0f);
    Tensor forward(Tensor& inp, const int start_pos=0);

public:
    int exec_time{0};

private:
    int d_head_;
    float rope_pct_;
};


class SelfAttention {
public:
    SelfAttention(int n_heads, int n_embed, int n_query_groups, int max_ctx, ModuleDtype dtype, float rope_pct=1.0f, bool qkv_bias=false);
    Tensor forward(const Tensor& inp, const int start_pos);

public:
    Linear query;
    Linear key;
    Linear value;
    Linear qkv_proj;
    Tensor qk_acv;
    Tensor qkv_acv;
    RotaryEmbedding q_rope;
    RotaryEmbedding k_rope;
    int exec_time_attn{0};

private:
    int32_t n_heads_;
    int max_ctx_;

private:
    Tensor masked_qkv_attn(const Tensor& q, const Tensor& k, const Tensor& v, const int start_pos);
};

} // namespace gten
