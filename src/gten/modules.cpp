#include <cmath>
#include <cstring>
#include <iostream>

#include "ops.h"
#include "modules.h"
#include "utils.h"


namespace gten {

Embedding::Embedding(int n_vocab, int n_embd, int max_ctx, ModuleDtype dtype)
    : m_weight{Tensor({n_vocab, n_embd}, dtype.wdtype)},
      m_emb_acv{Tensor({max_ctx, n_embd}, dtype.adtype)}
{
}

Tensor Embedding::forward(const Tensor& tokens, const int start_pos) {
    Timer timer{&m_exec_time_ms};
    
    const int n_embd = m_weight.dimsize(1);
    m_emb_acv.resize({tokens.numel(), n_embd});

    ops::token_embed(m_weight, tokens, m_emb_acv, start_pos);

    return m_emb_acv;
}

TiedEmbedding::TiedEmbedding(int n_vocab, int n_embd, int max_ctx, ModuleDtype dtype)
    : m_weight{Tensor({n_vocab, n_embd}, dtype.wdtype)},
      m_emb_acv{Tensor({max_ctx, n_embd}, dtype.adtype)},
      m_proj_acv{Tensor({n_vocab}, kFloat32)}
{

}

Tensor TiedEmbedding::forward_embed(const Tensor& tokens, const int start_pos) {
    Timer timer{&m_emb_exec_time_ms};
    
    const int n_embd = m_weight.dimsize(1);
    m_emb_acv.resize({tokens.numel(), n_embd});

    ops::token_embed(m_weight, tokens, m_emb_acv, start_pos);

    return m_emb_acv;
}

Tensor TiedEmbedding::forward_proj(const Tensor& inp)
{
    Timer timer{&m_proj_exec_time_ms};

    // Hack to allow us to compute the logits for the last token only.
    m_proj_acv.set_strides({0});
    const int start_pos = inp.dimsize(0) - 1;
    ops::matmul_2d(inp, m_weight, m_proj_acv, start_pos);
    m_proj_acv.set_strides({1});

    return m_proj_acv;
}

Residual::Residual(int max_ctx, int n_out, Dtype dtype)
    : m_acv{Tensor({max_ctx, n_out}, dtype)}
{
}

Tensor Residual::forward(const Tensor& inp0, const Tensor& inp1, const int start_pos) {
    Timer timer{&ms_exec_time_ms};

    const int n_ctx = inp0.dimsize(0);
    const int n_embd = inp0.dimsize(1);

    m_acv.resize({n_ctx, n_embd});
    ops::add(inp0, inp1, m_acv, start_pos);

    return m_acv;
}

Linear::Linear(int n_in, int n_out, int max_ctx, ModuleDtype dtype, bool has_bias)
    : m_weight{Tensor({n_out, n_in}, dtype.wdtype)},
      m_acv{Tensor({max_ctx, n_out}, dtype.adtype)},
      m_max_ctx{max_ctx},
      m_has_bias{has_bias}
{
    if (has_bias) {
        m_bias = Tensor({n_out}, kFloat16);
    }
}

Tensor Linear::forward(const Tensor &inp, const int start_pos) {
    Timer timer{&m_exec_time_ms};

    const int n_ctx = inp.dimsize(0);
    const int n_out = m_weight.dimsize(0);
    
    m_acv.resize({n_ctx, n_out});

    ops::matmul_2d(inp, m_weight, m_acv, start_pos);

    if (m_has_bias) {
        ops::bias_add_inplace(m_acv, m_bias, start_pos);
    }

    return m_acv;
}

EmbeddingLinear::EmbeddingLinear(int n_embd, int n_vocab, int max_ctx, ModuleDtype dtype)
    : m_weight{Tensor({n_vocab, n_embd}, dtype.wdtype)}, m_acv{Tensor({n_vocab}, kFloat32)}
{
}

Tensor EmbeddingLinear::forward(const Tensor& inp)
{
    Timer timer{&m_exec_time_ms};

    // Hack to allow us to compute the logits for the last token only.
    m_acv.set_strides({0});
    const int start_pos = inp.dimsize(0) - 1;
    ops::matmul_2d(inp, m_weight, m_acv, start_pos);
    m_acv.set_strides({1});

    return m_acv;
}

RMSNorm::RMSNorm(int d_in, int max_ctx, ModuleDtype dtype)
    : m_weight{Tensor({d_in}, kFloat16)}, m_acv{Tensor({max_ctx, d_in}, dtype.adtype)}
{
}

Tensor RMSNorm::forward(const Tensor& inp, const int start_pos)
{
    Timer timer{&m_exec_time_ms};

    const int n_ctx = inp.dimsize(0);
    const int n_embd = inp.dimsize(1);

    m_acv.resize({n_ctx, n_embd});

    ops::rms_norm(inp, m_weight, m_acv, start_pos);

    return m_acv;
}


RMSNorm3D::RMSNorm3D(int d_in, int d_norm, int max_ctx, ModuleDtype dtype)
    : m_weight{Tensor({d_norm}, kFloat16)}, m_acv{Tensor({max_ctx, d_in}, dtype.adtype)}
{
}

Tensor RMSNorm3D::forward(const Tensor& inp, const int start_pos)
{
    Timer timer{&m_exec_time_ms};

    GTEN_ASSERT(inp.is_3d());

    const int n_ctx = inp.dimsize(0);
    const int n_embd = inp.dimsize(1) * inp.dimsize(2);

    m_acv.resize({n_ctx, n_embd});

    ops::rms_norm(inp, m_weight, m_acv, start_pos);

    return m_acv;
}

LayerNorm::LayerNorm(int d_in, int max_ctx, ModuleDtype dtype)
    : m_weight{Tensor({d_in}, kFloat16)},
      m_bias{Tensor({d_in}, kFloat16)},
      m_acv{Tensor({max_ctx, d_in}, dtype.adtype)},
      m_max_ctx{max_ctx}
{
}


Tensor LayerNorm::forward(const Tensor &inp, const int start_pos) {
    Timer timer(&m_exec_time_ms);

    const int n_ctx = inp.dimsize(0);
    const int n_embd = inp.dimsize(1);
    m_acv.resize({n_ctx, n_embd});
    
    ops::layer_norm(inp, m_weight, m_bias, m_acv, start_pos);

    return m_acv;
}

Multiply::Multiply(int max_ctx, int d_out, Dtype dtype, const bool inplace)
    : m_inplace{inplace}
{
    if (!inplace) {
        m_acv = Tensor({max_ctx, d_out}, dtype);
    }
}

Tensor Multiply::forward(Tensor &inp0, const Tensor &inp1, const int start_pos)
{
    Timer timer{&m_exec_time_ms};

    if (m_inplace)
    {
        ops::multiply_inplace(inp0, inp1, start_pos);

        return inp0;
    } else 
    {
        const int n_ctx = inp0.dimsize(0);
        const int n_embd = inp0.dimsize(1);
        m_acv.resize({n_ctx, n_embd});

        ops::multiply(inp0, inp1, m_acv, start_pos);

        return m_acv;
    }
}

SiLU::SiLU(int max_ctx, int d_out, Dtype dtype, const bool inplace)
    : m_inplace{inplace}
{
    if (!inplace) {
        m_acv = Tensor({max_ctx, d_out}, dtype);
    }
}

Tensor SiLU::forward(Tensor &inp, const int start_pos)
{
    Timer timer{&m_exec_time_ms};

    if (m_inplace) {
        ops::silu_inplace(inp, start_pos);

        return inp;
    } else {
        const int n_ctx = inp.dimsize(0);
        const int n_embd = inp.dimsize(1);

        m_acv.resize({n_ctx, n_embd});
        ops::silu(inp, m_acv, start_pos);

        return m_acv;
    }
}

RotaryEmbedding::RotaryEmbedding(const int n_embed, const int d_head, const int max_ctx, const bool inplace, const float rope_pct)
    : m_d_head{d_head}, m_rope_pct{rope_pct}, m_inplace{inplace}
{
    GTEN_ASSERT(rope_pct <= 1.0f && rope_pct >= 0.0f);

    if (!inplace) {
        m_acv = Tensor({max_ctx, n_embed}, kFloat16);
    }
}

Tensor RotaryEmbedding::forward(Tensor& inp, const int start_pos)
{
    Timer timer{&m_exec_time_ms};

    if (m_inplace) {
        ops::rotary_emb(inp, inp, m_d_head, m_rope_pct, start_pos);

        return inp;
    } else {
        m_acv.resize({inp.dimsize(0), inp.dimsize(1)});

        ops::rotary_emb(inp, m_acv, m_d_head, m_rope_pct, start_pos);

        return m_acv;
    }
}


SelfAttention::SelfAttention(int n_heads, int n_embd, int n_query_groups, int max_ctx, ModuleDtype dtype, float rope_pct, bool qkv_bias)
    : m_query{Linear(n_embd, n_embd, max_ctx, dtype, /*has_bias=*/qkv_bias)},
      m_qkv_proj{Linear(n_embd, n_embd, max_ctx, dtype)},
      m_qk_acv{Tensor({n_heads, max_ctx, max_ctx}, dtype.adtype)},
      m_qkv_acv{Tensor({max_ctx, n_embd}, dtype.adtype)},
      m_q_rope{RotaryEmbedding{n_embd, n_embd/n_heads, max_ctx, /*inplace=*/true, rope_pct}},
      m_k_rope{RotaryEmbedding{n_embd, n_embd/n_heads, max_ctx, /*inplace=*/true, rope_pct}},
      m_n_heads{n_heads}, m_max_ctx{max_ctx}
{
    const int d_head = n_embd / n_heads;
    const int kv_dim = d_head * n_query_groups;
    m_key = Linear{n_embd, kv_dim, max_ctx, dtype, /*has_bias=*/qkv_bias};
    m_value = Linear{n_embd, kv_dim, max_ctx, dtype, /*has_bias=*/qkv_bias};
}


Tensor SelfAttention::forward(const Tensor &inp, const int start_pos)
{
    Tensor q = m_query.forward(inp, start_pos);
    Tensor k = m_key.forward(inp, start_pos);

    q = m_q_rope.forward(q, start_pos);
    k = m_k_rope.forward(k, start_pos);

    Tensor v = m_value.forward(inp, start_pos);

    const Tensor qkv = masked_qkv_attn(q, k, v, start_pos);
    const Tensor out = m_qkv_proj.forward(qkv, start_pos);

    return out;
}

Tensor SelfAttention::masked_qkv_attn(const Tensor& q, const Tensor& k, const Tensor& v, const int start_pos)
{
    Timer timer{&m_exec_time_attn_ms};

    const int n_ctx = q.dimsize(0);
    const int n_embd = q.dimsize(1);

    m_qk_acv.resize({m_n_heads, n_ctx, n_ctx});
    m_qkv_acv.resize({n_ctx, n_embd});

    ops::qkv_attn(q, k, v, m_qk_acv, m_qkv_acv, m_max_ctx, start_pos);

    return m_qkv_acv;
}

} // namespace gten
