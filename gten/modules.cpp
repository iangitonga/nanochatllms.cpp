#include <cmath>
#include <cstring>
#include <iostream>

#include "modules.h"
#include "ops.h"


namespace gten {

Embedding::Embedding(int n_vocab, int n_embd, int max_ctx, ModuleDtype dtype)
    : weight{Tensor({n_vocab, n_embd}, dtype.wdtype)},
      emb_acv{Tensor({max_ctx, n_embd}, dtype.adtype)}
{
}

Tensor Embedding::forward(const Tensor& tokens, const int start_pos) {
    Timer timer{&exec_time};
    
    const int n_embd = weight.dimsize(1);
    emb_acv.resize({tokens.numel(), n_embd});

    ops::token_embed(weight, tokens, emb_acv, start_pos);

    return emb_acv;
}

Residual::Residual(int max_ctx, int n_out, Dtype dtype)
    : acv{Tensor({max_ctx, n_out}, dtype)}
{
}

Tensor Residual::forward(const Tensor& inp0, const Tensor& inp1, const int start_pos) {
    Timer timer{&exec_time};

    const int n_ctx = inp0.dimsize(0);
    const int n_embd = inp0.dimsize(1);

    acv.resize({n_ctx, n_embd});
    ops::add(inp0, inp1, acv, start_pos);

    return acv;
}

Linear::Linear(int n_in, int n_out, int max_ctx, ModuleDtype dtype, bool has_bias)
    : weight{Tensor({n_out, n_in}, dtype.wdtype)},
      acv{Tensor({max_ctx, n_out}, dtype.adtype)},
      max_ctx_{max_ctx},
      has_bias_{has_bias}
{
    if (has_bias) {
        bias = Tensor({n_out}, kFloat16);
    }
}

Tensor Linear::forward(const Tensor &inp, const int start_pos) {
    Timer timer{&exec_time};

    const int n_ctx = inp.dimsize(0);
    const int n_out = weight.dimsize(0);
    
    acv.resize({n_ctx, n_out});

    ops::matmul_2d(inp, weight, acv, start_pos);

    if (has_bias_) {
        ops::bias_add_inplace(acv, bias, start_pos);
    }

    return acv;
}

EmbeddingLinear::EmbeddingLinear(int n_embd, int n_vocab, int max_ctx, ModuleDtype dtype)
    : weight{Tensor({n_vocab, n_embd}, dtype.wdtype)}, acv{Tensor({n_vocab}, kFloat32)}
{
}

Tensor EmbeddingLinear::forward(const Tensor& inp)
{
    Timer timer{&exec_time};

    // Hack to allow us to compute the logits for the last token only.
    acv.set_strides({0});
    const int start_pos = inp.dimsize(0) - 1;
    ops::matmul_2d(inp, weight, acv, start_pos);
    acv.set_strides({1});

    return acv;
}

RMSNorm::RMSNorm(int d_in, int max_ctx, ModuleDtype dtype)
    : weight{Tensor({d_in}, kFloat16)}, acv{Tensor({max_ctx, d_in}, dtype.adtype)}
{
}

Tensor RMSNorm::forward(const Tensor& inp, const int start_pos)
{
    Timer timer{&exec_time};

    const int n_ctx = inp.dimsize(0);
    const int n_embd = inp.dimsize(1);

    acv.resize({n_ctx, n_embd});

    ops::rms_norm(inp, weight, acv, start_pos);

    return acv;
}

LayerNorm::LayerNorm(int d_in, int max_ctx, ModuleDtype dtype)
    : weight{Tensor({d_in}, kFloat16)},
      bias{Tensor({d_in}, kFloat16)},
      acv{Tensor({max_ctx, d_in}, dtype.adtype)},
      max_ctx_{max_ctx}
{
}


Tensor LayerNorm::forward(const Tensor &inp, const int start_pos) {
    Timer timer(&exec_time);

    const int n_ctx = inp.dimsize(0);
    const int n_embd = inp.dimsize(1);
    acv.resize({n_ctx, n_embd});
    
    ops::layer_norm(inp, weight, bias, acv, start_pos);

    return acv;
}

Multiply::Multiply(int max_ctx, int d_out, Dtype dtype, const bool inplace)
    : inplace_{inplace}
{
    if (!inplace) {
        acv = Tensor({max_ctx, d_out}, dtype);
    }
}

Tensor Multiply::forward(Tensor &inp0, const Tensor &inp1, const int start_pos)
{
    Timer timer{&exec_time};

    if (inplace_)
    {
        ops::mul_inplace(inp0, inp1, start_pos);

        return inp0;
    } else 
    {
        const int n_ctx = inp0.dimsize(0);
        const int n_embd = inp0.dimsize(1);
        acv.resize({n_ctx, n_embd});

        ops::mul(inp0, inp1, acv, start_pos);

        return acv;
    }
}

SiLU::SiLU(int max_ctx, int d_out, Dtype dtype, const bool inplace)
    : inplace_{inplace}
{
    if (!inplace) {
        acv = Tensor({max_ctx, d_out}, dtype);
    }
}

Tensor SiLU::forward(Tensor &inp, const int start_pos)
{
    Timer timer{&exec_time};

    if (inplace_) {
        ops::silu_inplace(inp, start_pos);

        return inp;
    } else {
        const int n_ctx = inp.dimsize(0);
        const int n_embd = inp.dimsize(1);

        acv.resize({n_ctx, n_embd});
        ops::silu(inp, acv, start_pos);

        return acv;
    }
}

RotaryEmbedding::RotaryEmbedding(const int d_head, const bool inplace, const float rope_pct)
    : d_head_{d_head}, rope_pct_{rope_pct}
{
    GTEN_ASSERT(rope_pct <= 1.0f && rope_pct >= 0.0f);

    if (!inplace) {
        GTEN_ASSERTM(false, "RotaryEmbedding inplace not implemented.");
    }
}

Tensor RotaryEmbedding::forward(Tensor& inp, const int start_pos)
{
    Timer timer{&exec_time};

    const int n_ctx = inp.dimsize(0);
    ops::rotary_emb(inp, d_head_, rope_pct_, start_pos);

    return inp;
}


SelfAttention::SelfAttention(int n_heads, int n_embd, int n_query_groups, int max_ctx, ModuleDtype dtype, float rope_pct, bool qkv_bias)
    : query{Linear(n_embd, n_embd, max_ctx, dtype, /*has_bias=*/qkv_bias)},
      qkv_proj{Linear(n_embd, n_embd, max_ctx, dtype)},
      qk_acv{Tensor({n_heads, max_ctx, max_ctx}, dtype.adtype)},
      qkv_acv{Tensor({max_ctx, n_embd}, dtype.adtype)},
      q_rope{RotaryEmbedding{n_embd/n_heads, /*inplace=*/true, rope_pct}},
      k_rope{RotaryEmbedding{n_embd/n_heads, /*inplace=*/true, rope_pct}},
      n_heads_{n_heads}, max_ctx_{max_ctx}
{
    const int d_head = n_embd / n_heads;
    const int kv_dim = d_head * n_query_groups;
    key = Linear{n_embd, kv_dim, max_ctx, dtype, /*has_bias=*/qkv_bias};
    value = Linear{n_embd, kv_dim, max_ctx, dtype, /*has_bias=*/qkv_bias};
}


Tensor SelfAttention::forward(const Tensor &inp, const int start_pos)
{
    Tensor q = query.forward(inp, start_pos);
    Tensor k = key.forward(inp, start_pos);

    q = q_rope.forward(q, start_pos);
    k = k_rope.forward(k, start_pos);

    Tensor v = value.forward(inp, start_pos);

    const Tensor qkv = masked_qkv_attn(q, k, v, start_pos);
    const Tensor out = qkv_proj.forward(qkv, start_pos);

    return out;
}

Tensor SelfAttention::masked_qkv_attn(const Tensor& q, const Tensor& k, const Tensor& v, const int start_pos)
{
    Timer timer{&exec_time_attn};

    const int n_ctx = q.dimsize(0);
    const int n_embd = q.dimsize(1);

    qk_acv.resize({n_heads_, n_ctx, n_ctx});
    qkv_acv.resize({n_ctx, n_embd});

    ops::qkv_attn(q, k, v, qk_acv, qkv_acv, max_ctx_, start_pos);

    return qkv_acv;
}

AttentionBlock::AttentionBlock(int n_heads, int n_embd, int n_query_groups, int n_mlp, int max_ctx, ModuleDtype dtype)
    : attn_norm{RMSNorm(n_embd, max_ctx, dtype)},
      attn{SelfAttention(n_heads, n_embd, n_query_groups, max_ctx, dtype)},
      inp_res{Residual(max_ctx, n_embd, dtype.adtype)},
      ffn_norm{RMSNorm(n_embd, max_ctx, dtype)},
      ffn_mul{Multiply(max_ctx, n_mlp, dtype.adtype, /*inplace=*/true)},
      ffn_gate_proj{Linear(n_embd, n_mlp, max_ctx, dtype)},
      ffn_up_proj{Linear(n_embd, n_mlp, max_ctx, dtype)},
      ffn_down_proj{Linear(n_mlp, n_embd, max_ctx, dtype)},
      attn_res{Residual(max_ctx, n_embd, dtype.adtype)},
      ffn_silu{SiLU(max_ctx, n_mlp, dtype.adtype, /*inplace=*/true)}
{
}

Tensor AttentionBlock::ffn_forward(const Tensor& inp, const int start_pos) {
    // self.w2(F.silu(self.w1(x)) * self.w3(x))
    Tensor w1 = ffn_gate_proj.forward(inp, start_pos);
    const Tensor w3 = ffn_up_proj.forward(inp, start_pos);
    Tensor sw1 = ffn_silu.forward(w1, start_pos);
    const Tensor w1w2 = ffn_mul.forward(sw1, w3, start_pos);
    Tensor out = ffn_down_proj.forward(w1w2, start_pos);

    return out;
}

Tensor AttentionBlock::forward(Tensor &inp, const int start_pos)
{
    Tensor h = inp_res.forward(inp, attn.forward(attn_norm.forward(inp, start_pos), start_pos), start_pos);
    Tensor out = attn_res.forward(h, ffn_forward(ffn_norm.forward(h, start_pos), start_pos), start_pos);
    return out;
}


AttentionBlock2::AttentionBlock2(int n_heads, int n_embd, int n_query_groups, int n_mlp, int max_ctx, ModuleDtype dtype, float rope_pct, bool qkv_bias)
    : attn_norm{LayerNorm(n_embd, max_ctx, dtype)},
      attn{SelfAttention(n_heads, n_embd, n_query_groups, max_ctx, dtype, rope_pct, /*qkv_bias=*/qkv_bias)},
      inp_res{Residual(max_ctx, n_embd, dtype.adtype)},
      ffn_norm{LayerNorm(n_embd, max_ctx, dtype)},
      ffn_mul{Multiply(max_ctx, n_mlp, dtype.adtype, /*inplace=*/true)},
      ffn_gate_proj{Linear(n_embd, n_mlp, max_ctx, dtype)},
      ffn_up_proj{Linear(n_embd, n_mlp, max_ctx, dtype)},
      ffn_down_proj{Linear(n_mlp, n_embd, max_ctx, dtype)},
      attn_res{Residual(max_ctx, n_embd, dtype.adtype)},
      ffn_silu{SiLU(max_ctx, n_mlp, dtype.adtype, /*inplace=*/true)}
{
}

Tensor AttentionBlock2::ffn_forward(const Tensor& inp, const int start_pos) {
    // self.w2(F.silu(self.w1(x)) * self.w3(x))
    Tensor w1 = ffn_gate_proj.forward(inp, start_pos);
    const Tensor w3 = ffn_up_proj.forward(inp, start_pos);
    Tensor sw1 = ffn_silu.forward(w1, start_pos);
    const Tensor w1w2 = ffn_mul.forward(sw1, w3, start_pos);
    Tensor out = ffn_down_proj.forward(w1w2, start_pos);

    return out;
}

Tensor AttentionBlock2::forward(Tensor &inp, const int start_pos)
{
    Tensor h = inp_res.forward(inp, attn.forward(attn_norm.forward(inp, start_pos), start_pos), start_pos);
    Tensor out = attn_res.forward(h, ffn_forward(ffn_norm.forward(h, start_pos), start_pos), start_pos);
    return out;
}

} // namespace gten
