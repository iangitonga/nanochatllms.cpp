#include <iomanip>

#include "gten/gten.h"
#include "openelm.h"


using namespace gten;

// s, 768
// s, 192

// 12, s, 64
//  3, s, 64

// 12, s, s

SelfAttention_Elm::SelfAttention_Elm(int q_heads, int d_head, int kv_dim, int n_embd, int max_ctx, ModuleDtype dtype, float rope_pct, bool qkv_bias)
    : m_query{Linear(n_embd, q_heads*d_head, max_ctx, dtype, /*has_bias=*/qkv_bias)},
      m_qkv_proj{Linear(q_heads*d_head, n_embd, max_ctx, dtype)},
      m_qk_acv{Tensor({q_heads, max_ctx, max_ctx}, dtype.adtype)},
      m_qkv_acv{Tensor({max_ctx, n_embd}, dtype.adtype)},
      m_q_rope{RotaryEmbedding{n_embd, d_head, max_ctx, /*inplace=*/false, rope_pct}},
      m_k_rope{RotaryEmbedding{n_embd, d_head, max_ctx, /*inplace=*/false, rope_pct}},
      m_q_norm{RMSNorm3D(n_embd, d_head, max_ctx, dtype)},
      m_k_norm{RMSNorm3D(n_embd, d_head, max_ctx, dtype)},
      m_n_heads{q_heads}, m_max_ctx{max_ctx}
{
    m_key = Linear{n_embd, kv_dim, max_ctx, dtype, /*has_bias=*/qkv_bias};
    m_value = Linear{n_embd, kv_dim, max_ctx, dtype, /*has_bias=*/qkv_bias};
}


Tensor SelfAttention_Elm::forward(const Tensor &inp, const int start_pos)
{
    Tensor q = m_query.forward(inp, start_pos);
    Tensor k = m_key.forward(inp, start_pos);
    Tensor v = m_value.forward(inp, start_pos);

    const int n_ctx = q.dimsize(0);
    const int n_embd = q.dimsize(1);
    const int d_head = 64;
    const int q_n_head = n_embd / d_head;

    Tensor q_norm = q.view({n_ctx, q_n_head, d_head});
    q = m_q_norm.forward(q_norm);

    const int k_n_head = k.dimsize(1) / d_head;
    Tensor k_norm = k.view({n_ctx, k_n_head, d_head});
    k = m_k_norm.forward(k_norm);

    q = m_q_rope.forward(q, start_pos);
    k = m_k_rope.forward(k, start_pos);

    const Tensor qkv = masked_qkv_attn(q, k, v, start_pos);
    const Tensor out = m_qkv_proj.forward(qkv, start_pos);

    return out;
}

Tensor SelfAttention_Elm::masked_qkv_attn(const Tensor& q, const Tensor& k, const Tensor& v, const int start_pos)
{
    Timer timer{&m_exec_time_attn_ms};

    const int n_ctx = q.dimsize(0);
    const int n_embd = q.dimsize(1);

    m_qk_acv.resize({m_n_heads, n_ctx, n_ctx});
    m_qkv_acv.resize({n_ctx, n_embd});

    ops::qkv_attn(q, k, v, m_qk_acv, m_qkv_acv, m_max_ctx, start_pos);

    return m_qkv_acv;
}


OpenELMBlock::OpenELMBlock(int n_heads, int d_head, int kv_dim, int n_embd, int n_mlp, int max_ctx, ModuleDtype dtype)
    : m_attn_norm{RMSNorm(n_embd, max_ctx, dtype)},
      m_self_attn{SelfAttention_Elm(n_heads, d_head, kv_dim, n_embd, max_ctx, dtype)},
      m_inp_res{Residual(max_ctx, n_embd, dtype.adtype)},
      m_mlp_norm{RMSNorm(n_embd, max_ctx, dtype)},
      m_mlp_gate_proj{Linear(n_embd, n_mlp, max_ctx, dtype)},
      m_mlp_up_proj{Linear(n_embd, n_mlp, max_ctx, dtype)},
      m_mlp_silu{SiLU(max_ctx, n_mlp, dtype.adtype, /*inplace=*/true)},
      m_mlp_mul{Multiply(max_ctx, n_mlp, dtype.adtype, /*inplace=*/true)},
      m_mlp_down_proj{Linear(n_mlp, n_embd, max_ctx, dtype)},
      m_attn_res{Residual(max_ctx, n_embd, dtype.adtype)}
{
}

Tensor OpenELMBlock::ffn_forward(const Tensor& inp, const int start_pos) {
    // self.w2(F.silu(self.w1(x)) * self.w3(x))
    // nc, ne . ne, no
    // nc, ne . ne, no*2
    Tensor w1 = m_mlp_gate_proj.forward(inp, start_pos);
    const Tensor w3 = m_mlp_up_proj.forward(inp, start_pos);
    Tensor sw1 = m_mlp_silu.forward(w1, start_pos);
    const Tensor w1w2 = m_mlp_mul.forward(sw1, w3, start_pos);
    Tensor out = m_mlp_down_proj.forward(w1w2, start_pos);

    return out;
}

Tensor OpenELMBlock::forward(Tensor &inp, const int start_pos)
{
    Tensor h = m_inp_res.forward(inp, m_self_attn.forward(m_attn_norm.forward(inp, start_pos), start_pos), start_pos);
    Tensor out = m_attn_res.forward(h, ffn_forward(m_mlp_norm.forward(h, start_pos), start_pos), start_pos);
    return out;
}


OpenELM::OpenELM(const int n_ctx, ModuleDtype dtype, const OpenELMConfig& config)
    : Model(n_ctx, config.max_ctx),
      m_dtype{dtype},
      m_config{config},
      m_tok_emb{TiedEmbedding(config.n_vocab, config.n_embd, n_ctx, dtype)},
      m_norm{RMSNorm(config.n_embd, n_ctx, {kFloat16, dtype.adtype})}
{
    m_blocks.reserve(config.n_layers);
    for (int i = 0; i < config.n_layers; i++) {
        const int n_query_heads = config.num_query_heads[i];
        const int kv_dim = config.d_head * config.num_kv_heads[i];
        const int n_ffn = config.ffn_intermediate_dim[i];

        m_blocks.push_back(
            OpenELMBlock(n_query_heads, config.d_head, kv_dim, config.n_embd, n_ffn, n_ctx, dtype)
        );
    }
}

Tensor OpenELM::logits(const Tensor& tokens, const int start_pos) {
    if (tokens.numel() > m_max_inference_ctx) {
        std::cerr << "Number of prompt tokens (" << tokens.numel() << ") exceed provided maximum ctx size (" << m_max_inference_ctx << ")\n";
        std::exit(EXIT_FAILURE);
    }

    Tensor logits = m_tok_emb.forward_embed(tokens, start_pos);

    for (auto& block : m_blocks) {
        logits = block.forward(logits, start_pos);
    }

    logits = m_norm.forward(logits, start_pos);
    logits = m_tok_emb.forward_proj(logits);

    return logits;
}

void OpenELM::print_perf(const int n_pred_tokens) {
    int linear_time_ms = 0;
    int attn_time_ms = 0;
    int non_linear_time_ms = 0;

    {
        int norm_time = m_norm.m_exec_time_ms;
        int res_time = 0;
        int rope_time = 0;
        int silu_time = 0;
        int mul_time = 0;

        for (const auto& b : m_blocks) {
            norm_time += b.m_attn_norm.m_exec_time_ms + b.m_mlp_norm.m_exec_time_ms + b.m_self_attn.m_q_norm.m_exec_time_ms + b.m_self_attn.m_k_norm.m_exec_time_ms;
            attn_time_ms += b.m_self_attn.m_exec_time_attn_ms;
            res_time  += b.m_attn_res.ms_exec_time_ms + b.m_inp_res.ms_exec_time_ms;
            rope_time += b.m_self_attn.m_q_rope.m_exec_time_ms + b.m_self_attn.m_k_rope.m_exec_time_ms;
            silu_time += b.m_mlp_silu.m_exec_time_ms;
            mul_time  += b.m_mlp_mul.m_exec_time_ms;
            linear_time_ms += b.m_self_attn.m_query.m_exec_time_ms + b.m_self_attn.m_key.m_exec_time_ms + b.m_self_attn.m_value.m_exec_time_ms + b.m_self_attn.m_qkv_proj.m_exec_time_ms;
            linear_time_ms += b.m_mlp_gate_proj.m_exec_time_ms + b.m_mlp_up_proj.m_exec_time_ms + b.m_mlp_down_proj.m_exec_time_ms;
        }

        const int emb_time = m_tok_emb.m_emb_exec_time_ms;
        non_linear_time_ms = emb_time + norm_time + res_time + rope_time + silu_time + mul_time;

        linear_time_ms += m_tok_emb.m_proj_exec_time_ms;
    }
    const int tot_inf_time_ms = linear_time_ms + attn_time_ms + non_linear_time_ms;

    const int total_tensor_mem_mb = Tensor::s_tensor_alloc_bytes / 1000000;

    const int weights_mem_mb = m_config.fp16_size_mb;

    const PerformanceMetrics metrics = {
        .tokens_generated = n_pred_tokens,
        .throughput_tok_per_sec = 1000.0f / (float)(tot_inf_time_ms/n_pred_tokens),
        .inference_total_secs = tot_inf_time_ms / 1000,
        .sample_time_secs = m_sample_time_ms / 1000,
        .load_time_secs = m_load_time_ms / 1000,
        .total_runtime_secs = (m_load_time_ms + m_sample_time_ms + tot_inf_time_ms) / 1000,
        .inference_time_per_tok_ms = tot_inf_time_ms / n_pred_tokens,
        .linear_time_per_tok_ms = linear_time_ms / n_pred_tokens,
        .attn_time_per_tok_ms = attn_time_ms / n_pred_tokens,
        .other_time_ms = non_linear_time_ms / n_pred_tokens,
        .mem_usage_total_mb = total_tensor_mem_mb,
        .mem_usage_weights_mb = weights_mem_mb,
        .mem_usage_acvs_mb = total_tensor_mem_mb - weights_mem_mb
    };

    print_performance_metrics(metrics);
}

void OpenELM::load_from_ckpt(std::ifstream &ckpt)
{
    Timer load_timer{&m_load_time_ms};

    const int64_t expected_magic = 0x454c49464e455447;
    int64_t magic;
    ckpt.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    GTEN_ASSERTM(magic == expected_magic, "Magic number in the binary does not match the expected one.\n");

    read_layer_header(ckpt);
    read_into_weight(ckpt, m_tok_emb.m_weight, m_dtype);

    for (auto& block : m_blocks)
    {
        // q_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.m_self_attn.m_query.m_weight, m_dtype);

        // k_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.m_self_attn.m_key.m_weight, m_dtype);

        // v_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.m_self_attn.m_value.m_weight, m_dtype);

        // o_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.m_self_attn.m_qkv_proj.m_weight, m_dtype);

        // ffn_gate_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.m_mlp_gate_proj.m_weight, m_dtype);

        // ffn_up_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.m_mlp_up_proj.m_weight, m_dtype);

        // ffn_down_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.m_mlp_down_proj.m_weight, m_dtype);

        // q_norm
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.m_self_attn.m_q_norm.m_weight, {kFloat16, m_dtype.adtype});

        // k_norm
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.m_self_attn.m_k_norm.m_weight, {kFloat16, m_dtype.adtype});

        // attn_norm
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.m_attn_norm.m_weight, {kFloat16, m_dtype.adtype});

        // ffn_norm
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.m_mlp_norm.m_weight, {kFloat16, m_dtype.adtype});
    }
    
    read_layer_header(ckpt);
    read_into_weight(ckpt, m_norm.m_weight, {kFloat16, m_dtype.adtype});
}
