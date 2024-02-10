#include <iomanip>

#include "gten/gten.h"
#include "tinyllama.h"


using namespace gten;


TinyLLamaBlock::TinyLLamaBlock(int n_heads, int n_embd, int n_query_groups, int n_mlp, int max_ctx, ModuleDtype dtype)
    : m_attn_norm{RMSNorm(n_embd, max_ctx, dtype)},
      m_self_attn{SelfAttention(n_heads, n_embd, n_query_groups, max_ctx, dtype)},
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

Tensor TinyLLamaBlock::ffn_forward(const Tensor& inp, const int start_pos) {
    // self.w2(F.silu(self.w1(x)) * self.w3(x))
    Tensor w1 = m_mlp_gate_proj.forward(inp, start_pos);
    const Tensor w3 = m_mlp_up_proj.forward(inp, start_pos);
    Tensor sw1 = m_mlp_silu.forward(w1, start_pos);
    const Tensor w1w2 = m_mlp_mul.forward(sw1, w3, start_pos);
    Tensor out = m_mlp_down_proj.forward(w1w2, start_pos);

    return out;
}

Tensor TinyLLamaBlock::forward(Tensor &inp, const int start_pos)
{
    Tensor h = m_inp_res.forward(inp, m_self_attn.forward(m_attn_norm.forward(inp, start_pos), start_pos), start_pos);
    Tensor out = m_attn_res.forward(h, ffn_forward(m_mlp_norm.forward(h, start_pos), start_pos), start_pos);
    return out;
}


TinyLLama::TinyLLama(const int n_ctx, ModuleDtype dtype)
    : Model(n_ctx, tinyllama_cfg.max_ctx),
      m_dtype{dtype},
      m_tok_emb{Embedding(tinyllama_cfg.n_vocab, tinyllama_cfg.n_embd, n_ctx, dtype)},
      m_norm{RMSNorm(tinyllama_cfg.n_embd, n_ctx, {kFloat16, dtype.adtype})},
      m_lm_head{EmbeddingLinear{tinyllama_cfg.n_embd, tinyllama_cfg.n_vocab, n_ctx, {dtype.wdtype, kFloat32}}}
{
    m_blocks.reserve(tinyllama_cfg.n_layers);
    for (int i = 0; i < tinyllama_cfg.n_layers; i++) {
        m_blocks.push_back(
            TinyLLamaBlock(tinyllama_cfg.n_heads, tinyllama_cfg.n_embd, tinyllama_cfg.n_query_groups, tinyllama_cfg.n_ffn, n_ctx, dtype)
        );
    }
}

Tensor TinyLLama::logits(const Tensor& tokens, const int start_pos) {
    if (tokens.numel() > m_max_inference_ctx) {
        std::cerr << "Number of prompt tokens (" << tokens.numel() << ") exceed provided maximum ctx size (" << m_max_inference_ctx << ")\n";
        std::exit(EXIT_FAILURE);
    }

    Tensor logits = m_tok_emb.forward(tokens, start_pos);

    for (auto& block : m_blocks) {
        logits = block.forward(logits, start_pos);
    }

    logits = m_norm.forward(logits, start_pos);
    logits = m_lm_head.forward(logits);

    return logits;
}

void TinyLLama::print_perf(const int n_pred_tokens) {
    int linear_time_ms = 0;
    int attn_time_ms = 0;
    int non_linear_time_ms = 0;

    {
        int norm_time = m_norm.m_exec_time_ms;
        int res_time = 0;
        int rope_time = 0;
        int silu_time = 0;
        int mul_time = 0;
        linear_time_ms += m_lm_head.m_exec_time_ms;

        for (const auto& b : m_blocks) {
            norm_time += b.m_attn_norm.m_exec_time_ms + b.m_mlp_norm.m_exec_time_ms;
            attn_time_ms += b.m_self_attn.m_exec_time_attn_ms;
            res_time  += b.m_attn_res.ms_exec_time_ms + b.m_inp_res.ms_exec_time_ms;
            rope_time += b.m_self_attn.m_q_rope.m_exec_time_ms + b.m_self_attn.m_k_rope.m_exec_time_ms;
            silu_time += b.m_mlp_silu.m_exec_time_ms;
            mul_time  += b.m_mlp_mul.m_exec_time_ms;
            linear_time_ms += b.m_self_attn.m_query.m_exec_time_ms + b.m_self_attn.m_key.m_exec_time_ms + b.m_self_attn.m_value.m_exec_time_ms + b.m_self_attn.m_qkv_proj.m_exec_time_ms;
            linear_time_ms += b.m_mlp_gate_proj.m_exec_time_ms + b.m_mlp_up_proj.m_exec_time_ms + b.m_mlp_down_proj.m_exec_time_ms;
        }

        const int emb_time = m_tok_emb.m_exec_time_ms;
        non_linear_time_ms = emb_time + norm_time + res_time + rope_time + silu_time + mul_time;
    }
    const int tot_inf_time_ms = linear_time_ms + attn_time_ms + non_linear_time_ms;

    const int total_tensor_mem_mb = Tensor::s_tensor_alloc_bytes / 1000000;

    int weights_mem_mb;
    if (m_dtype .wdtype== kFloat16) { weights_mem_mb = tinyllama_cfg.fp16_size_mb; }
    else if (m_dtype .wdtype== kQint8) { weights_mem_mb = tinyllama_cfg.q8_size_mb; }
    else { weights_mem_mb = tinyllama_cfg.q4_size_mb; }

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

void TinyLLama::load_from_ckpt(std::ifstream &ckpt)
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

        // attn_norm
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.m_attn_norm.m_weight, {kFloat16, m_dtype.adtype});

        // ffn_norm
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.m_mlp_norm.m_weight, {kFloat16, m_dtype.adtype});
    }
    
    read_layer_header(ckpt);
    read_into_weight(ckpt, m_norm.m_weight, {kFloat16, m_dtype.adtype});

    read_layer_header(ckpt);
    read_into_weight(ckpt, m_lm_head.m_weight, m_dtype);
}