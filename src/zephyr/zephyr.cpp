#include <iomanip>

#include "zephyr.h"
#include "gten/gten.h"


using namespace gten;


ZephyrBlock::ZephyrBlock(int n_heads, int n_embd, int n_query_groups, int n_mlp, int max_ctx, ModuleDtype dtype, float rope_pct)
    : attn_norm{LayerNorm(n_embd, max_ctx, dtype)},
      attn{SelfAttention(n_heads, n_embd, n_query_groups, max_ctx, dtype, rope_pct, /*qkv_bias=*/true)},
      inp_res{Residual(max_ctx, n_embd, dtype.adtype)},
      ffn_norm{LayerNorm(n_embd, max_ctx, dtype)},
      ffn_gate_proj{Linear(n_embd, n_mlp, max_ctx, dtype)},
      ffn_up_proj{Linear(n_embd, n_mlp, max_ctx, dtype)},
      ffn_silu{SiLU(max_ctx, n_mlp, dtype.adtype, /*inplace=*/true)},
      ffn_mul{Multiply(max_ctx, n_mlp, dtype.adtype, /*inplace=*/true)},
      ffn_down_proj{Linear(n_mlp, n_embd, max_ctx, dtype)},
      attn_res{Residual(max_ctx, n_embd, dtype.adtype)}
{
}

Tensor ZephyrBlock::ffn_forward(const Tensor& inp, const int start_pos) {
    // self.w2(F.silu(self.w1(x)) * self.w3(x))
    Tensor w1 = ffn_gate_proj.forward(inp, start_pos);
    const Tensor w3 = ffn_up_proj.forward(inp, start_pos);
    Tensor sw1 = ffn_silu.forward(w1, start_pos);
    const Tensor w1w2 = ffn_mul.forward(sw1, w3, start_pos);
    Tensor out = ffn_down_proj.forward(w1w2, start_pos);

    return out;
}


Tensor ZephyrBlock::forward(Tensor &inp, const int start_pos)
{
    Tensor h = inp_res.forward(inp, attn.forward(attn_norm.forward(inp, start_pos), start_pos), start_pos);
    Tensor out = attn_res.forward(h, ffn_forward(ffn_norm.forward(h, start_pos), start_pos), start_pos);
    return out;
}


Zephyr::Zephyr(const int n_ctx, ModuleDtype dtype)
    : Model(n_ctx, zephyr_cfg.max_ctx),
      m_dtype{dtype},
      tok_emb_{Embedding(zephyr_cfg.n_vocab, zephyr_cfg.n_embd, n_ctx, dtype)},
      norm_{LayerNorm(zephyr_cfg.n_embd, n_ctx, {kFloat16, dtype.adtype})},
      lm_head_{EmbeddingLinear{zephyr_cfg.n_embd, zephyr_cfg.n_vocab, n_ctx, {dtype.wdtype, kFloat32}}}
{
    blocks_.reserve(zephyr_cfg.n_layers);
    for (int i = 0; i < zephyr_cfg.n_layers; i++) {
        blocks_.push_back(
            ZephyrBlock(zephyr_cfg.n_heads, zephyr_cfg.n_embd, zephyr_cfg.n_query_groups, zephyr_cfg.n_ffn, n_ctx, dtype, zephyr_cfg.rope_pct)
        );
    }
}


Tensor Zephyr::logits(const Tensor& tokens, const int start_pos) {
    if (tokens.numel() > max_inference_ctx) {
        std::cerr << "Number of prompt tokens (" << tokens.numel() << ") exceed provided maximum ctx size (" << max_inference_ctx << ")\n";
        std::exit(EXIT_FAILURE);
    }

    Tensor logits = tok_emb_.forward(tokens, start_pos);

    for (auto& block : blocks_) {
        logits = block.forward(logits, start_pos);
    }

    logits = norm_.forward(logits, start_pos);
    logits = lm_head_.forward(logits);

    return logits;
}


void Zephyr::load_from_ckpt(std::ifstream &ckpt)
{
    Timer load_timer{&load_time_ms};

    const int64_t expected_magic = 0x454c49464e455447;
    int64_t magic;
    ckpt.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    GTEN_ASSERTM(magic == expected_magic, "Magic number in the binary does not match the expected one.\n");

    read_layer_header(ckpt);
    read_into_weight(ckpt, tok_emb_.weight, m_dtype);

    for (auto& block : blocks_)
    {
        // q_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn.query.weight, m_dtype);

        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn.query.bias, {kFloat16, m_dtype.adtype});

        // k_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn.key.weight, m_dtype);

        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn.key.bias, {kFloat16, m_dtype.adtype});

        // v_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn.value.weight, m_dtype);

        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn.value.bias, {kFloat16, m_dtype.adtype});

        // o_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn.qkv_proj.weight, m_dtype);

        // ffn_gate_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.ffn_gate_proj.weight, m_dtype);

        // ffn_up_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.ffn_up_proj.weight, m_dtype);

        // ffn_down_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.ffn_down_proj.weight, m_dtype);

        // attn_norm
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn_norm.weight, {kFloat16, m_dtype.adtype});
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn_norm.bias, {kFloat16, m_dtype.adtype});

        // ffn_norm
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.ffn_norm.weight, {kFloat16, m_dtype.adtype});
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.ffn_norm.bias, {kFloat16, m_dtype.adtype});
    }
    
    read_layer_header(ckpt);
    read_into_weight(ckpt, norm_.weight, {kFloat16, m_dtype.adtype});
    read_layer_header(ckpt);
    read_into_weight(ckpt, norm_.bias, {kFloat16, m_dtype.adtype});

    read_layer_header(ckpt);
    read_into_weight(ckpt, lm_head_.weight, m_dtype);
}

void Zephyr::print_perf(const int n_pred_tokens)
{
    int linear_time_ms = 0;
    int attn_time_ms = 0;
    int non_linear_time_ms = 0;

    {
        const int emb_time = tok_emb_.exec_time;
        int norm_time = norm_.exec_time;
        int res_time = 0;
        int rope_time = 0;
        int activ_time = 0;
        int mul_time = 0;
        linear_time_ms += lm_head_.exec_time;

        for (const auto& b : blocks_) {
            norm_time += b.attn_norm.exec_time + b.ffn_norm.exec_time;
            attn_time_ms += b.attn.exec_time_attn;
            res_time  += b.attn_res.exec_time + b.inp_res.exec_time;
            rope_time += b.attn.q_rope.exec_time + b.attn.k_rope.exec_time;
            activ_time += b.ffn_norm.exec_time;
            mul_time  += b.ffn_mul.exec_time;
            linear_time_ms += b.attn.query.exec_time + b.attn.key.exec_time + b.attn.value.exec_time + b.attn.qkv_proj.exec_time;
            linear_time_ms += b.ffn_gate_proj.exec_time + b.ffn_up_proj.exec_time + b.ffn_down_proj.exec_time;
        }

        non_linear_time_ms = emb_time + norm_time + res_time + rope_time + activ_time + mul_time;
    }
    const int tot_inf_time_ms = linear_time_ms + attn_time_ms + non_linear_time_ms;

    const int total_tensor_mem_mb = Tensor::s_tensor_alloc_bytes / 1000000;

    int weights_mem_mb;
    if (m_dtype.wdtype== kFloat16) { weights_mem_mb = zephyr_cfg.fp16_size_mb; }
    else if (m_dtype.wdtype== kQint8) { weights_mem_mb = zephyr_cfg.q8_size_mb; }
    else { weights_mem_mb = zephyr_cfg.q4_size_mb; }

    const PerformanceMetrics metrics = {
        .tokens_generated = n_pred_tokens,
        .throughput_tok_per_sec = 1000.0f / (float)(tot_inf_time_ms/n_pred_tokens),
        .inference_total_secs = tot_inf_time_ms / 1000,
        .sample_time_secs = sample_time_ms / 1000,
        .load_time_secs = load_time_ms / 1000,
        .total_runtime_secs = (load_time_ms + sample_time_ms + tot_inf_time_ms) / 1000,
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
