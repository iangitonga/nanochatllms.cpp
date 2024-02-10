#include <iomanip>

#include "gten/gten.h"
#include "tinyllama.h"


using namespace gten;


TinyLLamaBlock::TinyLLamaBlock(int n_heads, int n_embd, int n_query_groups, int n_mlp, int max_ctx, ModuleDtype dtype)
    : attn_norm{RMSNorm(n_embd, max_ctx, dtype)},
      attn{SelfAttention(n_heads, n_embd, n_query_groups, max_ctx, dtype)},
      inp_res{Residual(max_ctx, n_embd, dtype.adtype)},
      ffn_norm{RMSNorm(n_embd, max_ctx, dtype)},
      ffn_gate_proj{Linear(n_embd, n_mlp, max_ctx, dtype)},
      ffn_up_proj{Linear(n_embd, n_mlp, max_ctx, dtype)},
      ffn_silu{SiLU(max_ctx, n_mlp, dtype.adtype, /*inplace=*/true)},
      ffn_mul{Multiply(max_ctx, n_mlp, dtype.adtype, /*inplace=*/true)},
      ffn_down_proj{Linear(n_mlp, n_embd, max_ctx, dtype)},
      attn_res{Residual(max_ctx, n_embd, dtype.adtype)}
{
}

Tensor TinyLLamaBlock::ffn_forward(const Tensor& inp, const int start_pos) {
    // self.w2(F.silu(self.w1(x)) * self.w3(x))
    Tensor w1 = ffn_gate_proj.forward(inp, start_pos);
    const Tensor w3 = ffn_up_proj.forward(inp, start_pos);
    Tensor sw1 = ffn_silu.forward(w1, start_pos);
    const Tensor w1w2 = ffn_mul.forward(sw1, w3, start_pos);
    Tensor out = ffn_down_proj.forward(w1w2, start_pos);

    return out;
}

Tensor TinyLLamaBlock::forward(Tensor &inp, const int start_pos)
{
    Tensor h = inp_res.forward(inp, attn.forward(attn_norm.forward(inp, start_pos), start_pos), start_pos);
    Tensor out = attn_res.forward(h, ffn_forward(ffn_norm.forward(h, start_pos), start_pos), start_pos);
    return out;
}


TinyLLama::TinyLLama(const int n_ctx, ModuleDtype dtype)
    : Model(n_ctx, tinyllama_cfg.max_ctx),
      dtype_{dtype},
      tok_emb_{Embedding(tinyllama_cfg.n_vocab, tinyllama_cfg.n_embd, n_ctx, dtype)},
      norm_{RMSNorm(tinyllama_cfg.n_embd, n_ctx, {kFloat16, dtype.adtype})},
      lm_head_{EmbeddingLinear{tinyllama_cfg.n_embd, tinyllama_cfg.n_vocab, n_ctx, {dtype.wdtype, kFloat32}}}
{
    blocks_.reserve(tinyllama_cfg.n_layers);
    for (int i = 0; i < tinyllama_cfg.n_layers; i++) {
        blocks_.push_back(
            TinyLLamaBlock(tinyllama_cfg.n_heads, tinyllama_cfg.n_embd, tinyllama_cfg.n_query_groups, tinyllama_cfg.n_ffn, n_ctx, dtype)
        );
    }
}

Tensor TinyLLama::logits(const Tensor& tokens, const int start_pos) {
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

void TinyLLama::print_perf(const int n_pred_tokens) {
    int linear_time_ms = 0;
    int attn_time_ms = 0;
    int non_linear_time_ms = 0;

    {
        int norm_time = norm_.exec_time;
        int res_time = 0;
        int rope_time = 0;
        int silu_time = 0;
        int mul_time = 0;
        linear_time_ms += lm_head_.exec_time;

        for (const auto& b : blocks_) {
            norm_time += b.attn_norm.exec_time + b.ffn_norm.exec_time;
            attn_time_ms += b.attn.exec_time_attn;
            res_time  += b.attn_res.exec_time + b.inp_res.exec_time;
            rope_time += b.attn.q_rope.exec_time + b.attn.k_rope.exec_time;
            silu_time += b.ffn_silu.exec_time;
            mul_time  += b.ffn_mul.exec_time;
            linear_time_ms += b.attn.query.exec_time + b.attn.key.exec_time + b.attn.value.exec_time + b.attn.qkv_proj.exec_time;
            linear_time_ms += b.ffn_gate_proj.exec_time + b.ffn_up_proj.exec_time + b.ffn_down_proj.exec_time;
        }

        const int emb_time = tok_emb_.exec_time;
        non_linear_time_ms = emb_time + norm_time + res_time + rope_time + silu_time + mul_time;
    }
    const int tot_inf_time_ms = linear_time_ms + attn_time_ms + non_linear_time_ms;

    const int total_tensor_mem_mb = Tensor::s_tensor_alloc_bytes / 1000000;

    int weights_mem_mb;
    if (dtype_ .wdtype== kFloat16) { weights_mem_mb = tinyllama_cfg.fp16_size_mb; }
    else if (dtype_ .wdtype== kQint8) { weights_mem_mb = tinyllama_cfg.q8_size_mb; }
    else { weights_mem_mb = tinyllama_cfg.q4_size_mb; }

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

void TinyLLama::load_from_ckpt(std::ifstream &ckpt)
{
    Timer load_timer{&load_time_ms};

    const int64_t expected_magic = 0x454c49464e455447;
    int64_t magic;
    ckpt.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    GTEN_ASSERTM(magic == expected_magic, "Magic number in the binary does not match the expected one.\n");

    read_layer_header(ckpt);
    read_into_weight(ckpt, tok_emb_.weight, dtype_);

    for (auto& block : blocks_)
    {
        // q_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn.query.weight, dtype_);

        // k_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn.key.weight, dtype_);

        // v_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn.value.weight, dtype_);

        // o_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn.qkv_proj.weight, dtype_);

        // ffn_gate_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.ffn_gate_proj.weight, dtype_);

        // ffn_up_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.ffn_up_proj.weight, dtype_);

        // ffn_down_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.ffn_down_proj.weight, dtype_);

        // attn_norm
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn_norm.weight, {kFloat16, dtype_.adtype});

        // ffn_norm
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.ffn_norm.weight, {kFloat16, dtype_.adtype});
    }
    
    read_layer_header(ckpt);
    read_into_weight(ckpt, norm_.weight, {kFloat16, dtype_.adtype});

    read_layer_header(ckpt);
    read_into_weight(ckpt, lm_head_.weight, dtype_);
}