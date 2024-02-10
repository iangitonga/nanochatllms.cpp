#include <iomanip>

#include "gten/gten.h"
#include "minicpm.h"


using namespace gten;


static void copy_tensor(const Tensor& src, Tensor& dest)
{
    GTEN_ASSERT(src.is_2d());

    dest.resize(src.shape());

    size_t nbytes;
    if (src.dtype() == kQint4) { nbytes = src.dimsize(0) * (src.dimsize(1) / globs::q4_block_size) * sizeof(Q4Block);  }
    else if (src.dtype() == kQint8) { nbytes = src.dimsize(0) * (src.dimsize(1) / globs::q8_block_size) * sizeof(Q8Block);  }
    else { nbytes = src.dimsize(0) * src.dimsize(1) * src.itemsize();  }

    const char* src_data = src.data_ptr<char>();
    char* dest_data = dest.data_ptr<char>();

    std::memcpy(dest_data, src_data, nbytes);
}


MiniCPMAttentionBlock::MiniCPMAttentionBlock(int n_heads, int n_embd, int n_query_groups, int n_mlp, int max_ctx, ModuleDtype dtype)
    : input_norm{RMSNorm(n_embd, max_ctx, dtype)},
      self_attn{SelfAttention(n_heads, n_embd, n_query_groups, max_ctx, dtype)},
      inp_residual{Residual(max_ctx, n_embd, dtype.adtype)},
      post_attn_norm{RMSNorm(n_embd, max_ctx, dtype)},
      ffn_gate_proj{Linear(n_embd, n_mlp, max_ctx, dtype)},
      ffn_up_proj{Linear(n_embd, n_mlp, max_ctx, dtype)},
      ffn_silu{SiLU(max_ctx, n_mlp, dtype.adtype, /*inplace=*/true)},
      ffn_mul{Multiply(max_ctx, n_mlp, dtype.adtype, /*inplace=*/true)},
      ffn_down_proj{Linear(n_mlp, n_embd, max_ctx, dtype)},
      attn_res{Residual(max_ctx, n_embd, dtype.adtype)}      
{
}

Tensor MiniCPMAttentionBlock::mlp_forward(const Tensor& inp, const int start_pos) {
    Tensor h00 = ffn_gate_proj.forward(inp, start_pos);
    const Tensor h01 = ffn_up_proj.forward(inp, start_pos);
    Tensor sw1 = ffn_silu.forward(h00, start_pos);
    const Tensor w1w2 = ffn_mul.forward(sw1, h01, start_pos);
    Tensor out = ffn_down_proj.forward(w1w2, start_pos);

    return out;
}

Tensor MiniCPMAttentionBlock::forward(Tensor &inp, Tensor& scratch, const int start_pos)
{
    copy_tensor(inp, scratch);

    Tensor h00 = self_attn.forward(input_norm.forward(inp, start_pos), start_pos);
    const float h00_scaler = minicpm_cfg.scale_depth / std::sqrt(minicpm_cfg.n_layers);
    ops::scale(h00, h00_scaler, start_pos); // inplace

    Tensor h01 = inp_residual.forward(scratch, h00, start_pos);
    copy_tensor(h01, scratch);

    Tensor h02 = mlp_forward(post_attn_norm.forward(h01, start_pos), start_pos);
    ops::scale(h02, h00_scaler, start_pos); // inplace

    Tensor out = attn_res.forward(h02, scratch, start_pos);

    return out;
}


MiniCPM::MiniCPM(const int n_ctx, ModuleDtype dtype)
    : Model(n_ctx, minicpm_cfg.max_ctx),
      dtype_{dtype},
      tok_emb_{TiedEmbedding(minicpm_cfg.n_vocab, minicpm_cfg.n_embd, n_ctx, dtype)},
      norm_{RMSNorm(minicpm_cfg.n_embd, n_ctx, {kFloat16, dtype.adtype})},
      res_scratch{Tensor({minicpm_cfg.max_ctx, minicpm_cfg.n_embd}, dtype.adtype)}
{
    blocks_.reserve(minicpm_cfg.n_layers);
    for (int i = 0; i < minicpm_cfg.n_layers; i++) {
        blocks_.push_back(
            MiniCPMAttentionBlock(minicpm_cfg.n_heads, minicpm_cfg.n_embd, minicpm_cfg.n_query_groups, minicpm_cfg.n_ffn, n_ctx, dtype)
        );
    }
}


Tensor MiniCPM::logits(const Tensor& tokens, const int start_pos)
{
    if (tokens.numel() > max_inference_ctx) {
        std::cerr << "Number of prompt tokens (" << tokens.numel() << ") exceed provided maximum ctx size (" << max_inference_ctx << ")\n";
        std::exit(EXIT_FAILURE);
    }

    Tensor logits = tok_emb_.forward_embed(tokens, start_pos);
    ops::scale(logits, minicpm_cfg.scale_emb, start_pos);

    for (auto& block : blocks_) {
        logits = block.forward(logits, res_scratch, start_pos);
    }

    logits = norm_.forward(logits, start_pos);

    const float scaler = 1.0f / (minicpm_cfg.n_embd / minicpm_cfg.dim_model_base);
    ops::scale(logits, scaler, start_pos);

    logits = tok_emb_.forward_proj(logits);

    return logits;
}


void MiniCPM::load_from_ckpt(std::ifstream& ckpt)
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
        read_into_weight(ckpt, block.self_attn.query.weight, dtype_);

        // k_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.self_attn.key.weight, dtype_);

        // v_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.self_attn.value.weight, dtype_);

        // o_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.self_attn.qkv_proj.weight, dtype_);

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
        read_into_weight(ckpt, block.input_norm.weight, {kFloat16, dtype_.adtype});

        // ffn_norm
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.post_attn_norm.weight, {kFloat16, dtype_.adtype});
    }
    
    read_layer_header(ckpt);
    read_into_weight(ckpt, norm_.weight, {kFloat16, dtype_.adtype});
}


void MiniCPM::print_perf(const int n_pred_tokens)
{
    int linear_time_ms = 0;
    int attn_time_ms = 0;
    int non_linear_time_ms = 0;

    {
        int norm_time = norm_.exec_time;
        int res_time = 0;
        int rope_time = 0;
        int silu_time = 0;
        int mul_time = 0;
        linear_time_ms += tok_emb_.proj_exec_time;

        for (const auto& b : blocks_) {
            norm_time += b.input_norm.exec_time + b.post_attn_norm.exec_time;
            attn_time_ms += b.self_attn.exec_time_attn;
            res_time  += b.attn_res.exec_time + b.inp_residual.exec_time;
            rope_time += b.self_attn.q_rope.exec_time + b.self_attn.k_rope.exec_time;
            silu_time += b.ffn_silu.exec_time;
            mul_time  += b.ffn_mul.exec_time;
            linear_time_ms += b.self_attn.query.exec_time + b.self_attn.key.exec_time + b.self_attn.value.exec_time + b.self_attn.qkv_proj.exec_time;
            linear_time_ms += b.ffn_gate_proj.exec_time + b.ffn_up_proj.exec_time + b.ffn_down_proj.exec_time;
        }

        const int emb_time = tok_emb_.emb_exec_time;
        non_linear_time_ms = emb_time + norm_time + res_time + rope_time + silu_time + mul_time;
    }
    const int tot_inf_time_ms = linear_time_ms + attn_time_ms + non_linear_time_ms;

    const int total_tensor_mem_mb = static_cast<int>(Tensor::s_tensor_alloc_bytes / 1000000);

    int weights_mem_mb = 0;
    if (dtype_.wdtype== kFloat16) { weights_mem_mb = minicpm_cfg.fp16_size_mb; }
    else if (dtype_.wdtype== kQint8) { weights_mem_mb = minicpm_cfg.q8_size_mb; }
    else { weights_mem_mb = minicpm_cfg.q4_size_mb; }

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
