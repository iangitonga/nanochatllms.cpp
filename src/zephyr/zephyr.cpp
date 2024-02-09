#include <iomanip>

#include "zephyr.h"
#include "gten/gten.h"


using namespace gten;


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
            AttentionBlock2(zephyr_cfg.n_heads, zephyr_cfg.n_embd, zephyr_cfg.n_query_groups, zephyr_cfg.n_ffn, n_ctx, dtype, zephyr_cfg.rope_pct, /*qkv_bias=*/true)
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
    Timer load_timer{&load_time};

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
    int64_t linear_time = 0;
    int64_t attn_time = 0;
    int64_t non_linear_time = 0;

    {
        const int64_t emb_time = tok_emb_.exec_time;
        int64_t norm_time = norm_.exec_time;
        int64_t res_time = 0;
        int64_t rope_time = 0;
        int64_t activ_time = 0;
        int64_t mul_time = 0;
        linear_time += lm_head_.exec_time;

        for (const auto& b : blocks_) {
            norm_time += b.attn_norm.exec_time + b.ffn_norm.exec_time;
            attn_time += b.attn.exec_time_attn;
            res_time  += b.attn_res.exec_time + b.inp_res.exec_time;
            rope_time += b.attn.q_rope.exec_time + b.attn.k_rope.exec_time;
            activ_time += b.ffn_norm.exec_time;
            mul_time  += b.ffn_mul.exec_time;
            linear_time += b.attn.query.exec_time + b.attn.key.exec_time + b.attn.value.exec_time + b.attn.qkv_proj.exec_time;
            linear_time += b.ffn_gate_proj.exec_time + b.ffn_up_proj.exec_time + b.ffn_down_proj.exec_time;
        }

        non_linear_time = emb_time + norm_time + res_time + rope_time + activ_time + mul_time;
    }
    const int64_t tot_inf_time = linear_time + attn_time + non_linear_time;

    const int64_t tensor_mem = Tensor::s_tensor_alloc_bytes;
    int64_t weights_mem = 0;

    {
        const auto bytes = [](const Tensor& t) { return t.nbytes(); };

        weights_mem += bytes(tok_emb_.weight);
        weights_mem += bytes(norm_.weight) + bytes(norm_.bias);
        weights_mem += bytes(lm_head_.weight);

        for (const auto& b : blocks_) {
            weights_mem += bytes(b.attn_norm.weight) + bytes(b.attn_norm.bias) + bytes(b.ffn_norm.weight) + bytes(b.ffn_norm.bias);
            weights_mem += bytes(b.attn.query.weight) + bytes(b.attn.key.weight) + bytes(b.attn.value.weight) + bytes(b.attn.qkv_proj.weight);
            weights_mem += bytes(b.attn.query.bias) + bytes(b.attn.key.bias) + bytes(b.attn.value.bias);
            weights_mem += bytes(b.ffn_gate_proj.weight) + bytes(b.ffn_up_proj.weight) + bytes(b.ffn_down_proj.weight);
        }
    }

    const int acv_mem = tensor_mem - weights_mem;


    std::cout << "\n-------------------------------\n";
    std::cout << " " << "PERFORMANCE\n";
    std::cout << "-------------------------------\n";
    std::cout << " " << "Throughput [tok/s]  : " << std::setw(5) << 1000.0f / (float)(tot_inf_time/n_pred_tokens) << "\n";
    std::cout << " " << "Inference [per tok] : " << std::setw(5) << tot_inf_time/n_pred_tokens << "ms\n";
    std::cout << " " << "Sample time         : " << std::setw(5) << sample_time << "ms\n";
    std::cout << " " << "Load time           : " << std::setw(5) << load_time << "ms\n";
    std::cout << " " << "Inference [total]   : " << std::setw(5) << tot_inf_time << "ms\n";
    std::cout << " " << "Total runtime       : " << std::setw(5) << load_time + sample_time + tot_inf_time << "ms\n";
    std::cout << "-------------------------------\n";
    std::cout << " " << "Mem usage [total]   : " << std::setw(4) << tensor_mem/1000000 << "MB\n";
    std::cout << " " << "Mem usage [model]   : " << std::setw(4) << weights_mem/1000000 << "MB\n";
    std::cout << " " << "Mem usage [actvs]   : " << std::setw(4) << acv_mem/1000000 << "MB\n";
    std::cout << "-------------------------------\n";
    std::cout << " " << "Lin time [per tok]  : " << std::setw(5) << linear_time/n_pred_tokens << "ms\n";
    std::cout << " " << "Attn time [per tok] : " << std::setw(5) << attn_time/n_pred_tokens << "ms\n";
    std::cout << " " << "Other     [per tok] : " << std::setw(5) << non_linear_time/n_pred_tokens << "ms\n";
    std::cout << "-------------------------------\n\n";
}
