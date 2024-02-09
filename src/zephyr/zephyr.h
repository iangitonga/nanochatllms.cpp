#pragma once


#include "gten/gten.h"


using namespace gten;


struct ZephyrParams {
    const int n_vocab = 100352;
    const int max_ctx = 4096;
    const int n_embd = 2048;
    const int n_ffn = 5632;
    const int n_layers = 24;
    const int n_heads = 32;
    const int n_query_groups = 32;
    const float rope_pct = 0.25f;
    const int eos = 100257;
};

static const ZephyrParams zephyr_cfg = ZephyrParams{};


class Zephyr : public Model {
public:
    ModuleDtype m_dtype;

public:
    Zephyr(const int n_ctx, ModuleDtype dtype);
    Tensor logits(const Tensor& tokens, const int start_pos=0);
    void load_from_ckpt(std::ifstream& ckpt);
    void print_perf(const int n_pred_tokens);

private:
    Embedding tok_emb_;
    LayerNorm norm_;
    EmbeddingLinear lm_head_;
    std::vector<AttentionBlock2> blocks_;
};
