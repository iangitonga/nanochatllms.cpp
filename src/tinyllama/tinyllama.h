#pragma once


#include <iomanip>

#include "gten/gten.h"


using namespace gten;


struct TinyLLamaConfig {
    const int n_vocab = 32003;
    const int max_ctx = 2048;
    const int n_embd = 2048;
    const int n_ffn = 5632;
    const int n_layers = 22;
    const int n_heads = 32;
    const int n_query_groups = 4;
    const int eos = 32002;
};

static const TinyLLamaConfig tinyllama_cfg = TinyLLamaConfig{};

class TinyLLama : public Model {
public:
    int64_t load_time = 0;
    int64_t sample_time = 0;

public:
    TinyLLama(const int n_ctx, ModuleDtype dtype);

    Tensor logits(const Tensor& tokens, const int start_pos=0);
    void load_from_ckpt(std::ifstream& ckpt);
    void print_perf(const int n_pred_tokens);

private:
    ModuleDtype dtype_;
    Embedding tok_emb_;
    RMSNorm norm_;
    EmbeddingLinear lm_head_;
    std::vector<AttentionBlock> blocks_;
};
