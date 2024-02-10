// ABSTRACT BASE CLASSES

#pragma once

#include "tensor.h"
#include "gten_types.h"


namespace gten {

// Base class that all models must inherit from.
class Model {
public:
    int load_time_ms = 0;
    int sample_time_ms = 0;
    int max_inference_ctx;
    int max_train_ctx;

public:
    Model(int inference_ctx, int train_ctx)
        : max_inference_ctx{inference_ctx},
          max_train_ctx{train_ctx}
    {
    }
    virtual Tensor logits(const Tensor& tokens, const int start_pos=0) = 0;
    virtual void load_from_ckpt(std::ifstream& ckpt) = 0;
    virtual void print_perf(const int n_pred_tokens) = 0;
};


class Tokenizer {
public:
    int vocab_size;
    int eos_token;
    std::string prompt_prefix = "";
    std::string prompt_suffix = "";
    std::vector<int> prompt_prefix_tokens;
    std::vector<int> prompt_suffix_tokens;

public:
    Tokenizer(int vocab_size_, int eos_token_,
              const std::string& prompt_prefix_, const std::string& prompt_suffix_,
              const std::vector<int>& prompt_prefix_tokens_, const std::vector<int>& prompt_suffix_tokens_)
        : vocab_size{vocab_size_},
          eos_token{eos_token_},
          prompt_prefix{prompt_prefix_},
          prompt_suffix{prompt_suffix_},
          prompt_prefix_tokens{prompt_prefix_tokens_},
          prompt_suffix_tokens{prompt_suffix_tokens_}
    {
    }
    virtual const char* decode(int prev_token, int current_token) = 0;
    virtual std::vector<int> encode(std::string& prompt) = 0;
};

} //namespace gten
