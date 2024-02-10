// ABSTRACT BASE CLASSES

#pragma once

#include "gten_types.h"
#include "tensor.h"


namespace gten {

// Base class that all models must inherit from.
class Model {
public:
    int m_load_time_ms = 0;
    int m_sample_time_ms = 0;
    int m_max_inference_ctx;
    int m_max_train_ctx;

public:
    Model(int inference_ctx, int train_ctx)
        : m_max_inference_ctx{inference_ctx},
          m_max_train_ctx{train_ctx}
    {
    }
    virtual Tensor logits(const Tensor& tokens, const int start_pos=0) = 0;
    virtual void load_from_ckpt(std::ifstream& ckpt) = 0;
    virtual void print_perf(const int n_pred_tokens) = 0;
};


class Tokenizer {
public:
    int m_vocab_size;
    int m_eos_token;
    std::string m_prompt_prefix = "";
    std::string m_prompt_suffix = "";
    std::vector<int> m_tokens_prefix;
    std::vector<int> m_tokens_suffix;

public:
    Tokenizer(int vocab_size_, int eos_token_,
              const std::string& prompt_prefix_, const std::string& prompt_suffix_,
              const std::vector<int>& prompt_prefix_tokens_, const std::vector<int>& prompt_suffix_tokens_)
        : m_vocab_size{vocab_size_},
          m_eos_token{eos_token_},
          m_prompt_prefix{prompt_prefix_},
          m_prompt_suffix{prompt_suffix_},
          m_tokens_prefix{prompt_prefix_tokens_},
          m_tokens_suffix{prompt_suffix_tokens_}
    {
    }
    virtual const char* decode(int prev_token, int current_token) = 0;
    virtual std::vector<int> encode(std::string& prompt) = 0;
};

} //namespace gten
