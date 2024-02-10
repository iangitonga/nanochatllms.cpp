#include <algorithm>
#include <random>

#include "abc.h"
#include "utils.h"
#include "sampler.h"


namespace gten {

void greedy_sample(std::string& prompt, Model* model, Tokenizer* tokenizer, bool showstat)
{
    const int n_predict = std::min(model->m_max_inference_ctx, model->m_max_train_ctx);
    std::vector<int> tokens = tokenizer->encode(prompt);
    tokens.reserve(n_predict);

    const int max_iters = n_predict - tokens.size();
    int n_iters = 0;
    for (int i = 0; i < max_iters; i++)
    {
        n_iters += 1;

        Tensor input{tokens.data(), {(int)tokens.size()}, kInt32};

        const int start_pos = (i == 0) ? 0 : input.numel() - 1; 
        Tensor logits = model->logits(input, start_pos);

        Timer sample_timer{&model->m_sample_time_ms};

        const int logits_size = logits.numel();
        const float *logits_data = logits.data_ptr<float>();

        float max_prob = -std::numeric_limits<float>::infinity();
        int max_index = 0;
        for (int j = 0; j < logits_size; ++j){
            const float val = logits_data[j];
            if (val > max_prob) {
                max_prob = val;
                max_index = j;
            }
        }

        const int pred_token = max_index;

        if (pred_token == tokenizer->m_eos_token) {
            break;
        }

        const int prev_token = (i == 0) ? 1 : tokens.back();
        std::cerr << tokenizer->decode(prev_token, pred_token);

        tokens.push_back(pred_token);
    }
    
    std::cerr << '\n';

    if (showstat) {
        model->print_perf(n_iters);
    }
}


void topk_sample(std::string& prompt, Model* model, Tokenizer* tokenizer, float temp, int top_k, bool showstat)
{
    // TODO: fix topk max situation.
    top_k = std::min(top_k, 1000);
    temp = std::min(temp, 2.0f);
    const int n_predict = std::min(model->m_max_inference_ctx, model->m_max_train_ctx);

    std::random_device rd;
    std::mt19937 gen(rd());

    // model: sample_timers, logits_size

    std::vector<int> tokens = tokenizer->encode(prompt);
    tokens.reserve(n_predict);
    std::vector<std::pair<double, int>> logits_probs;

    const int n_pred_tokens = n_predict - tokens.size();
    int n_iters = 0;
    for (int i = 0; i < n_pred_tokens; i++)
    {
        n_iters += 1;
        gten::Tensor input{tokens.data(), {(int)tokens.size()}, gten::kInt32};
        const int start_pos = (i == 0) ? 0 : input.numel() - 1; 
        gten::Tensor logits = model->logits(input, start_pos);

        Timer sample_timer{&model->m_sample_time_ms};

        const float* logits_data = logits.data_ptr<float>();

        logits_probs.clear();

        const int logits_size = logits.numel();
        for (int j = 0; j < logits_size; ++j) {
            logits_probs.push_back(std::make_pair((double)logits_data[j] / temp, j));
        }
        
        // Select top k elements.
        std::partial_sort(
                logits_probs.begin(),
                logits_probs.begin() + top_k,
                logits_probs.end(),
                [](const std::pair<double, int> &rhs, const std::pair<double, int> &lhs) {
            return rhs.first > lhs.first;
        });
        logits_probs.resize(top_k);
        
        // compute softmax
        double sum_exp = 0;
        for (int j = 0; j < top_k; ++j)
        {
            logits_probs[j].first = std::exp(logits_probs[j].first);
            sum_exp += logits_probs[j].first;
        }
        for (int j = 0; j < top_k; ++j)
            logits_probs[j].first = logits_probs[j].first / sum_exp;

        std::vector<double> probs(logits_size, 0.0);
        for (int j = 0; j < top_k; j++)
        {
            const auto &prob_pair = logits_probs[j];
            probs[prob_pair.second] = prob_pair.first;
        }

        std::discrete_distribution dist(probs.begin(), probs.end());
        int pred_token = dist(gen);
        if (pred_token == tokenizer->m_eos_token) {
            break;
        }

        const int prev_token = (i == 0) ? 1 : tokens.back();
        std::cerr << tokenizer->decode(prev_token, pred_token);

        tokens.push_back(pred_token);
    }

    std::cerr << "\n";

    if (showstat) {
        model->print_perf(n_iters);
    }
}

} // namespace gten

