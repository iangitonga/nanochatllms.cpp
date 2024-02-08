 // <|user|>
// Which famous math number begins with 1.6 ...?<|endoftext|>
// <|assistant|>

#include <gten/gten.h>
#include <random>
#include <algorithm>

#include "tokenizer.h"


using namespace gten;


struct Zephyr1_6bParams {
    const int n_vocab = 100352;
    const int max_ctx = 4096;
    const int n_embd = 2048;
    const int n_ffn = 5632;
    const int n_layers = 24;
    const int n_heads = 32;
    const int n_query_groups = 32;
    const float rope_pct = 0.25f;
    const int eot_token = 100257;
};


class Zephyr1_6b {
public:
    const Zephyr1_6bParams params = Zephyr1_6bParams{};
    ModuleDtype dtype_;
    int n_ctx_;

public:
    Zephyr1_6b(const int n_ctx, ModuleDtype dtype)
        : n_ctx_{n_ctx},
          dtype_{dtype},
          tok_emb_{Embedding(params.n_vocab, params.n_embd, n_ctx, dtype)},
          norm_{LayerNorm(params.n_embd, n_ctx, {kFloat16, dtype.adtype})},
          lm_head_{EmbeddingLinear{params.n_embd, params.n_vocab, n_ctx, {dtype.wdtype, kFloat32}}}
    {
        blocks_.reserve(params.n_layers);
        for (int i = 0; i < params.n_layers; i++) {
            blocks_.push_back(
                AttentionBlock2(params.n_heads, params.n_embd, params.n_query_groups, params.n_ffn, n_ctx, dtype, params.rope_pct, /*qkv_bias=*/true)
            );
        }
    }

    Tensor logits(const Tensor& tokens, const int start_pos=0) {
        if (tokens.numel() > n_ctx_) {
            std::cerr << "Number of prompt tokens (" << tokens.numel() << ") exceed provided maximum ctx size (" << n_ctx_ << ")\n";
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

    void load_from_ckpt(std::ifstream& ckpt);
    void print_perf(const int n_pred_tokens);

private:
    Embedding tok_emb_;
    LayerNorm norm_;
    EmbeddingLinear lm_head_;
    std::vector<AttentionBlock2> blocks_;

public:
    int64_t load_time = 0;
    int64_t sample_time = 0;
};


static inline void read_into_weight(
    std::ifstream& fin, gten::Tensor& tensor, ModuleDtype dtype)
{
    std::string weight_name;
    int32_t weight_name_size;
    fin.read(reinterpret_cast<char*>(&weight_name_size), sizeof(weight_name_size));
    weight_name.resize(weight_name_size);
    fin.read(reinterpret_cast<char*>(weight_name.data()), weight_name_size);

    int32_t weight_payload_size;
    fin.read(reinterpret_cast<char*>(&weight_payload_size), sizeof(weight_payload_size));


    GTEN_ASSERTM(
        static_cast<size_t>(weight_payload_size) == tensor.nbytes(),
        "Weight `%s` data size: %d does not match the expected size: %ld.",
        weight_name.c_str(), weight_payload_size, tensor.nbytes());
    fin.read(tensor.data_ptr<char>(), weight_payload_size);
}


static inline void read_layer_header(std::ifstream& fin, bool debug = false) {
    std::string layer_name;
    int32_t layer_name_size;
    fin.read(reinterpret_cast<char*>(&layer_name_size), sizeof(layer_name_size));
    layer_name.resize(layer_name_size);
    fin.read(reinterpret_cast<char*>(layer_name.data()), layer_name_size);

    if (debug) {
        std::cout << "Layer: " << layer_name << "\n";
    }
}

void Zephyr1_6b::load_from_ckpt(std::ifstream &ckpt)
{
    Timer load_timer{&load_time};

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

        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn.query.bias, {kFloat16, dtype_.adtype});

        // k_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn.key.weight, dtype_);

        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn.key.bias, {kFloat16, dtype_.adtype});

        // v_proj
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn.value.weight, dtype_);

        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn.value.bias, {kFloat16, dtype_.adtype});

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
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.attn_norm.bias, {kFloat16, dtype_.adtype});

        // ffn_norm
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.ffn_norm.weight, {kFloat16, dtype_.adtype});
        read_layer_header(ckpt);
        read_into_weight(ckpt, block.ffn_norm.bias, {kFloat16, dtype_.adtype});
    }
    
    read_layer_header(ckpt);
    read_into_weight(ckpt, norm_.weight, {kFloat16, dtype_.adtype});
    read_layer_header(ckpt);
    read_into_weight(ckpt, norm_.bias, {kFloat16, dtype_.adtype});

    read_layer_header(ckpt);
    read_into_weight(ckpt, lm_head_.weight, dtype_);
}

void Zephyr1_6b::print_perf(const int n_pred_tokens)
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

        non_linear_time = norm_time + res_time + rope_time + activ_time + mul_time;
    }
    const int64_t tot_inf_time = linear_time + attn_time + non_linear_time;

    const int64_t tensor_mem = G_TensorMemAllocated;
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


void greedy_sample(const std::string& prompt, Zephyr1_6b& model, BPETokenizer& tokenizer, bool showstat)
{
    const int n_predict = model.n_ctx_;
    std::vector<int> tokens = tokenizer.encode(prompt);
    tokens.reserve(n_predict);

    const int max_iters = n_predict - tokens.size();
    int n_iters = 0;
    for (int i = 0; i < max_iters; i++)
    {
        n_iters += 1;

        Tensor input{tokens.data(), {(int)tokens.size()}, kInt32};

        const int start_pos = (i == 0) ? 0 : input.numel() - 1; 
        Tensor logits = model.logits(input, start_pos);

        Timer sample_timer{&model.sample_time};

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

        if (pred_token == model.params.eot_token) {
            break;
        }

        std::cerr << tokenizer.decode(pred_token);

        tokens.push_back(pred_token);
    }
    
    std::cerr << '\n';

    if (showstat) {
        model.print_perf(n_iters);
    }
}

void topk_sample(const std::string& prompt, Zephyr1_6b& model, BPETokenizer& tokenizer, const float temp, const int top_k, bool showstat)
{
    const int n_predict = model.n_ctx_;

    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<int> tokens = tokenizer.encode(prompt);
    tokens.reserve(n_predict);
    const int logits_size = model.params.n_vocab;
    std::vector<std::pair<double, int>> logits_probs;
    logits_probs.reserve(logits_size);

    const int n_pred_tokens = n_predict - tokens.size();
    int n_iters = 0;
    for (int i = 0; i < n_pred_tokens; i++)
    {
        n_iters += 1;
        gten::Tensor input{tokens.data(), {(int)tokens.size()}, gten::kInt32};
        const int start_pos = (i == 0) ? 0 : input.numel() - 1; 
        gten::Tensor logits = model.logits(input, start_pos);

        Timer sample_timer{&model.sample_time};

        const float* logits_data = logits.data_ptr<float>();

        logits_probs.clear();
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
        uint32_t pred_token = dist(gen);
        if (pred_token == model.params.eot_token) {
            break;
        }

        std::cerr << tokenizer.decode(pred_token);

        tokens.push_back(pred_token);
    }

    std::cerr << "\n";

    if (showstat) {
        model.print_perf(n_iters);
    }
}


static const char *usage_message = R"(
USAGE:
./zephyr [options] -p PROMPT  for a single prompt or
./zephyr [options] for a chat interface. 

Optional args:
-fp16:     Use 16-bit float model (3GB) [default].
-q8  :     Use 8-bit quantized model (1.6GB).
-q4  :     Use 4-bit quantized model (0.9GB).
-showstat : Show inference performance stats.
--temp T : Temperature to use during sampling. It must be greater than 0. [default=0.9].
--npred  N : Number of tokens to generate. Minimum is 1 and max is 4096. [default=768].
--topk K : Top tokens to randomly select from during prediction. [default=40].

Examples:
  ./zephyr
  ./zephyr -q8 -p "Give three tips for staying healthier."

)";

int main(int argc, char const *argv[])
{
    Dtype model_dtype = kFloat16;
    std::string model_path = "models/zephyr1_6b.fp16.gten";
    std::string prompt = "";
    int n_predict = 768;
    bool use_greedy_sampler = false;
    float sampling_temp = 0.9f;
    int topk = 50;
    bool showstat = false;

    for (int i = 1; i < argc; i++)
    {
        std::string_view arg{argv[i]};
        if (arg == "--help" || arg == "-h") {
            std::cout << usage_message << "\n";
            return 0;
        }
        else if (arg == "-fp16") {
            continue;
        }
        else if (arg == "-q8") {
            model_dtype = kQint8;
            model_path = "models/zephyr1_6b.q8.gten";
        }
        else if (arg == "-q4") {
            model_dtype = kQint4;
            model_path = "models/zephyr1_6b.q4.gten";
        }
        else if (arg == "-p") {
            if (i + 1 < argc) {
                prompt = argv[i + 1];
                i += 1; // fast-forward
            } else {
                std::cerr << "error: Prompt not provided.\n" << usage_message << "\n";
                std::exit(EXIT_FAILURE);
            }
        } else if (arg == "-greedy") {
           use_greedy_sampler = true;
        } else if (arg == "-showstat") {
           showstat = true;
        }  else if (arg == "--npred") {
            if (argc <= i+1) {
                std::cerr << "npred value is missing.\n";
                return -1;
            }
            int npred;
            try {
                npred = std::stoi(argv[i+1]);
            } catch (...) {
                std::cerr << "Invalid npred value.\n";
                return -1;
            }
            if (npred < 1 || npred > 4096) {
                std::cerr << "npred must be greater than 1 and less than 4096.\n";
                return -1;
            }
            n_predict = npred;
            i += 1; // skip len param
        } else if (arg == "--temp") {
            if (argc <= i+1) {
                std::cerr << "temp value is missing.\n";
                return -1;
            }
            float arg_temp;
            try {
                arg_temp = std::stof(argv[i+1]);
            } catch (...) {
                std::cerr << "Invalid temp value \n";
                return -1;
            }
            if (arg_temp <= 0.0f) {
                std::cerr << "temp value must be greater than zero.\n";
                return -1;
            }
            sampling_temp = arg_temp;
            i += 1; // skip parsed temp.
        } else if (arg == "--topk") {
            if (argc <= i+1) {
                std::cerr << "topk value is missing.\n";
                return -1;
            }
            int arg_top_k;
            try {
                arg_top_k = std::stoi(argv[i+1]);
            } catch (...) {
                std::cerr << "Invalid topk value.\n";
                return -1;
            }
            const int n_vocab = 100352;
            if (arg_top_k < 1 || arg_top_k > 100352) {
                std::cerr << "topk must be gte 1 and lte " << 100352 << ".\n";
                return -1;
            }
            topk = arg_top_k;
            i += 1;
        }
        else {
            std::cerr << "error: Unknown argument: " << arg << "\n" << usage_message;
            std::exit(EXIT_FAILURE);
        }
    }
    

    std::string model_id;
    if (model_dtype == kFloat16) {
        model_id = "fp16";
    } else if (model_dtype == kQint4) {
        model_id = "q4";
    } else if (model_dtype == kQint8) {
        model_id = "q8";
    } else { GTEN_ASSERT(false); }

#ifdef _WIN32
    int res = std::system(("python model_dl.py " + model_id).c_str());
#else
    int res = std::system(("python3 model_dl.py " + model_id).c_str());
#endif
    if (res != 0) {
        std::cerr << "Error: Failed to download the model. Check your network connectivity.\n";
        return -1;
    }

    std::ifstream checkpoint{model_path, std::ios::binary};
    GTEN_ASSERT(checkpoint.is_open());

    ModuleDtype dtype;
    if (model_dtype == kFloat16) {
        dtype.wdtype = kFloat16;
        dtype.adtype = kFloat16;
    } else {
        dtype.wdtype = model_dtype;
        dtype.adtype = kQint8;
    }

    Zephyr1_6b model{n_predict, dtype};
    model.load_from_ckpt(checkpoint);

    BPETokenizer tokenizer{"zephyr_tok.bin", 100352};
    
    if (prompt == "") {
        std::cout << "Chat interface. Write your prompt and press enter to submit. Enter q or press ctrl+c to quit.\n";
        std::string prompt;
        while (true) {
            std::cerr << "\n\n[You]: ";
            std::getline(std::cin, prompt);
            if (prompt == "q")
                break;

            std::cerr << "\n[Zephyr1.6b]: \n\n";
            if (use_greedy_sampler) {
                greedy_sample(prompt, model, tokenizer, showstat);
            } else {
                topk_sample(prompt, model, tokenizer, sampling_temp, topk, showstat);
            }
        }
    }
    else {
        if (use_greedy_sampler) {
            greedy_sample(prompt, model, tokenizer, showstat);
        } else {
            topk_sample(prompt, model, tokenizer, sampling_temp, topk, showstat);
        }
    }

    return 0;
}

