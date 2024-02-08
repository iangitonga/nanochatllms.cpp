#include "gten/gten.h"

#include "tokenizer.h"

#include <string_view>
#include <random>
#include <algorithm>

using namespace gten;


struct MiniCPMConfig {
    const int n_vocab = 122753;
    const int max_ctx = 2048;
    const int n_embd = 2304;
    const int n_ffn = 5760;
    const int n_layers = 40;
    const int n_heads = 36;
    const int n_query_groups = 36;
    const float scale_emb = 12.0f;
    const int dim_model_base = 256;
    const float scale_depth = 1.4f;
};

static const MiniCPMConfig config = MiniCPMConfig{};


static void vec_scale_f32(float* a, const float scalar, int vec_size)
{
    const int unrolled_vec_size = (vec_size / 8) * 8;

    for (int i = 0; i < unrolled_vec_size; i += 8) {
        a[i + 0] = a[i + 0] * scalar;
        a[i + 1] = a[i + 1] * scalar;
        a[i + 2] = a[i + 2] * scalar;
        a[i + 3] = a[i + 3] * scalar;
        a[i + 4] = a[i + 4] * scalar;
        a[i + 5] = a[i + 5] * scalar;
        a[i + 6] = a[i + 6] * scalar;
        a[i + 7] = a[i + 7] * scalar;
    } 

    // leftovers
    for (int i = unrolled_vec_size; i < vec_size; i++) {
        a[i] = a[i] * scalar;
    }
}

static void scale(Tensor& inp, float scaler, const int start_pos)
{
    char* inp_data = inp.data_ptr<char>();

    const int n_ctx = inp.dimsize(0);
    const int n_embd = inp.dimsize(1);
    const int inp_st0 = inp.bstride(0);

    float* inp_buf = ops::g_ops_state.buf(n_embd);

    for (int i = start_pos; i < n_ctx; ++i)
    {
        ops::read_row_to_float(inp_data + i * inp_st0, inp.dtype(), inp_buf, n_embd);

        vec_scale_f32(inp_buf, scaler, n_embd);

        ops::write_row_from_float(inp_buf, inp_data + i * inp_st0, inp.dtype(), n_embd);   
    }
}

void copy_tensor(const Tensor& src, Tensor& dest)
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

class MiniCPMAttentionBlock {
public:
    MiniCPMAttentionBlock(int n_heads, int d_embed, int n_query_groups, int n_mlp, int max_ctx, ModuleDtype dtype);
    Tensor forward(Tensor& inp, Tensor& scratch, const int start_pos);
    Tensor mlp_forward(const Tensor& inp, const int start_pos=0);

public:
    RMSNorm input_norm;
    SelfAttention self_attn;
    Residual inp_residual;
    RMSNorm post_attn_norm;
    Linear ffn_gate_proj;
    Linear ffn_up_proj;
    SiLU ffn_silu;
    Multiply ffn_mul;
    Linear ffn_down_proj;
    Residual attn_res;
};

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
    const float h00_scaler = config.scale_depth / std::sqrt(config.n_layers);
    scale(h00, h00_scaler, start_pos); // inplace

    Tensor h01 = inp_residual.forward(scratch, h00, start_pos);
    copy_tensor(h01, scratch);

    Tensor h02 = mlp_forward(post_attn_norm.forward(h01, start_pos), start_pos);
    scale(h02, h00_scaler, start_pos); // inplace

    Tensor out = attn_res.forward(h02, scratch, start_pos);

    return out;
}


class MiniCPM {
public:
    const MiniCPMConfig params = MiniCPMConfig{};
    ModuleDtype dtype_;
    int n_ctx_;

public:
    MiniCPM(const int n_ctx, ModuleDtype dtype)
        : n_ctx_{n_ctx},
          dtype_{dtype},
          tok_emb_{TiedEmbedding(params.n_vocab, params.n_embd, n_ctx, dtype)},
          norm_{RMSNorm(params.n_embd, n_ctx, {kFloat16, dtype.adtype})},
          res_scratch{Tensor({params.max_ctx, params.n_embd}, dtype.adtype)}
    {
        blocks_.reserve(params.n_layers);
        for (int i = 0; i < params.n_layers; i++) {
            blocks_.push_back(
                MiniCPMAttentionBlock(params.n_heads, params.n_embd, params.n_query_groups, params.n_ffn, n_ctx, dtype)
            );
        }
    }

    Tensor logits(const Tensor& tokens, const int start_pos=0) {
        if (tokens.numel() > n_ctx_) {
            std::cerr << "Number of prompt tokens (" << tokens.numel() << ") exceed provided maximum ctx size (" << n_ctx_ << ")\n";
            std::exit(EXIT_FAILURE);
        }

        Tensor logits = tok_emb_.forward_embed(tokens, start_pos);
        scale(logits, params.scale_emb, start_pos);

        for (auto& block : blocks_) {
            logits = block.forward(logits, res_scratch, start_pos);
        }

        logits = norm_.forward(logits, start_pos);

        const float scaler = 1.0f / (config.n_embd / config.dim_model_base);
        scale(logits, scaler, start_pos);

        logits = tok_emb_.forward_proj(logits);

        return logits;
    }

    void load_from_ckpt(std::ifstream& ckpt);
    void print_perf(const int n_pred_tokens);

private:
    TiedEmbedding tok_emb_;
    RMSNorm norm_;
    std::vector<MiniCPMAttentionBlock> blocks_;
    Tensor res_scratch; // Used to hold a tensor copy for residual ops.

public:
    int64_t load_time = 0;
    int64_t sample_time = 0;
};

void greedy_sample(std::string& prompt, MiniCPM& model, Tokenizer& tokenizer, const int n_predict, bool showstat);
void topk_sample(std::string& prompt, MiniCPM& model, Tokenizer& tokenizer, const int n_predict, const float temp, const int top_k, bool showstat);


static const char *usage_message = R"(
USAGE:
./minicpm [options] -p PROMPT  for a single prompt or
./minicpm [options] for a chat interface. 

Optional args. 
-f16 :     Use float-16 model and inference (5.5GB). [default]
-q8  :     Use 8-bit quantized model (2.9GB).
-q4  :     Use 4-bit quantized model (1.5GB).
-showstat  : Show inference performance stats.
--temp T   : Temperature to use during sampling. It must be greater than 0. [default=0.9].
--npred  N : Number of tokens to generate. Minimum is 1 and max is 2048. [default=512].
--topk K   : Top tokens to randomly select from during prediction. [default=40].

Examples:
  ./minicpm
  ./minicpm -q8 --npred 1000
  ./minicpm -p "Give three tips for staying healthier."

)";


int main(int argc, char const *argv[])
{
    Dtype model_dtype = kFloat16;
    std::string model_path = "models/minicpm.fp16.gten";
    std::string prompt = "";
    int n_predict = 256;
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
        if (arg == "-f16") {
            continue;
        }
        else if (arg == "-q8") {
            model_dtype = kQint8;
            model_path = "models/minicpm.q8.gten";
        }
        else if (arg == "-q4") {
            model_dtype = kQint4;
            model_path = "models/minicpm.q4.gten";
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
        } else if (arg == "--npred") {
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
            if (npred < 1 || npred > 2048) {
                std::cerr << "npred must be greater than 1 and less than 2048.\n";
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
            const int n_vocab = config.n_vocab;
            if (arg_top_k < 1 || arg_top_k > config.n_vocab) {
                std::cerr << "topk must be gte 1 and lte " << config.n_vocab << ".\n";
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

    MiniCPM model{n_predict, dtype};
    model.load_from_ckpt(checkpoint);

    Tokenizer tokenizer{"tokenizer.bin", config.n_vocab};
    
    if (prompt == "") {
        std::cout << "Chat interface. Write your prompt and press enter to submit. Enter q or press ctrl+c to quit.\n";
        std::string prompt;
        while (true) {
            std::cerr << "\n\n[You]: ";
            std::getline(std::cin, prompt);
            if (prompt == "q")
                break;

            std::cerr << "\n[miniCPM-Chat]: \n\n";
            if (use_greedy_sampler) {
                greedy_sample(prompt, model, tokenizer, n_predict, showstat);
            } else {
                topk_sample(prompt, model, tokenizer, n_predict, sampling_temp, topk, showstat);
            }
        }
    }
    else {
        if (use_greedy_sampler) {
            greedy_sample(prompt, model, tokenizer, n_predict, showstat);
        } else {
            topk_sample(prompt, model, tokenizer, n_predict, sampling_temp, topk, showstat);
        }
    }

    return 0;
}


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

    // if (debug)
        // std::cout << weight_name << " (" << weight_payload_size << ")\n";

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

void MiniCPM::load_from_ckpt(std::ifstream &ckpt)
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


void greedy_sample(std::string& prompt, MiniCPM& model, Tokenizer& tokenizer, const int n_predict, bool showstat)
{
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

        const int maxi = max_index;
        if (maxi == tokenizer.eos) {
            // std::cerr << "<EOT>";
            break;
        }
        const int prev_token = (i == 0) ? 1 : tokens.back();
        std::cerr << tokenizer.decode(prev_token, maxi);

        tokens.push_back(maxi);
    }
    
    std::cerr << '\n';

    if (showstat) {
        model.print_perf(n_iters);
    }
}

void topk_sample(std::string& prompt, MiniCPM& model, Tokenizer& tokenizer, const int n_predict, const float temp, const int top_k, bool showstat)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<int> tokens = tokenizer.encode(prompt);
    tokens.reserve(n_predict);
    const int logits_size = model.params.n_vocab;
    std::vector<std::pair<double, int>> logits_probs;
    logits_probs.reserve(logits_size);

    const int eot_token = tokenizer.eos;

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
        uint32_t maxi = dist(gen);
        if (maxi == eot_token) {
            // std::cerr << "<EOT>";
            break;
        }

        const int prev_token = (i == 0) ? 1 : tokens.back();
        std::cerr << tokenizer.decode(prev_token, maxi);

        tokens.push_back(maxi);
    }

    std::cerr << "\n";

    if (showstat) {
        model.print_perf(n_iters);
    }
}


void MiniCPM::print_perf(const int n_pred_tokens)
{
    int64_t linear_time = 0;
    int64_t attn_time = 0;
    int64_t non_linear_time = 0;

    {
        const int64_t emb_time = tok_emb_.emb_exec_time;
        int64_t norm_time = norm_.exec_time;
        int64_t res_time = 0;
        int64_t rope_time = 0;
        int64_t silu_time = 0;
        int64_t mul_time = 0;
        linear_time += tok_emb_.proj_exec_time;

        for (const auto& b : blocks_) {
            norm_time += b.input_norm.exec_time + b.post_attn_norm.exec_time;
            attn_time += b.self_attn.exec_time_attn;
            res_time  += b.attn_res.exec_time + b.inp_residual.exec_time;
            rope_time += b.self_attn.q_rope.exec_time + b.self_attn.k_rope.exec_time;
            silu_time += b.ffn_silu.exec_time;
            mul_time  += b.ffn_mul.exec_time;
            linear_time += b.self_attn.query.exec_time + b.self_attn.key.exec_time + b.self_attn.value.exec_time + b.self_attn.qkv_proj.exec_time;
            linear_time += b.ffn_gate_proj.exec_time + b.ffn_up_proj.exec_time + b.ffn_down_proj.exec_time;
        }

        non_linear_time = norm_time + res_time + rope_time + silu_time + mul_time;
    }
    const int64_t tot_inf_time = linear_time + attn_time + non_linear_time;

    const int64_t tensor_mem = G_TensorMemAllocated;
    int64_t weights_mem = 0;

    {
        const auto bytes = [](const Tensor& t) { return t.nbytes(); };

        weights_mem += bytes(tok_emb_.weight);
        weights_mem += bytes(norm_.weight);

        for (const auto& b : blocks_) {
            weights_mem += bytes(b.input_norm.weight) + bytes(b.post_attn_norm.weight);
            weights_mem += bytes(b.self_attn.query.weight) + bytes(b.self_attn.key.weight) + bytes(b.self_attn.value.weight) + bytes(b.self_attn.qkv_proj.weight);
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
