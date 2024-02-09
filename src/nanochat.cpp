#include "gten/gten.h"
#include "tinyllama/tinyllama.h"
#include "zephyr/zephyr.h"
#include "minicpm/minicpm.h"


using namespace gten;



static const char *usage_message = R"(
USAGE:
./nanochat [options] -m MODEL_NAME -p PROMPT  for a single prompt or
./nanochat [options] -m MODEL_NAME for a chat interface. 

Available MODEL_NAME options are [tinyllama, zephyr, minicpm].

Optional args. 
-f16 :     Use float-16 model and inference. [default]
-q8  :     Use 8-bit quantized model.
-q4  :     Use 4-bit quantized model.
-showstat  : Show inference performance stats.
--temp T   : Temperature to use during sampling. It must be greater than 0.
--npred  N : Max number of tokens to generate. Minimum is 1.
--topk K   : Top tokens to randomly select from during prediction. Minimum is 1.
)";


// build/bin/
// build/models/
// assets/tokenizers/tinyllama_tok.bin

int main(int argc, char const *argv[])
{

    Dtype model_dtype = kFloat16;
    std::string model_name = "";
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
        if (arg == "-f16") {
            continue;
        }
        else if (arg == "-q8") {
            model_dtype = kQint8;
        }
        else if (arg == "-q4") {
            model_dtype = kQint4;
        }
        else if (arg == "-m") {
            if (i + 1 < argc) {
                model_name = argv[i + 1];
                if (model_name == "tinyllama" || model_name == "zephyr" || model_name == "minicpm") {
                    i += 1; // fast-forward
                } else {
                    std::cerr << "error: Unknown model name: " << model_name << ".\n" << usage_message << "\n";
                    std::exit(EXIT_FAILURE);
                }
            } else {
                std::cerr << "error: Model name not provided.\n" << usage_message << "\n";
                std::exit(EXIT_FAILURE);
            }
        }
        else if (arg == "-p") {
            if (i + 1 < argc) {
                prompt = argv[i + 1];
                i += 1; // fast-forward
            } else {
                std::cerr << "error: Prompt not provided.\n" << usage_message << "\n";
                std::exit(EXIT_FAILURE);
            }
        }
        else if (arg == "-greedy") {
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
            if (npred < 1) {
                std::cerr << "npred must be greater than 1.\n";
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
            if (arg_top_k < 1) {
                std::cerr << "topk must be gte 1.\n";
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

    if (model_name == "") {
        std::cerr << "error: Model name not provided.\n" << usage_message << "\n";
        std::exit(EXIT_FAILURE);
    }

    std::string model_dtype_str;
    if (model_dtype == kFloat16) {
        model_dtype_str = "fp16";
    } else if (model_dtype == kQint4) {
        model_dtype_str = "q4";
    } else if (model_dtype == kQint8) {
        model_dtype_str = "q8";
    } else { GTEN_ASSERT(false); }

#ifdef _WIN32
    int res = std::system(("python model_dl.py " + model_name + " " + model_dtype_str).c_str());
#else
    int res = std::system(("python3 model_dl.py " + model_name + " " + model_dtype_str).c_str());
#endif
    if (res != 0) {
        std::cerr << "Error: Failed to download the model. Check your network connectivity.\n";
        return -1;
    }

    // models/yinylama.fp16.gten
    const std::string model_path = "./models/" + model_name + "." + model_dtype_str + ".gten";

    std::ifstream checkpoint{model_path, std::ios::binary};
    GTEN_ASSERT(checkpoint.is_open());

    ModuleDtype dtype;
    if (model_dtype == kQint4)      { dtype = mQint4; }
    else if (model_dtype == kQint8) { dtype = mQint8; }
    else                            { dtype = mFloat16; }


    Model* model_ptr;
    Tokenizer* tokenizer;

    if (model_name == "tinyllama")
    {
        model_ptr = new TinyLLama{n_predict, dtype};
        const int vocab_size = tinyllama_cfg.n_vocab - 3;
        tokenizer = new LLamaTokenizer{"./assets/tokenizers/tinyllama_tokenizer.bin", vocab_size, tinyllama_cfg.eos};
    }
    else if (model_name == "zephyr")
    {
        model_ptr = new Zephyr{n_predict, dtype};
        tokenizer = new Gpt2Tokenizer{"./assets/tokenizers/zephyr_tokenizer.bin", zephyr_cfg.n_vocab, zephyr_cfg.eos};
    }
    else if (model_name == "minicpm")
    {
        model_ptr = new MiniCPM{n_predict, dtype};
        tokenizer = new LLamaTokenizer{"./assets/tokenizers/minicpm_tokenizer.bin", minicpm_cfg.n_vocab, minicpm_cfg.eos};
    }
    else
    {
        GTEN_ASSERT(false);
    }

    model_ptr->load_from_ckpt(checkpoint);

    if (prompt == "") {
        std::cout << "Chat interface. Write your prompt and press enter to submit. Enter q or press ctrl+c to quit.\n";
        std::string prompt;
        while (true) {
            std::cerr << "\n\n[You]: ";
            std::getline(std::cin, prompt);
            if (prompt == "q")
                break;

            std::cerr << "\n[" + model_name + "-Chat]: \n\n";
            if (use_greedy_sampler) {
                greedy_sample(prompt, model_ptr, tokenizer, showstat);
            } else {
                topk_sample(prompt, model_ptr, tokenizer, sampling_temp, topk, showstat);
            }
        }
    }
    else {
        if (use_greedy_sampler) {
            greedy_sample(prompt, model_ptr, tokenizer, showstat);
        } else {
            topk_sample(prompt, model_ptr, tokenizer, sampling_temp, topk, showstat);
        }
    }

    return 0;
}

