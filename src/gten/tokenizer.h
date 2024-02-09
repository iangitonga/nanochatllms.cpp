#pragma once

// Modified version of code obtained from the repo: https://github.com/karparthy/llama2.c
// created by the brilliant Andrej Karparthy. [MIT licence].


#include <map>
#include <regex>
#include <string>
#include <vector>

#include "abc.h"



namespace gten {
// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

struct TokenIndex{
    char *str;
    int id;
};


class LLamaTokenizer : public Tokenizer {
public:
    LLamaTokenizer(const char* path, int vocab_size, int eos);
    ~LLamaTokenizer();
    std::vector<int> encode(std::string& prompt);
    const char* decode(int prev_token, int token);

private:
    void encode_internal(const std::string& prompt, std::vector<int>& out_tokens);

private:
    char** vocab_;
    float* vocab_scores_;
    TokenIndex *sorted_vocab_;
    unsigned int max_token_length_;
    unsigned char byte_pieces_[512]; // stores all single-byte strings
};


// ------------------------- GPT2 Tokenizer ----------------------------------------------
/// @brief A tokenizer that performs words encoding and token id decoding as-per GPT2 vocabulary.
class Gpt2Tokenizer : public Tokenizer {
public:
    Gpt2Tokenizer(const std::string vocab_path, const int n_vocab, int eos);

    // Convert a single token id into text.
    const char* decode(int /*prev_token*/, int32_t token_id);

    // Convert a string of arbitrary text to a sequence of tokens ids.
    std::vector<int32_t> encode(std::string& text);

private:
    const std::string pat_ = R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";
    const std::vector<int> encode_prefix = {27, 91, 882, 91, 29, 397}; // <|user|>\n
    const std::vector<int> encode_suffix = {100257, 198, 27, 91, 78191, 91, 397}; // <|endoftext|>\n<|assistant|>\n
    std::map<std::string, int32_t> token_to_id_;
    std::map<int32_t, std::string> id_to_token_;
};

} // namespace gten

