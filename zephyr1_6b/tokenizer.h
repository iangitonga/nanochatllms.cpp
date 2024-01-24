/**
 * Implementation of GP2 tokenizer based of code obtained from ggml lib (https://github.com/ggerganov/ggml)
 * which is fantastic.
 * */


#pragma once


#include <cstdint>
#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <regex>

#include "gten/log.h"


namespace gten
{

/// @brief A tokenizer that performs words encoding and token id decoding as-per GPT2 vocabulary.
class BPETokenizer
{
public:
    BPETokenizer() {};

    BPETokenizer(const std::string vocab_path, const int n_vocab)
        : n_vocab_{n_vocab}
    {
        std::ifstream fin{vocab_path, std::ios_base::binary};
        GTEN_ASSERTM(fin.is_open(), "Failed to open vocab file: %s.", vocab_path.c_str());

        std::string word;
        for (int i = 0; i < n_vocab_; i++)
        {
            uint32_t len;
            fin.read((char *) &len, sizeof(len));

            word.resize(len);
            fin.read((char *) word.data(), len);

            token_to_id_[word] = i;
            id_to_token_[i] = word;
        }
    }

    BPETokenizer& operator=(gten::BPETokenizer &&rhs)
    {
        if (this != &rhs) {
        token_to_id_ = std::move(rhs.token_to_id_);
        id_to_token_ = std::move(rhs.id_to_token_);
        }
        return *this;
    }

    // Convert a single token id into text.
    const std::string &decode(const int32_t token_id) { return id_to_token_[token_id]; }

    // Convert a string of arbitrary text to a sequence of tokens ids.
    std::vector<int32_t> encode(const std::string &text) const {
        std::vector<std::string> words;

        // first split the text into words
        std::string str = text;
        std::regex re(pat_);
        std::smatch m;

        while (std::regex_search(str, m, re)) {
            for (auto x : m) {
                words.push_back(x);
            }
            str = m.suffix();
        }

        // find the longest tokens that form the words:
        std::vector<int32_t> tokens;
        tokens.reserve(encode_prefix.size());
        // prepend prefix.
        tokens.insert(tokens.end(), encode_prefix.begin(), encode_prefix.end());

        for (const auto & word : words)
        {
            if (word.size() == 0) continue;

            int i = 0;
            int n = word.size();
            while (i < n) {
                int j = n;
                while (j > i)
                {
                    auto it = token_to_id_.find(word.substr(i, j-i));
                    if (it != token_to_id_.end()) {
                        tokens.push_back(it->second);
                        i = j;
                        break;
                    }
                    --j;
                }
                if (i == n)
                    break;
                if (j == i)
                {
                    auto sub = word.substr(i, 1);
                    if (token_to_id_.find(sub) != token_to_id_.end())
                        tokens.push_back(token_to_id_.at(sub));
                    else
                        fprintf(stderr, "%s: unknown token '%s'\n", __func__, sub.data());
                    ++i;
                }
            }
        }

        // append suffix.
        tokens.reserve(tokens.size() + encode_suffix.size());
        tokens.insert(tokens.end(), encode_suffix.begin(), encode_suffix.end());

        return tokens;
    }

private:
    int32_t n_vocab_;
    const std::string pat_ = R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";
    const std::vector<int> encode_prefix = {27, 91, 882, 91, 29, 397}; // <|user|>\n
    const std::vector<int> encode_suffix = {100257, 198, 27, 91, 78191, 91, 397}; // <|endoftext|>\n<|assistant|>\n
    std::map<std::string, int32_t> token_to_id_;
    std::map<int32_t, std::string> id_to_token_;
};

} // namespace gten
