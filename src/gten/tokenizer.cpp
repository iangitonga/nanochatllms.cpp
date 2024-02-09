// Modified version of code obtained from the repo: https://github.com/karparthy/llama2.c
// created by the brilliant Andrej Karparthy. [MIT licence].

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctype.h>
#include <fstream>
#include <iostream>
#include <map>
#include <regex>
#include <string>
#include <vector>

#include "abc.h"
#include "tokenizer.h"



namespace gten {
// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

static int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}


LLamaTokenizer::LLamaTokenizer(
    const char* tokenizer_path, int vocab_size, int eos,
    const std::string& prompt_prefix, const std::string& prompt_suffix,
    const std::vector<int>& tokens_prefix, const std::vector<int>& tokens_suffix)
    : Tokenizer(vocab_size, eos, prompt_prefix, prompt_suffix, tokens_prefix, tokens_suffix)
{
    // malloc space to hold the scores and the strings
    vocab_ = (char**)std::malloc(vocab_size * sizeof(char*));
    vocab_scores_ = (float*)std::malloc(vocab_size * sizeof(float));
    sorted_vocab_ = NULL; // initialized lazily
    for (int i = 0; i < 256; i++) {
        byte_pieces_[i * 2] = (unsigned char)i;
        byte_pieces_[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }
    if (fread(&max_token_length_, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "0failed read\n");
        exit(EXIT_FAILURE); }
    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(vocab_scores_ + i, sizeof(float), 1, file) != 1) {
            fprintf(stderr, "1failed read\n");
            exit(EXIT_FAILURE);
        }
        if (fread(&len, sizeof(int), 1, file) != 1) {
            fprintf(stderr, "2failed read\n");
            exit(EXIT_FAILURE);
        }
        vocab_[i] = (char *)malloc(len + 1);
        if (fread(vocab_[i], len, 1, file) != 1) {
            exit(EXIT_FAILURE); }
        vocab_[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

LLamaTokenizer::~LLamaTokenizer() {
    for (int i = 0; i < vocab_size; i++) {
        free(vocab_[i]);
    }
    free(vocab_);
    free(vocab_scores_);
    free(sorted_vocab_);
}


const char* LLamaTokenizer::decode(int prev_token, int token) {
    if (token >= vocab_size) {
        return "";
    }
    char *piece = vocab_[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)byte_pieces_ + byte_val * 2;
    }
    return piece;
}

static int str_lookup(const char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    const TokenIndex tok = { .str = const_cast<char*>(str) }; // acts as the key to search for
    TokenIndex *res = (TokenIndex*)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}


std::vector<int> LLamaTokenizer::encode(std::string& prompt)
{
    prompt = prompt_prefix + prompt + prompt_suffix;
    // prompt format: <|im_start|>user\nPROMPT<|im_end|>\n<|im_start|>assistant\n

    // [<bos>, <|im_start|>]
    // static const int pre_prompt_tokens[] = {1, 32001};
    // [<|im_end|>, \, n, <|im_start|>, assistant\, n]
    // static const int post_prompt_tokens[] = {32002, 29871, 13, 32001, 20255, 13};

    const int num_prompt_tokens = prompt.size() + 1; // +1 for '\0'

    std::vector<int> prompt_tokens;
    prompt_tokens.reserve(num_prompt_tokens);

    // prompt.insert(0, "user\n");
    encode_internal(prompt, prompt_tokens);

    /// TODO: Use a single vector.
    std::vector<int> out_tokens;
    out_tokens.reserve(num_prompt_tokens);

    for (int i = 0; i < int(prompt_prefix_tokens.size()); i++) {
        out_tokens.push_back(prompt_prefix_tokens[i]);
    }

    for (int token : prompt_tokens) {
        out_tokens.push_back(token);
    }

    for (int i = 0; i < int(prompt_suffix_tokens.size()); i++)
    {
        out_tokens.push_back(prompt_suffix_tokens[i]);
    }
    
    return out_tokens;
}

void LLamaTokenizer::encode_internal(const std::string& prompt, std::vector<int>& out_tokens) {
    const char* text = prompt.c_str();
    // encode the string text (input) into an upper-bound preallocated tokens[] array int *tokens, int *n_tokens
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE); }

    if (sorted_vocab_ == NULL) {
        // lazily malloc and sort the vocabulary
        sorted_vocab_ = (TokenIndex*)malloc(vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < vocab_size; i++) {
            sorted_vocab_[i].str = vocab_[i];
            sorted_vocab_[i].id = i;
        }
        qsort(sorted_vocab_, vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    char* str_buffer = (char*)malloc((max_token_length_*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", sorted_vocab_, vocab_size);
        out_tokens.push_back(dummy_prefix);
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (const char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, sorted_vocab_, vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            out_tokens.push_back(id);
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < int(str_len); i++) {
                out_tokens.push_back((unsigned char)str_buffer[i] + 3);
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < int(out_tokens.size()-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", vocab_[out_tokens[i]], vocab_[out_tokens[i+1]]);
            int id = str_lookup(str_buffer, sorted_vocab_, vocab_size);
            if (id != -1 && vocab_scores_[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = vocab_scores_[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        out_tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < int(out_tokens.size()-1); i++) {
            out_tokens[i] = out_tokens[i+1];
        }
        out_tokens.pop_back();
    }

    free(str_buffer);
}


// ------------------------- GPT2 Tokenizer ----------------------------------------------

Gpt2Tokenizer::Gpt2Tokenizer(const std::string vocab_path, const int n_vocab, int eos)
    : Tokenizer(n_vocab, eos, "", "", {}, {})
{
    std::ifstream fin{vocab_path, std::ios_base::binary};
    GTEN_ASSERTM(fin.is_open(), "Failed to open vocab file: %s.", vocab_path.c_str());

    std::string word;
    for (int i = 0; i < n_vocab; i++)
    {
        uint32_t len;
        fin.read((char *) &len, sizeof(len));

        word.resize(len);
        fin.read((char *) word.data(), len);

        token_to_id_[word] = i;
        id_to_token_[i] = word;
    }
}


// Convert a single token id into text.
const char* Gpt2Tokenizer::decode(int /*prev_token*/, int32_t token_id) {
    return id_to_token_[token_id].c_str();
}

// Convert a string of arbitrary text to a sequence of tokens ids.
std::vector<int32_t> Gpt2Tokenizer::encode(std::string& text) {
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

} // namespace gten
