#pragma once

#include <algorithm>
#include <random>

#include "abc.h"
#include "utils.h"


namespace gten {

void greedy_sample(std::string& prompt, Model* model, Tokenizer* tokenizer, bool showstat);

void topk_sample(std::string& prompt, Model* model, Tokenizer* tokenizer, float temp, int top_k, bool showstat);

} // namespace gten

