#pragma once


#include "tensor.h"


namespace gten {
namespace ops {

/// @brief Copies the indexed rows of the source tensor to output tensor.
/// @param src A 2-d tensor to be indexed.
/// @param indices A 1-d tensor of indices with dtype = int.
/// @param out A 2d tensor with enough capacity to fit the indexed rows. Its dtype
///  must be the same as source tensor.
/// @param last_token_only Whether to index the last token only, if others are cached.
void token_embed(const Tensor& weight, const Tensor& tokens, Tensor& out, const int start_pos = 0);

void matmul_2d(const Tensor& x, const Tensor& w, Tensor& out, const int start_pos=0);

void bias_add_inplace(Tensor& inp, const Tensor& bias, const int start_pos=0);

void layer_norm(const Tensor& inp, const Tensor& weight, const Tensor& bias, Tensor& out, const int start_pos = 0);

void silu(const Tensor& inp, Tensor& out, const int start_pos=0);

void silu_inplace(Tensor& inp, const int start_pos=0);

void rotary_emb(Tensor& inp, const int d_head, const float rope_pct, const int start_pos=0);

void rms_norm(const Tensor& inp, const Tensor& weight, Tensor& out, const int start_pos=0);

void multiply(const Tensor& inp0, const Tensor& inp1, Tensor& out, const int start_pos=0);

void multiply_inplace(Tensor& inp0, const Tensor& inp1, const int start_pos=0);

void scale(Tensor& inp, float scaler, const int start_pos=0);

void add(const Tensor& x0, const Tensor& x1, Tensor& out, const int start_pos=0);

void qkv_attn(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& qk, Tensor& qkv, const int max_ctx, const int start_pos=0);

} // namespace ops
} // namespace gten
