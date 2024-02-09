#pragma once


#include <cmath>
#include <memory>

#include "gten_types.h"
#include "log.h"


namespace globs {
static const int q8_block_size = 32;
static const int q4_block_size = 32;
}

struct Q8Block
{
    gten::Float16 delta;
    gten::Qint8 data[globs::q8_block_size];
};

static_assert(sizeof(Q8Block) == sizeof(gten::Float16) + globs::q8_block_size, "Incorrect Q8Block alignment.");

struct Q4Block
{
    gten::Float16 delta;
    gten::Qint8 data[globs::q4_block_size / 2];
};

static_assert(sizeof(Q4Block) == sizeof(gten::Float16) + globs::q4_block_size / 2, "Incorrect Q4Block alignment.");


namespace gten {
namespace ops {

[[nodiscard]]
Qint8 q8_quantize_single(float x, float delta);


[[nodiscard]]
float q8_dequantize_single(Qint8 x, float delta);

void q8_quantize_row(const float* inp, Q8Block* out, const int rowsize);


void q8_dequantize_row(const Q8Block* inp, float* out, int rowsize);


void q4_dequantize_row(const Q4Block* inp, float* out, int rowsize);


void q8_dequantize_row_delta(const Qint8* x, float* out, float delta, int size);

} // namespace ops

} // namespace gten