#include <cmath>
#include <memory>

#include "gten_types.h"
#include "log.h"
#include "quants.h"

#if defined(__AVX__)
#include <immintrin.h>
#endif

namespace gten {

namespace ops {

static void q8_quantize_block(const float* inp, Q8Block* out, const int block_size) {
    float absmax = 0;
    for (int j = 0; j < block_size; j++) {
        const float x = inp[j];
        absmax = std::max(absmax, fabsf(x));
    }

    const float delta = absmax / 127.0f;
    out->delta = fp32_to_fp16(delta);

    const float scale = delta ? 1.0f/delta : 0.0f;
    for (int i = 0; i < block_size; i++) {
        out->data[i] = static_cast<Qint8>(roundf(inp[i] * scale));
    }
}


static void q8_dequantize_block(const Q8Block* inp, float* out, const int block_size) {
    const float delta = fp16_to_fp32(inp->delta);
    for (int i = 0; i < block_size; i++) {
        out[i] = inp->data[i] * delta;
    }
}

[[nodiscard]]
Qint8 q8_quantize_single(float x, float delta) {
    const float id = delta ? 1.0f/delta : 0.0f;

    const float x0 = x * id;
    const Qint8 quantized = static_cast<Qint8>(roundf(x0));

    return quantized;
}


[[nodiscard]]
float q8_dequantize_single(Qint8 x, float delta) {
    return x * delta;
}
#include <immintrin.h>
void q4_dequantize_block(const Q4Block* inp, float* out) {
    const int block_size = globs::q4_block_size;

#if defined(__AVX__)
    for (int i = 0; i < block_size/16; i++)
    {
        // Q4 layout: the first 16 quants are in the higher 4-bits of the 16 bytes
        // and the next 16 values are in the lower.

        // Here, we want to map the 16 quants into 32 floats (4 __m128)

        // 16 vals -> 4 128, 2 [256]
        // load a block of 16 4-bit ints (8 bytes) into lower 64 bits.
        const __m128i b00 = _mm_loadu_si64(inp->data + i*8);
        // cvt each of the 8 bytes to 16-bit integer.
        const __m128i b01 = _mm_cvtepu8_epi16(b00); // 8 16-bit quants => 128bits
        const __m128i add_vec = _mm_set1_epi16(-7);
        // extract the first 8 4-bit quants stored in high 4 bits in each of the 8 bytes.
        const __m128i b03 = _mm_add_epi16(_mm_srli_epi16(b01, 4), add_vec); // first 8 values
        // // extract the 8 4-bit quants stored in low 4 bits in each of the 8 bytes. 
        const __m128i and_vec = _mm_set1_epi16(0b0000000000001111);
        const __m128i b04 = _mm_add_epi16(_mm_and_si128(b01, and_vec), add_vec);
        // cvt 4 low b03 to floats.
        const __m128 b05 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(b03));
        // cvt 4 high b03 to floats.
        // note: we shift right by 8 bytes instead of left because the elements
        // in the SSE/AVX registers are stored in reverse format.
        const __m128 b06 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_bsrli_si128(b03, 8))); // v << 64

        // cvt 4 low b03 to floats.
        const __m128 b07 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(b04));
        // cvt 4 high b03 to floats.
        const __m128 b08 = _mm_cvtepi32_ps(_mm_cvtepi16_epi32(_mm_bsrli_si128(b04, 8)));

        const __m128 delta_vec = _mm_set1_ps(fp16_to_fp32(inp->delta));
        const __m128 b09 = _mm_mul_ps(b05, delta_vec);
        const __m128 b10 = _mm_mul_ps(b06, delta_vec);
        const __m128 b11 = _mm_mul_ps(b07, delta_vec);
        const __m128 b12 = _mm_mul_ps(b08, delta_vec);

        _mm_storeu_ps(out + i*8, b09);
        _mm_storeu_ps(out + i*8 + 4, b10);

        _mm_storeu_ps(out + i*8 + 16, b11);
        _mm_storeu_ps(out + i*8 + 16 + 4, b12);
    }
#else
    const float delta = fp16_to_fp32(inp->delta);
    const int half_block_size = block_size / 2;
    for (int i = 0; i < half_block_size; i += 1) {
        const Qint4 packed = inp->data[i];
        const Qint8 high = (packed >> 4) - 7;
        const Qint8 low = (packed & 0b00001111) - 7; 
        out[i] = high * delta;
        out[i+half_block_size] = low * delta;
    }
#endif
}

void q8_quantize_row(const float* inp, Q8Block* out, const int rowsize) {
    const int block_size = globs::q8_block_size;
    const int n_blocks = rowsize / block_size;

    for (int i = 0; i < n_blocks; i++) {
        const float* inp_block_data = inp + i * block_size;
        Q8Block* out_block_data = out + i;

        q8_quantize_block(inp_block_data, out_block_data, globs::q8_block_size);
    }
    
    // Quantize the partial last block (i.e it contains < block_size numbers) if it exists.
    const int remsize = rowsize % block_size;
    if (remsize != 0) {
        const float* last_inp_block_data = inp + n_blocks * block_size;
        Q8Block* last_block = out + n_blocks;
        q8_quantize_block(last_inp_block_data, last_block, remsize);
    }
}

void q8_dequantize_row(const Q8Block* inp, float* out, int rowsize) {
    const int block_size = globs::q8_block_size;
    const int n_blocks = rowsize / block_size;

    for (int i = 0; i < n_blocks; i++) {
        q8_dequantize_block(inp + i, out + i * block_size, globs::q8_block_size);
    }

    // De-quantize the partial last block (i.e it contains < block_size numbers) if it exists.
    const int remsize = rowsize % block_size;
    if (remsize != 0) {
        float* last_out_block_data = out + n_blocks * block_size;
        const Q8Block* last_block = inp + n_blocks;
        q8_dequantize_block(last_block, last_out_block_data, remsize);
    }
}

void q4_dequantize_row(const Q4Block* inp, float* out, int rowsize) {
    const int block_size = globs::q4_block_size;
    GTEN_ASSERT(rowsize % block_size == 0);
    const int n_blocks = rowsize / block_size;

    for (int i = 0; i < n_blocks; i++) {
        q4_dequantize_block(inp + i, out + i * block_size);
    }
}

void q8_dequantize_row_delta(const Qint8* x, float* out, float delta, int size) {
    for (int i = 0; i < size; i++) {
        const Qint8 x_val = x[i];
        out[i] = x_val * delta;
    }
}

} // namespace ops

} // namespace gten