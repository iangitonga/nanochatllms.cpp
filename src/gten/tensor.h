#pragma once

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#include "gten_types.h"
#include "log.h"
#include "quants.h"



namespace gten {

class Tensor {
public:
    static int64_t s_tensor_alloc_bytes;
public:
    Tensor() = default;
    Tensor(const std::vector<int>& shape, Dtype dtype);
    Tensor(const void* data_ptr, const std::vector<int>& shape, Dtype dtype);
    Tensor(const Tensor& rhs) = default;
    Tensor(Tensor&& rhs) = default;
    Tensor& operator=(const Tensor& rhs) = default;
    Tensor& operator=(Tensor&& rhs) = default;
    friend std::ostream& operator<<(std::ostream& stream, const Tensor& tensor);
    Tensor permute(const std::vector<int>& new_shape);
    void print() const;
    void print_info() const;
    // Resize the tensor to have a new shape. This function does not perform
    // any reallocation and therefore, the tensor must have enough capacity
    // to accommodate the number of elements in the new shape.
    // NOTE: The purpose of this function is to allow us to allocate for
    // activations tensors to be able to hold all future predictions
    // activations but reshape them as we continously add activations.
    void resize(const std::vector<int>& new_shape);
    void set_strides(const std::vector<int>& strides);
    std::string shape_str() const;
    std::string strides_str() const;
    void save(const std::string& path) const;
    Tensor view(const std::vector<int>& new_shape) const;

    // Get the pointer to internal data buffer.
    template <typename T>
    T* data_ptr() { return reinterpret_cast<T*>(m_data_ptr.get()); }

    template <typename T>
    const T* data_ptr() const { return reinterpret_cast<const T*>(m_data_ptr.get()); }

    const void* data_ptr() const { return m_data_ptr.get(); }
    void* data_ptr() { return m_data_ptr.get(); }

    Dtype dtype() const { return M_dtype; }

    // Get the number of bytes that an element in the tensor occupies.
    int itemsize() const {
        switch (M_dtype) {
            case kQint8:
                return 1;
            case kInt32:
                return 4;
            case kFloat16:
                return 2;
            case kFloat32:
                return 4;
            default:
                GTEN_ASSERT(false);
                return 4;
        }
    }

    bool is_quantized() const  { return M_dtype == kQint8; }
    bool is_1d() const { return m_shape.size() == 1; }
    bool is_2d() const { return m_shape.size() == 2; }
    bool is_3d() const { return m_shape.size() == 3; }
    int ndims() const { return m_shape.size(); }

    // Get the number of elems in the tensor.
    int numel() const { return m_numel; }

    /// Returns the size of the give dimension.
    int dimsize(int i) const {
        GTEN_ASSERT(i < int(m_shape.size()));
        return m_shape[i];
    }

    /// Returns the size of the give dimension.
    int stride(int i) const {
        GTEN_ASSERT(i < int(m_strides.size()));
        return m_strides[i];
    }

    /// Returns the size of the give dimension in bytes.
    int bstride(int i) const {
        GTEN_ASSERT(i < int(m_strides.size()));

        switch (M_dtype)
        {
            case kQint4: {
                if (m_strides[i] == 1) {
                    return 1;
                }
                return (m_strides[i]/globs::q4_block_size) * sizeof(Q4Block);
            }
            case kQint8: {
                if (m_strides[i] == 1) {
                    return 1;
                }
                return (m_strides[i]/globs::q8_block_size) * sizeof(Q8Block);
            }
            default:
                return m_strides[i] * itemsize();
        }
    }

    size_t nbytes() const { return m_storage_size; }

    const std::vector<int>& shape() const { return m_shape; }

    bool shape_eq(const std::vector<int>& shape) const { return shape == m_shape; }

private:
    Dtype M_dtype = kInt32;
    std::shared_ptr<uint8_t> m_data_ptr;
    int m_storage_size = 0;  // in_bytes
    int m_numel = 0;
    std::vector<int> m_shape;
    std::vector<int> m_strides;

private:
    void validate_shape(const std::vector<int>& shape) const;
    void set_strides_from_shape(const std::vector<int>& shape);
    int numel_from_shape(const std::vector<int>& shape) const;
    void print_single(int item_idx, int row_idx, int col_idx, int n_cols) const;
};

} // Namespace xten
