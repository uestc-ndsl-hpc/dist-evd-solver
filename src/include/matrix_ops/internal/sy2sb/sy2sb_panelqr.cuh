#pragma once

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>

#include <cstddef>

#include "gpu_handle_wrappers.h"

namespace matrix_ops {
namespace internal {
namespace sy2sb {

/**
 * @brief functor to compute A - I
 */
template <typename T>
struct identity_minus_A_functor_2d {
    T* A_ptr_;
    size_t m_;
    size_t lda_;

    identity_minus_A_functor_2d(thrust::device_ptr<T> A, size_t m, size_t lda)
        : A_ptr_(thrust::raw_pointer_cast(A)), m_(m), lda_(lda) {}

    __device__ void operator()(const size_t& k) const {
        size_t row = k % m_;
        size_t col = k / m_;

        size_t physical_index = col * lda_ + row;

        if (row == col) {
            A_ptr_[physical_index] = 1.0 - A_ptr_[physical_index];
        } else {
            A_ptr_[physical_index] = -A_ptr_[physical_index];
        }
    }
};

/**
 * @brief functor for extract Lower
 *
 * @tparam T
 */
template <typename T>
struct extract_L_functor {
    const size_t m;
    const size_t n;
    const size_t lda;

    extract_L_functor(size_t m, size_t n, size_t lda) : m(m), n(n), lda(lda) {}

    __host__ __device__ T operator()(const thrust::tuple<T, size_t>& t) const {
        const auto val = thrust::get<0>(t);
        const auto idx = thrust::get<1>(t);
        const auto col = idx / lda;
        const auto row = idx % lda;

        if (col >= n || row >= m) {
            return val;
        }

        if (row < col) {
            return static_cast<T>(0.0);
        } else if (row == col) {
            return static_cast<T>(1.0);
        } else {
            return val;
        }
    }
};

/**
 * @brief Get the Iminus Q L4panel Q R object
 *
 * @tparam T
 * @param handle The cuSolverDn handle.
 * @param m row size of I - Q
 * @param n column size of I - Q
 * @param A_inout the matrix I - Q
 * @param lda leading dimension of A_inout
 */
template <typename T>
void getIminusQL4panelQR(const common::CusolverDnHandle& handle, size_t m,
                         size_t n, thrust::device_ptr<T> A_inout, size_t lda);

/**
 * @brief Perform QR decomposition of a panel of a symmetric matrix.
 *
 * @tparam T
 * @param cublasHandle
 * @param cusolverDnHandle
 * @param m The number of rows of the panel.
 * @param n The number of columns of the panel.
 * @param A_inout The panel. This is a non-owning pointer.
 * @param lda The leading dimension of the panel.
 * @param R The matrix R. This is a non-owning pointer.
 * @param ldr The leading dimension of the matrix R.
 * @param W The matrix W. This is a non-owning pointer.
 * @param ldw The leading dimension of the matrix W.
 */
template <typename T>
void panelQR(const common::CublasHandle& cublasHandle,
             const common::CusolverDnHandle& cusolverDnHandle, size_t m,
             size_t n, thrust::device_ptr<T> A_inout, size_t lda,
             thrust::device_ptr<T> R, size_t ldr, thrust::device_ptr<T> W,
             size_t ldw);

}  // namespace sy2sb
}  // namespace internal
}  // namespace matrix_ops