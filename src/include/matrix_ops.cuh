#pragma once

#include <cublas_v2.h>
#include <curand.h>
#include <curand_kernel.h>
#include <fmt/format.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <time.h>

#include <cstddef>
#include <string>

#include "gpu_handle_wrappers.h"

namespace matrix_ops {

template <typename T>
void print(const thrust::device_vector<T>& d_vec, size_t n,
           const std::string& title);

template <typename T>
void print(const thrust::device_vector<T>& d_vec, size_t m, size_t n,
           const std::string& title);

template <typename T>
void print(thrust::device_ptr<T> data, size_t m, size_t n,
           const std::string& title);

template <typename T>
thrust::device_vector<T> create_symmetric_random(size_t n);

template <typename T>
thrust::device_vector<T> create_uniform_random(size_t n);

template <typename T>
thrust::device_vector<T> create_uniform_random(size_t m, size_t n);

template <typename T>
thrust::device_vector<T> create_normal_random(size_t n, T mean = 0.0,
                                              T stddev = 1.0);

template <typename T>
thrust::device_vector<T> create_normal_random(size_t m, size_t n, T mean = 0.0,
                                              T stddev = 1.0);

/**
 * @brief In-place Tall-and-Skinny QR decomposition on a single GPU.
 *
 * This function computes the QR decomposition of a tall-and-skinny matrix A.
 * The input matrix A is overwritten by the orthogonal matrix Q.
 *
 * @tparam T The data type of the matrix elements (e.g., float, double).
 * @param handle A handle to the cuBLAS library context.
 * @param m The number of rows of matrix A.
 * @param n The number of columns of matrix A.
 * @param A_inout On input, the m x n matrix A. On output, the m x n orthogonal
 * matrix Q.
 * @param R On output, the n x n upper triangular matrix R.
 */
template <typename T>
void tsqr(const common::CublasHandle& handle, size_t m, size_t n,
          thrust::device_ptr<T> A_inout, thrust::device_ptr<T> R);

template <typename T>
void tsqr(const common::CublasHandle& handle, size_t m, size_t n,
          thrust::device_ptr<T> A_inout, thrust::device_ptr<T> R, size_t lda,
          size_t ldr);

/**
 * @brief Convert a symmetric matrix to a symmetric banded matrix.
 *
 * This function converts a symmetric matrix to a symmetric banded matrix.
 * The input matrix A is overwritten by the symmetric banded matrix.
 *
 * @tparam T The data type of the matrix elements (e.g., float, double).
 * @param handle A handle to the cuBLAS library context.
 * @param n The number of columns of matrix A.
 * @param A_inout On input, the n x n symmetric matrix A. On output, the n x n
 * symmetric banded matrix.
 * @param Y_inout On output, the n x n upper triangular matrix Y.
 * @param W_inout On output, the n x n matrix W.
 * @param lda The leading dimension of matrix A.
 * @param ldy The leading dimension of matrix Y.
 * @param ldw The leading dimension of matrix W.
 */
template <typename T>
void sy2sb(const common::CublasHandle& handle, size_t n,
           thrust::device_ptr<T> A_inout, size_t lda,
           thrust::device_ptr<T> Y_inout, size_t ldy,
           thrust::device_ptr<T> W_inout, size_t ldw);

}  // namespace matrix_ops