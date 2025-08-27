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
#include "log.h"

// for a800
#define MAX_WARP_COUNT 16
#define U_COL_EXRTERN_COUNT 64

#define U_COUNT 8
#define U_LEN_PROC_1TIME (U_COUNT * 32)
#define SYNC_THREAD_NUM (32 / U_COUNT)

// fmt::formatter for cublasStatus_t
template <>
struct fmt::formatter<cublasStatus_t> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(const cublasStatus_t& status, FormatContext& ctx) const {
        // 调用我们放在 util 命名空间中的辅助函数
        return fmt::format_to(ctx.out(), "{}",
                              util::cublasGetErrorString(status));
    }
};

namespace matrix_ops {

/**
 * @brief General matrix multiplication.
 *
 * @tparam T
 * @param handle A handle to the cuBLAS library context.
 * @param m The number of rows of matrix A.
 * @param n The number of columns of matrix B.
 * @param k The number of columns of matrix A.
 * @param alpha The scalar alpha.
 * @param A The m x k matrix A.
 * @param lda The leading dimension of matrix A.
 * @param B The k x n matrix B.
 * @param ldb The leading dimension of matrix B.
 * @param beta The scalar beta.
 * @param C The m x n matrix C.
 * @param ldc The leading dimension of matrix C.
 */
template <typename T>
void gemm(const common::CublasHandle& handle, size_t m, size_t n, size_t k,
          T alpha, thrust::device_ptr<T> A, size_t lda, thrust::device_ptr<T> B,
          size_t ldb, T beta, thrust::device_ptr<T> C, size_t ldc);

/**
 * @brief General matrix multiplication.
 *
 * @tparam T
 * @param handle A handle to the cuBLAS library context.
 * @param m The number of rows of matrix A.
 * @param n The number of columns of matrix B.
 * @param k The number of columns of matrix A.
 * @param alpha The scalar alpha.
 * @param A The m x k matrix A.
 * @param lda The leading dimension of matrix A.
 * @param transA Whether to transpose matrix A.
 * @param B The k x n matrix B.
 * @param ldb The leading dimension of matrix B.
 * @param transB Whether to transpose matrix B.
 * @param beta The scalar beta.
 * @param C The m x n matrix C.
 * @param ldc The leading dimension of matrix C.
 */
template <typename T>
void gemm(const common::CublasHandle& handle, size_t m, size_t n, size_t k,
          T alpha, thrust::device_ptr<T> A, size_t lda, bool transA,
          thrust::device_ptr<T> B, size_t ldb, bool transB, T beta,
          thrust::device_ptr<T> C, size_t ldc);

/**
 * @brief copy matrix from src to dst
 *
 * @tparam srcPtr source matrix ptr type
 * @tparam dstPtr destination matrix ptr type
 * @param src source matrix ptr
 * @param src_ld source matrix leading dimension
 * @param dst destination matrix ptr
 * @param dst_ld destination matrix leading dimension
 * @param m number of rows
 * @param n number of columns
 */
template <typename srcPtr, typename dstPtr, typename T>
void matrix_copy(srcPtr src, size_t src_ld, dstPtr dst, size_t dst_ld, size_t m,
                 size_t n);

/**
 * @brief print matrix for row = m, col = n, and the input is host ptr (column
 * major and lda provided)
 *
 * @tparam T type of the matrix elements
 * @param data host ptr
 * @param m number of rows
 * @param n number of columns
 * @param lda leading dimension of the matrix
 * @param title title of the matrix
 */
template <typename T>
void print(T* data, size_t m, size_t n, size_t lda, const std::string& title);

/**
 * @brief print matrix for row = m, col = n, and the input is host ptr
 *
 * @tparam T
 * @param data host ptr
 * @param m number of rows
 * @param n number of columns
 * @param title title of the matrix
 */
template <typename T>
void print(T* data, size_t m, size_t n, const std::string& title);

/**
 * @brief print matrix for row = col = n (column major)
 *
 * @tparam T
 * @param d_vec device vector
 * @param n number of rows
 * @param title title of the matrix
 */
template <typename T>
void print(thrust::device_vector<T>& d_vec, size_t n, const std::string& title);

/**
 * @brief print matrix for row = m, col = n (column major)
 *
 * @tparam T
 * @param d_vec device vector
 * @param m number of rows
 * @param n number of columns
 * @param title title of the matrix
 */
template <typename T>
void print(thrust::device_vector<T>& d_vec, size_t m, size_t n,
           const std::string& title);

/**
 * @brief print matrix for row = m, col = n, and the input is device ptr (column
 * major)
 *
 * @tparam T
 * @param data device ptr
 * @param m number of rows
 * @param n number of columns
 * @param title title of the matrix
 */
template <typename T>
void print(thrust::device_ptr<T> data, size_t m, size_t n,
           const std::string& title);

/**
 * @brief print matrix for row = m, col = n, and the input is device ptr (column
 * major and lda provided)
 *
 * @tparam T
 * @param data
 * @param m
 * @param n
 * @param lda
 * @param title
 */
template <typename T>
void print(thrust::device_ptr<T> data, size_t m, size_t n, size_t lda,
           const std::string& title);

/**
 * @brief print matrix for row = m, col = n, and the input is device ptr (column
 * major and lda provided)
 *
 * @tparam T
 * @param h_vec
 * @param m
 * @param n
 * @param lda
 * @param title
 */
template <typename T>
void print(thrust::device_vector<T> h_vec, size_t m, size_t n, size_t lda,
           const std::string& title);

template <typename T>
thrust::device_vector<T> create_symmetric_random(size_t n,
                                                 bool fixed_seed = false);

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

/**
 * @brief In-place Tall-and-Skinny QR decomposition on a single GPU.
 *
 * @tparam T
 * @param handle A handle to the cuBLAS library context.
 * @param m The number of rows of matrix A.
 * @param n The number of columns of matrix A.
 * @param A_inout On input, the m x n matrix A. On output, the m x n orthogonal
 * matrix Q.
 * @param R On output, the n x n upper triangular matrix R.
 * @param lda The leading dimension of matrix A.
 * @param ldr The leading dimension of matrix R.
 */
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
 * @param nb The first size of the block.
 * @param b The second size of the block.
 */
template <typename T>
void sy2sb(const common::CublasHandle& handle, size_t n,
           thrust::device_ptr<T> A_inout, size_t lda,
           thrust::device_ptr<T> Y_inout, size_t ldy,
           thrust::device_ptr<T> W_inout, size_t ldw, size_t nb = 128,
           size_t b = 32);

/**
 * @brief Compute C = alpha * A * B^T + alpha * B * A^T + beta * C
 *
 * @tparam T
 * @param handle A handle to the cuBLAS library context.
 * @param n The number of rows of matrix A.
 * @param k The number of columns of matrix A.
 * @param alpha The scalar alpha.
 * @param A The n x k matrix A.
 * @param lda The leading dimension of matrix A.
 * @param B The n x k matrix B.
 * @param ldb The leading dimension of matrix B.
 * @param beta The scalar beta.
 * @param C The n x n matrix C.
 * @param ldc The leading dimension of matrix C.
 */
template <typename T>
void syr2k(const common::CublasHandle& handle, size_t n, size_t k, T alpha,
           thrust::device_ptr<T> A, size_t lda, thrust::device_ptr<T> B,
           size_t ldb, T beta, thrust::device_ptr<T> C, size_t ldc);

namespace sb2tr {

// 函数说明

template <typename T>
void sb2tr(int n, int b, int ns, T* dSubA, int ldSubA, T* dU, int ldU,
           int PEIndex, int PENum, int* com, int* prePEWriteCom,
           int* nextPEWriteTailSweepProcRow);

template <typename T>
__global__ void kernel_bugle_chasing_cpydA2dSubA(int n, int b, int cols_perPE,
                                                 int rank, T* dA, long ldA,
                                                 T* dSubA, int ldSubA);

}  // namespace sb2tr

namespace tr2sb {
template <typename T>
__global__ void BC_kernel_computerQ_1Col_V8_10(
    int n, int perBlockN, int largeBlockNum, int sweepCount,
    int lastSweepUCount, int sweepIndex, T* dCU, T* dQ, long ldQ);
}  // namespace tr2sb

}  // namespace matrix_ops