#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>

#include <cstddef>
#include <type_traits>

#include "gpu_handle_wrappers.h"
#include "matrix_ops.cuh"

namespace matrix_ops {

namespace internal {
namespace sy2sb {

/**
 * @brief functor for identity minus A
 *
 * @tparam T
 */
template <typename T>
struct identity_minus_A_functor {
    const size_t m;
    const size_t n;
    const size_t lda;

    identity_minus_A_functor(size_t m, size_t n, size_t lda)
        : m(m), n(n), lda(lda) {}

    __host__ __device__ T operator()(const thrust::tuple<T, size_t>& t) const {
        const auto val = thrust::get<0>(t);
        const auto idx = thrust::get<1>(t);
        const auto col = idx / lda;
        const auto row = idx % lda;

        if (col >= n || row >= m) {
            return val;
        }

        if (row == col) {
            return static_cast<T>(1.0) - val;
        } else {
            return -val;
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
                         size_t n, thrust::device_ptr<T> A_inout, size_t lda) {
    auto lwork = (size_t)0;
    if constexpr (std::is_same_v<T, double>) {
        cusolverDnDgetrf_bufferSize(
            handle, m, n, thrust::raw_pointer_cast(A_inout), lda, &lwork);
    } else if constexpr (std::is_same_v<T, float>) {
        cusolverDnSgetrf_bufferSize(
            handle, m, n, thrust::raw_pointer_cast(A_inout), lda, &lwork);
    } else {
        throw std::runtime_error("Unsupported type.");
    }
    auto work = thrust::device_vector<T>(lwork);
    auto info = thrust::device_vector<int>(1);
    // excute LU factorization inplace
    if constexpr (std::is_same_v<T, double>) {
        cusolverDnDgetrf(handle, m, n, thrust::raw_pointer_cast(A_inout), lda,
                         thrust::raw_pointer_cast(work.data()), NULL,
                         thrust::raw_pointer_cast(info.data()));
    } else if constexpr (std::is_same_v<T, float>) {
        cusolverDnSgetrf(handle, m, n, thrust::raw_pointer_cast(A_inout), lda,
                         thrust::raw_pointer_cast(work.data()), NULL,
                         thrust::raw_pointer_cast(info.data()));
    } else {
        throw std::runtime_error("Unsupported type.");
    }
    if (info[0] != 0) {
        throw std::runtime_error("Failed to factorize the matrix.");
    }
    // extract the lower triangular part of the matrix
    thrust::transform(
        thrust::device,
        thrust::make_zip_iterator(
            thrust::make_tuple(A_inout, thrust::counting_iterator<size_t>(0))),
        thrust::make_zip_iterator(thrust::make_tuple(
            A_inout + n * lda, thrust::counting_iterator<size_t>(n * lda))),
        A_inout, extract_L_functor<T>(m, n, lda));
}

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
             size_t ldw) {
    // tsqr, A_inout <- Q R <- R
    matrix_ops::tsqr(cublasHandle, m, n, A_inout, R, lda, ldr);

    // A <- I - A
    thrust::transform(
        thrust::device,
        thrust::make_zip_iterator(
            thrust::make_tuple(A_inout, thrust::counting_iterator<size_t>(0))),
        thrust::make_zip_iterator(thrust::make_tuple(
            A_inout + n * lda, thrust::counting_iterator<size_t>(n * lda))),
        A_inout, identity_minus_A_functor<T>(m, n, lda));

    // W = A_inout (a.k.a. I-Q)
    thrust::copy(A_inout, A_inout + m * n, W);

    getIminusQL4panelQR(cusolverDnHandle, m, n, A_inout, lda);
    const auto alpha = static_cast<T>(1.0);
    if constexpr (std::is_same_v<T, double>) {
        cublasDtrsm(cublasHandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, m, n, &alpha,
                    thrust::raw_pointer_cast(A_inout), lda,
                    thrust::raw_pointer_cast(W), ldw);
    } else if constexpr (std::is_same_v<T, float>) {
        cublasStrsm(cublasHandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, m, n, &alpha,
                    thrust::raw_pointer_cast(A_inout), lda,
                    thrust::raw_pointer_cast(W), ldw);
    } else {
        throw std::runtime_error("Unsupported type.");
    }
}

}  // namespace sy2sb
}  // namespace internal

/**
 * @brief the function to execute symmetric matrix to symmetric band matrix
 *
 * @tparam T
 * @param handle cublas handler
 * @param n size of the matrix A
 * @param A_inout the matrix A
 */
template <typename T>
void sy2sb(const common::CublasHandle& handle, size_t n,
           thrust::device_ptr<T> A_inout) {
    // the size of the matrix A
    const auto m = (size_t)n;
    // the panel size
    const auto b = (size_t)32;
    // the block size
    const auto nb = (size_t)b * 4;

    return;
}

}  // namespace matrix_ops