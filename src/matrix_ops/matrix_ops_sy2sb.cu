#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>

#include <cstddef>

#include "gpu_handle_wrappers.h"
#include "matrix_ops.cuh"

namespace matrix_ops {

namespace internal {
namespace sy2sb {

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
 * @brief Perform QR decomposition of a panel of a symmetric matrix.
 * 
 * @tparam T 
 * @param cublasHandle 
 * @param cusolverDnHandle 
 * @param m The number of rows of the panel.
 * @param n The number of columns of the panel.
 * @param A_inout The panel.
 * @param lda The leading dimension of the panel.
 * @param R The matrix R.
 * @param ldr The leading dimension of the matrix R.
 * @param W The matrix W.
 * @param ldw The leading dimension of the matrix W.
 */
template <typename T>
void qr(const common::CublasHandle& cublasHandle,
        const common::CusolverDnHandle& cusolverDnHandle, size_t m, size_t n,
        thrust::device_vector<T>& A_inout, size_t lda,
        thrust::device_vector<T>& R, size_t ldr, thrust::device_vector<T> W,
        size_t ldw) {
    matrix_ops::tsqr(cublasHandle, m, n, A_inout, R, lda, ldr);
    // A <- I - A
    thrust::transform(
        thrust::device,
        thrust::make_zip_iterator(thrust::make_tuple(
            A_inout.begin(), thrust::counting_iterator<size_t>(0))),
        thrust::make_zip_iterator(
            thrust::make_tuple(A_inout.begin() + n * lda,
                               thrust::counting_iterator<size_t>(n * lda))),
        A_inout.begin(), identity_minus_A_functor<T>(m, n, lda));
    W = A_inout;
}

}  // namespace sy2sb
}  // namespace internal

template <typename T>
void sy2sb(const CublasHandle& handle, size_t n,
           thrust::device_vector<T>& A_inout) {
    // the size of the matrix A
    const auto m = (size_t)n;
    // the panel size
    const auto b = (size_t)32;
    // the block size
    const auto nb = (size_t)b * 4;

    return;
}

}  // namespace matrix_ops