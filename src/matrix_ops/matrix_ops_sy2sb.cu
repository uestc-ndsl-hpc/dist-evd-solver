#include <thrust/device_vector.h>

#include <cstddef>

#include "gpu_handle_wrappers.h"
#include "internal/sy2sb/sy2sb_panelqr.cuh"
#include "matrix_ops.cuh"

namespace matrix_ops {

namespace internal {

/**
 * @brief Functor to copy the lower triangular part of a matrix to the upper
 * triangular part.
 *
 * @tparam T Data type of the matrix elements.
 */
template <typename T>
struct make_symmetric_functor {
    thrust::device_ptr<T> A_;
    size_t n_;
    size_t lda_;

    make_symmetric_functor(thrust::device_ptr<T> A, size_t n, size_t lda)
        : A_(A), n_(n), lda_(lda) {}

    __device__ void operator()(const size_t& k) const {
        size_t j = k % n_;  // row
        size_t i = k / n_;  // col
        if (j < i) {
            A_[j + i * lda_] = A_[i + j * lda_];
        }
    }
};

/**
 * @brief Functor to clear a submatrix to zero.
 *
 * @tparam T Data type of the matrix elements.
 */
template <typename T>
struct clear_matrix_functor {
    thrust::device_ptr<T> ptr_;
    size_t rows_;
    size_t lda_;

    clear_matrix_functor(thrust::device_ptr<T> p, size_t rows, size_t lda)
        : ptr_(p), rows_(rows), lda_(lda) {}

    __device__ void operator()(const size_t& k) const {
        size_t i = k % rows_;  // row index
        size_t j = k / rows_;  // column index
        ptr_[i + j * lda_] = 0;
    }
};

/**
 * @brief the recursive function to compute the SBR of the panel
 *
 * @tparam T the data type of the matrix
 * @param cublasHandle cublas handle
 * @param cusolverHandle cusolver handle
 * @param n size of the panel
 * @param A panel ptr
 * @param lda leading dimension of panel
 * @param Y Y panel ptr
 * @param ldy leading dimension of Y panel
 * @param W W panel ptr
 * @param ldw leading dimension of W panel
 * @param oriA original matrix A ptr
 * @param ldoA leading dimension of original matrix A
 * @param Z Z panel ptr
 * @param ldz leading dimension of Z panel
 * @param R R panel ptr
 * @param ldr leading dimension of R panel
 * @param nb panel size
 * @param b block size
 */
// TODO: Finish timplement of the sy2sb recursive function
template <typename T>
void sy2sb_recrusive(const common::CublasHandle& cublasHandle,
                     const common::CusolverDnHandle& cusolverHandle, size_t n,
                     thrust::device_ptr<T> A, size_t lda,
                     thrust::device_ptr<T> Y, size_t ldy,
                     thrust::device_ptr<T> W, size_t ldw,
                     thrust::device_ptr<T> oriA, size_t ldoA,
                     thrust::device_ptr<T> Z, size_t ldz,
                     thrust::device_ptr<T> R, size_t ldr, size_t nb, size_t b) {
    // tmp work space for gemm

    thrust::device_vector<T> work(nb * nb);
    auto work_ptr = work.data();
    auto ldwork = nb;

    for (auto i = b; i <= nb && i < n; i += b) {
        auto panel_m = n - i;
        auto panel_n = b;
        auto panel_ptr = A + i + (i - b) * lda;

        auto panel_W_ptr = W + i + (i - b) * ldw;
        auto panel_R_ptr = R + i + (i - b) * ldr;
        auto panel_Y_ptr = Y + i + (i - b) * ldy;
        auto panel_Z_ptr = Z + i + (i - b) * ldz;

        // compute the panel QR
        internal::sy2sb::panelQR(cublasHandle, cusolverHandle, panel_m, panel_n,
                                 panel_ptr, lda, panel_R_ptr, ldr, panel_W_ptr,
                                 ldw);

        // copy panel data to panelY (using lda)
        matrix_ops::matrix_copy<thrust::device_ptr<T>, thrust::device_ptr<T>,
                                T>(panel_ptr, lda, panel_Y_ptr, ldy, panel_m,
                                   panel_n);

        // copy panelR data to panel (using lda)
        matrix_ops::matrix_copy<thrust::device_ptr<T>, thrust::device_ptr<T>,
                                T>(panel_R_ptr, ldr, panel_ptr, lda, panel_m,
                                   panel_n);

        // do gemm to compute Z,Y.W
        // first panel process

        if (i == b) {
            auto panel_OriA_ptr = oriA + b * ldoA + b;
            matrix_ops::gemm(cublasHandle, panel_m, b, panel_m, (T)1,
                             panel_OriA_ptr, ldoA, false, panel_W_ptr, ldw,
                             false, (T)0, panel_Z_ptr, ldz);
            matrix_ops::gemm(cublasHandle, b, b, panel_m, (T)1, panel_W_ptr,
                             ldw, true, panel_Z_ptr, ldz, false, (T)0, work_ptr,
                             ldwork);
            matrix_ops::gemm(cublasHandle, panel_m, b, b, (T)(-0.5),
                             panel_Y_ptr, ldy, false, work_ptr, ldwork, false,
                             (T)1, panel_Z_ptr, ldz);
        } else {
            auto panel_OriA_ptr = oriA + i * ldoA + i;
            matrix_ops::gemm(cublasHandle, panel_m, b, panel_m, (T)1,
                             panel_OriA_ptr, ldoA, false, panel_W_ptr, ldw,
                             false, (T)0, panel_Z_ptr, ldz);
            matrix_ops::gemm(cublasHandle, i - b, b, panel_m, (T)1, Z + i, ldz,
                             true, panel_W_ptr, ldw, false, (T)0, work_ptr,
                             ldwork);
            matrix_ops::gemm(cublasHandle, panel_m, b, i - b, (T)(-1), Y + i,
                             ldy, false, work_ptr, ldwork, false, (T)1,
                             panel_Z_ptr, ldz);
            matrix_ops::gemm(cublasHandle, i - b, b, panel_m, (T)(1), Y + i,
                             ldy, true, panel_W_ptr, ldw, false, (T)0, work_ptr,
                             ldwork);
            matrix_ops::gemm(cublasHandle, panel_m, b, i - b, (T)(-1), Z + i,
                             ldz, false, work_ptr, ldwork, false, (T)1,
                             panel_Z_ptr, ldz);
            matrix_ops::gemm(cublasHandle, b, b, panel_m, (T)1, panel_W_ptr,
                             ldw, true, panel_Z_ptr, ldz, false, T(0), work_ptr,
                             ldwork);
            matrix_ops::gemm(cublasHandle, panel_m, b, b, (T)(-0.5),
                             panel_Y_ptr, ldy, false, work_ptr, ldwork, false,
                             (T)1, panel_Z_ptr, ldz);
        }
        if (i < nb) {
            matrix_ops::gemm(cublasHandle, panel_m, b, i, (T)(-1), Y + i, ldy,
                             false, Z + i, ldz, true, (T)1, A + i + i * lda,
                             lda);
            matrix_ops::gemm(cublasHandle, panel_m, b, i, (T)(-1), Z + i, ldz,
                             false, Y + i, ldy, true, (T)1, A + i + i * lda,
                             lda);
        }
    }

    // recursive exit
    if (n <= nb) return;

    // execute syr2k
    matrix_ops::syr2k(cublasHandle, n - nb, nb, (T)(-1), Y + nb, ldy, Z + nb,
                      ldz, (T)1, oriA + nb * ldoA + nb, ldoA);

    // copy Lower to Upper to build a full symmetric matrix for the updated part
    auto sub_matrix_ptr_oA = oriA + nb * ldoA + nb;
    auto sub_matrix_n = n - nb;
    auto sub_matrix_ptr_A = A + nb * lda + nb;
    thrust::for_each(
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(sub_matrix_n * sub_matrix_n),
        make_symmetric_functor<T>(sub_matrix_ptr_oA, sub_matrix_n, ldoA));

    matrix_ops::matrix_copy<thrust::device_ptr<T>, thrust::device_ptr<T>, T>(
        sub_matrix_ptr_oA, ldoA, sub_matrix_ptr_A, lda, sub_matrix_n,
        sub_matrix_n);

    // Clear a part of the Z matrix for the next recursive step.
    {
        size_t rows_to_clear = n - nb - b;
        size_t cols_to_clear = nb;
        auto ptr_to_clear = Z + b;
        size_t total_elements = rows_to_clear * cols_to_clear;

        if (n - nb > b) {
            thrust::for_each(
                thrust::make_counting_iterator<size_t>(0),
                thrust::make_counting_iterator<size_t>(total_elements),
                clear_matrix_functor<T>(ptr_to_clear, rows_to_clear, ldz));
        }
    }

    // recursive for rest
    sy2sb_recrusive(cublasHandle, cusolverHandle, n - nb, sub_matrix_ptr_A, lda,
                    Y + nb + nb * ldy, ldy, W + nb + nb * ldw, ldw,
                    sub_matrix_ptr_oA, ldoA, Z, ldz, R + nb, ldr, nb, b);
}
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
           thrust::device_ptr<T> A_inout, size_t lda,
           thrust::device_ptr<T> Y_inout, size_t ldy,
           thrust::device_ptr<T> W_inout, size_t ldw) {
    // the panel size
    const auto b = (size_t)32;
    // the block size
    const auto nb = (size_t)b * 4;

    auto cusolverHandle = common::CusolverDnHandle();

    // tmp R for compute W && Y
    auto R = thrust::device_vector<T>(n * nb);
    auto R_ptr = R.data();
    auto ldr = n;

    // oriA for SBR computations
    auto oriA = thrust::device_vector<T>(n * n);
    auto oriA_ptr = oriA.data();
    thrust::copy(A_inout, A_inout + n * n, oriA_ptr);
    auto ldoA = n;

    // Z for SBR computations
    auto Z = thrust::device_vector<T>(n * nb);
    auto Z_ptr = Z.data();
    auto ldz = n;

    internal::sy2sb_recrusive(handle, cusolverHandle, n, A_inout, lda, Y_inout,
                              ldy, W_inout, ldw, oriA_ptr, ldoA, Z_ptr, ldz,
                              R_ptr, ldr, nb, b);

    // make A_inout symmetric
    thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                     thrust::make_counting_iterator<size_t>(n * n),
                     internal::make_symmetric_functor<T>(A_inout, n, lda));

    return;
}

}  // namespace matrix_ops

template void matrix_ops::sy2sb<float>(
    const common::CublasHandle& handle, size_t n,
    thrust::device_ptr<float> A_inout, size_t lda,
    thrust::device_ptr<float> Y_inout, size_t ldy,
    thrust::device_ptr<float> W_inout, size_t ldw);

template void matrix_ops::sy2sb<double>(
    const common::CublasHandle& handle, size_t n,
    thrust::device_ptr<double> A_inout, size_t lda,
    thrust::device_ptr<double> Y_inout, size_t ldy,
    thrust::device_ptr<double> W_inout, size_t ldw);
