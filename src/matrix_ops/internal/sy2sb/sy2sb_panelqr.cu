#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <stdexcept>

#include "gpu_handle_wrappers.h"
#include "matrix_ops.cuh"
#include "sy2sb_panelqr.cuh"

namespace matrix_ops {
namespace internal {
namespace sy2sb {

template <typename T>
void getIminusQL4panelQR(const common::CusolverDnHandle& handle, size_t m,
                         size_t n, thrust::device_ptr<T> A_inout, size_t lda) {
    auto lwork = (int)0;
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
    // matrix_ops::print(A_inout, m, n, lda, "A_inout before getrf");
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
    auto info_host = thrust::host_vector<int>(info);
    if (info_host[0] != 0) {
        throw std::runtime_error("Failed to factorize the matrix.");
    }
    // matrix_ops::print(A_inout, m, n, lda, "A_inout before get L");

    // extract the lower triangular part of the matrix
    try {
        size_t num_elements = m * n;
        thrust::for_each(thrust::device, thrust::counting_iterator<size_t>(0),
                         thrust::counting_iterator<size_t>(num_elements),
                         extract_L_functor_2d<T>(A_inout, m, lda));
    } catch (std::exception& e) {
        throw std::runtime_error(fmt::format(
            "Error in panelQR extract L \n n = {} lda = {} \n error: {}", n,
            lda, e.what()));
    }

    // matrix_ops::print(A_inout, m, n, lda, "A_inout after get L");
}

template <typename T>
void panelQR(const common::CublasHandle& cublasHandle,
             const common::CusolverDnHandle& cusolverDnHandle, size_t m,
             size_t n, thrust::device_ptr<T> A_inout, size_t lda,
             thrust::device_ptr<T> R, size_t ldr, thrust::device_ptr<T> W,
             size_t ldw) {
    // tsqr, A_inout <- Q R
    matrix_ops::tsqr<T>(cublasHandle, m, n, A_inout, R, lda, ldr);

    try {
        size_t num_elements = m * n;
        thrust::for_each(thrust::device, thrust::counting_iterator<size_t>(0),
                         thrust::counting_iterator<size_t>(num_elements),
                         identity_minus_A_functor_2d<T>(A_inout, m, lda));
    } catch (std::exception& e) {
        throw std::runtime_error(
            fmt::format("Error in panelQR I-Q \n n = {} lda = {} \n error: {}",
                        n, lda, e.what()));
    }

    // matrix_ops::print(A_inout, m, n, lda, "A_inout after I-Q");

    // W = A_inout (a.k.a. I-Q)
    matrix_ops::matrix_copy<thrust::device_ptr<T>, thrust::device_ptr<T>, T>(
        A_inout, lda, W, ldw, m, n);

    // A <- "(I - Q) --> LU" [L]
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

template void getIminusQL4panelQR<float>(const common::CusolverDnHandle&,
                                         size_t, size_t,
                                         thrust::device_ptr<float>, size_t);
template void getIminusQL4panelQR<double>(const common::CusolverDnHandle&,
                                          size_t, size_t,
                                          thrust::device_ptr<double>, size_t);

template void panelQR<float>(const common::CublasHandle&,
                             const common::CusolverDnHandle&, size_t, size_t,
                             thrust::device_ptr<float>, size_t,
                             thrust::device_ptr<float>, size_t,
                             thrust::device_ptr<float>, size_t);
template void panelQR<double>(const common::CublasHandle&,
                              const common::CusolverDnHandle&, size_t, size_t,
                              thrust::device_ptr<double>, size_t,
                              thrust::device_ptr<double>, size_t,
                              thrust::device_ptr<double>, size_t);

}  // namespace sy2sb
}  // namespace internal
}  // namespace matrix_ops