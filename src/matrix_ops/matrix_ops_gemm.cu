#include "matrix_ops.cuh"

namespace matrix_ops {

template <typename T>
void gemm(const common::CublasHandle& handle, size_t m, size_t n,
                 size_t k, T alpha, thrust::device_ptr<T> A, size_t lda,
                 bool transA, thrust::device_ptr<T> B, size_t ldb, bool transB,
                 T beta, thrust::device_ptr<T> C, size_t ldc) {
    auto cublas_op_a = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto cublas_op_b = transB ? CUBLAS_OP_T : CUBLAS_OP_N;
    auto cublas_compute_type = CUBLAS_COMPUTE_64F;
    auto cuda_data_type = CUDA_R_64F;
    if constexpr (std::is_same_v<T, double>) {
        cublas_compute_type = CUBLAS_COMPUTE_64F;
        cuda_data_type = CUDA_R_64F;
    } else if constexpr (std::is_same_v<T, float>) {
        cublas_compute_type = CUBLAS_COMPUTE_32F;
        cuda_data_type = CUDA_R_32F;
    } else {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "matrix_gemm only supports float and double");
    }
    cublasGemmEx(handle, cublas_op_a, cublas_op_b, m, n, k, &alpha,
                 thrust::raw_pointer_cast(A), cuda_data_type, lda,
                 thrust::raw_pointer_cast(B), cuda_data_type, ldb, &beta,
                 thrust::raw_pointer_cast(C), cuda_data_type, ldc,
                 cublas_compute_type, CUBLAS_GEMM_DEFAULT);
}

template <typename T>
void gemm(const common::CublasHandle& handle, size_t m, size_t n,
                 size_t k, T alpha, thrust::device_ptr<T> A, size_t lda,
                 thrust::device_ptr<T> B, size_t ldb, T beta,
                 thrust::device_ptr<T> C, size_t ldc) {
    gemm(handle, m, n, k, alpha, A, lda, false, B, ldb, false, beta, C,
                ldc);
}

// explicit instantiation
template void gemm<float>(const common::CublasHandle& handle, size_t m,
                                 size_t n, size_t k, float alpha,
                                 thrust::device_ptr<float> A, size_t lda,
                                 bool transA, thrust::device_ptr<float> B,
                                 size_t ldb, bool transB, float beta,
                                 thrust::device_ptr<float> C, size_t ldc);
template void gemm<double>(const common::CublasHandle& handle, size_t m,
                                  size_t n, size_t k, double alpha,
                                  thrust::device_ptr<double> A, size_t lda,
                                  bool transA, thrust::device_ptr<double> B,
                                  size_t ldb, bool transB, double beta,
                                  thrust::device_ptr<double> C, size_t ldc);
}  // namespace matrix_ops