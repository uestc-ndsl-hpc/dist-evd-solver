#include <cstddef>

#include "matrix_ops.cuh"

namespace matrix_ops {

template <typename T>
void syr2k(const common::CublasHandle& handle, size_t n, size_t k, T alpha,
           thrust::device_ptr<T> A, size_t lda, thrust::device_ptr<T> B,
           size_t ldb, T beta, thrust::device_ptr<T> C, size_t ldc, size_t nb) {
    auto num_block = n / nb;
    auto remain = n % nb;
    auto one = (T)1;

    if constexpr (std::is_same_v<T, float>) {
        cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, nb, nb, k,
                                  &alpha, A.get(), lda, nb, B.get(), ldb, nb,
                                  &beta, C.get(), ldc, nb + nb * ldc,
                                  num_block);
        cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, nb, nb, k,
                                  &alpha, B.get(), ldb, nb, A.get(), lda, nb,
                                  &one, C.get(), ldc, nb + nb * ldc, num_block);
    } else if constexpr (std::is_same_v<T, double>) {
        cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, nb, nb, k,
                                  &alpha, A.get(), lda, nb, B.get(), ldb, nb,
                                  &beta, C.get(), ldc, nb + nb * ldc,
                                  num_block);
        cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, nb, nb, k,
                                  &alpha, B.get(), ldb, nb, A.get(), lda, nb,
                                  &one, C.get(), ldc, nb + nb * ldc, num_block);
    }

    if (remain > 0) {
        auto offset = num_block * nb;
        auto A_ptr = A + offset;
        auto B_ptr = B + offset;
        auto C_ptr = C + offset + offset * ldc;

        if constexpr (std::is_same_v<T, float>) {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, remain, remain, k,
                        &alpha, A_ptr.get(), lda, B_ptr.get(), ldb, &beta,
                        C_ptr.get(), ldc);
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, remain, remain, k,
                        &alpha, B_ptr.get(), ldb, A_ptr.get(), lda, &one,
                        C_ptr.get(), ldc);
        } else if constexpr (std::is_same_v<T, double>) {
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, remain, remain, k,
                        &alpha, A_ptr.get(), lda, B_ptr.get(), ldb, &beta,
                        C_ptr.get(), ldc);
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, remain, remain, k,
                        &alpha, B_ptr.get(), ldb, A_ptr.get(), lda, &one,
                        C_ptr.get(), ldc);
        }
    }

    for (auto i = nb; i < n; i *= 2) {
        num_block = n / (i * 2);
        remain = n - (num_block * i * 2);

        if constexpr (std::is_same_v<T, float>) {
            cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, i, i, k,
                                      &alpha, A.get() + i, lda, 2 * i, B.get(),
                                      ldb, 2 * i, &beta, C.get() + i, ldc,
                                      2 * (i + i * ldc), num_block);
            cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, i, i, k,
                                      &alpha, B.get() + i, ldb, 2 * i, A.get(),
                                      lda, 2 * i, &one, C.get() + i, ldc,
                                      2 * (i + i * ldc), num_block);
        } else if constexpr (std::is_same_v<T, double>) {
            cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, i, i, k,
                                      &alpha, A.get() + i, lda, 2 * i, B.get(),
                                      ldb, 2 * i, &beta, C.get() + i, ldc,
                                      2 * (i + i * ldc), num_block);
            cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_T, i, i, k,
                                      &alpha, B.get() + i, ldb, 2 * i, A.get(),
                                      lda, 2 * i, &one, C.get() + i, ldc,
                                      2 * (i + i * ldc), num_block);
        }

        if (remain > i) {
            auto offset_col = num_block * 2 * i;
            auto offset_row = i + offset_col;

            if constexpr (std::is_same_v<T, float>) {
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, remain - i, i, k,
                            &alpha, A.get() + offset_row, lda,
                            B.get() + offset_col, ldb, &beta,
                            C.get() + offset_row + offset_col * ldc, ldc);
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, remain - i, i, k,
                            &alpha, B.get() + offset_row, ldb,
                            A.get() + offset_col, lda, &one,
                            C.get() + offset_row + offset_col * ldc, ldc);
            } else if constexpr (std::is_same_v<T, double>) {
                cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, remain - i, i, k,
                            &alpha, A.get() + offset_row, lda,
                            B.get() + offset_col, ldb, &beta,
                            C.get() + offset_row + offset_col * ldc, ldc);
                cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, remain - i, i, k,
                            &alpha, B.get() + offset_row, ldb,
                            A.get() + offset_col, lda, &one,
                            C.get() + offset_row + offset_col * ldc, ldc);
            }
        }
    }
}

template <typename T>
void syr2k(const common::CublasHandle& handle, size_t n, size_t k, T alpha,
           thrust::device_ptr<T> A, size_t lda, thrust::device_ptr<T> B,
           size_t ldb, T beta, thrust::device_ptr<T> C, size_t ldc) {
    syr2k(handle, n, k, alpha, A, lda, B, ldb, beta, C, ldc, 32);
    return;
}

template void syr2k<float>(const common::CublasHandle& handle, size_t n,
                           size_t k, float alpha, thrust::device_ptr<float> A,
                           size_t lda, thrust::device_ptr<float> B, size_t ldb,
                           float beta, thrust::device_ptr<float> C, size_t ldc);
template void syr2k<double>(const common::CublasHandle& handle, size_t n,
                            size_t k, double alpha,
                            thrust::device_ptr<double> A, size_t lda,
                            thrust::device_ptr<double> B, size_t ldb,
                            double beta, thrust::device_ptr<double> C,
                            size_t ldc);

}  // namespace matrix_ops