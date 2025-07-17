#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

#include <cstddef>
#include <memory>

#include "cublas_v2.h"
#include "gpu_handle_wrappers.h"
#include "matrix_ops.cuh"

template <typename T>
struct subtract_op {
    __host__ __device__ T operator()(const T& a, const T& b) const {
        return a - b;
    }
};

template <typename T>
void cublas_gemm(cublasHandle_t handle, cublasOperation_t transa,
                 cublasOperation_t transb, int m, int n, int k, T alpha,
                 const T* A, int lda, const T* B, int ldb, T beta, T* C,
                 int ldc);

template <>
void cublas_gemm<float>(cublasHandle_t handle, cublasOperation_t transa,
                        cublasOperation_t transb, int m, int n, int k,
                        float alpha, const float* A, int lda, const float* B,
                        int ldb, float beta, float* C, int ldc) {
    cublasSgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta,
                C, ldc);
}

template <>
void cublas_gemm<double>(cublasHandle_t handle, cublasOperation_t transa,
                         cublasOperation_t transb, int m, int n, int k,
                         double alpha, const double* A, int lda,
                         const double* B, int ldb, double beta, double* C,
                         int ldc) {
    cublasDgemm(handle, transa, transb, m, n, k, &alpha, A, lda, B, ldb, &beta,
                C, ldc);
}

template <typename T>
void cublas_nrm2(cublasHandle_t handle, int n, const T* x, int incx, T* result);

template <>
void cublas_nrm2<float>(cublasHandle_t handle, int n, const float* x, int incx,
                        float* result) {
    cublasSnrm2(handle, n, x, incx, result);
}

template <>
void cublas_nrm2<double>(cublasHandle_t handle, int n, const double* x,
                         int incx, double* result) {
    cublasDnrm2(handle, n, x, incx, result);
}

template <typename T>
class TsqrTest : public ::testing::Test {
   protected:
    void SetUp() override {
        handle_ = std::make_unique<common::CublasHandle>();
    }

    void TearDown() override { handle_.reset(); }

    void run_tsqr_test(size_t m, size_t n, bool all_one = false) {
        ASSERT_GE(m, n);

        const size_t lda = m;
        const size_t ldr = n;

        // Initialize input matrix A
        thrust::host_vector<T> h_A_original(m * n);
        if (all_one) {
            thrust::fill(h_A_original.begin(), h_A_original.end(), (T)1.0);
        } else {
            for (size_t i = 0; i < m * n; ++i) {
                h_A_original[i] =
                    static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
            }
        }

        // Copy to device
        thrust::device_vector<T> d_A_original = h_A_original;
        thrust::device_vector<T> d_A_inout = h_A_original;
        thrust::device_vector<T> d_R_result(n * n);

        // Run TSQR
        EXPECT_NO_THROW(matrix_ops::tsqr<T>(*handle_, m, n, d_A_inout.data(),
                                            d_R_result.data(), lda, ldr));

        // After tsqr, d_A_inout contains Q, d_R_result contains R
        thrust::device_ptr<T> d_Q = d_A_inout.data();
        thrust::device_ptr<T> d_R = d_R_result.data();

        // --- Verification 1: Backward Error ||A - QR|| / ||A|| ---
        thrust::device_vector<T> d_A_reconstructed(m * n);
        T alpha = 1.0;
        T beta = 0.0;

        cublas_gemm<T>(handle_->get(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, alpha,
                       thrust::raw_pointer_cast(d_Q), lda,
                       thrust::raw_pointer_cast(d_R), ldr, beta,
                       thrust::raw_pointer_cast(d_A_reconstructed.data()), lda);

        thrust::device_vector<T> d_residual = d_A_original;
        thrust::transform(d_A_original.begin(), d_A_original.end(),
                          d_A_reconstructed.begin(), d_residual.begin(),
                          subtract_op<T>());

        T norm_A_orig, norm_residual;
        cublasSetPointerMode(handle_->get(), CUBLAS_POINTER_MODE_DEVICE);
        thrust::device_vector<T> d_norm_A_orig(1);
        thrust::device_vector<T> d_norm_residual(1);

        cublas_nrm2<T>(handle_->get(), m * n,
                       thrust::raw_pointer_cast(d_A_original.data()), 1,
                       thrust::raw_pointer_cast(d_norm_A_orig.data()));
        cublas_nrm2<T>(handle_->get(), m * n,
                       thrust::raw_pointer_cast(d_residual.data()), 1,
                       thrust::raw_pointer_cast(d_norm_residual.data()));

        cublasSetPointerMode(handle_->get(), CUBLAS_POINTER_MODE_HOST);

        norm_A_orig = d_norm_A_orig[0];
        norm_residual = d_norm_residual[0];

        double backward_error =
            ((norm_A_orig > 1e-9) ? (norm_residual / norm_A_orig) : 0.0) / n;
        double tolerance = std::is_same_v<T, float> ? 1e-5 : 1e-12;
        EXPECT_LT(backward_error, tolerance);

        // --- Verification 2: Orthogonality of Q ||I - Q'Q|| / ||I|| ---
        thrust::device_vector<T> d_I_reconstructed(n * n);
        thrust::host_vector<T> h_I(n * n, 0.0);
        for (size_t i = 0; i < n; ++i) h_I[i + i * n] = 1.0;
        thrust::device_vector<T> d_I = h_I;

        cublas_gemm<T>(handle_->get(), CUBLAS_OP_T, CUBLAS_OP_N, n, n, m, alpha,
                       thrust::raw_pointer_cast(d_Q), lda,
                       thrust::raw_pointer_cast(d_Q), lda, beta,
                       thrust::raw_pointer_cast(d_I_reconstructed.data()), ldr);

        thrust::device_vector<T> d_residual_I = d_I;
        thrust::transform(d_I.begin(), d_I.end(), d_I_reconstructed.begin(),
                          d_residual_I.begin(), subtract_op<T>());

        T norm_I, norm_residual_I;
        cublasSetPointerMode(handle_->get(), CUBLAS_POINTER_MODE_DEVICE);
        thrust::device_vector<T> d_norm_I(1);
        thrust::device_vector<T> d_norm_residual_I(1);

        cublas_nrm2<T>(handle_->get(), n * n,
                       thrust::raw_pointer_cast(d_I.data()), 1,
                       thrust::raw_pointer_cast(d_norm_I.data()));
        cublas_nrm2<T>(handle_->get(), n * n,
                       thrust::raw_pointer_cast(d_residual_I.data()), 1,
                       thrust::raw_pointer_cast(d_norm_residual_I.data()));

        cublasSetPointerMode(handle_->get(), CUBLAS_POINTER_MODE_HOST);

        norm_I = d_norm_I[0];
        norm_residual_I = d_norm_residual_I[0];

        double orthogonality_error =
            ((norm_I > 1e-9) ? (norm_residual_I / norm_I) : 0.0) / n;
        EXPECT_LT(orthogonality_error, tolerance);

        // --- Verification 3: Check if R is upper triangular ---
        thrust::host_vector<T> h_R_result = d_R_result;
        double upper_tri_tolerance = 1e-12;
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = j + 1; i < n; ++i) {
                const size_t index = i + j * ldr;
                EXPECT_NEAR(h_R_result[index], 0.0, upper_tri_tolerance);
            }
        }
    }

    std::unique_ptr<common::CublasHandle> handle_;
};

using MyTypes = ::testing::Types<float, double>;

TYPED_TEST_SUITE(TsqrTest, MyTypes);

TYPED_TEST(TsqrTest, AlignedSize) { this->run_tsqr_test(512, 32); }

TYPED_TEST(TsqrTest, UnalignedSize) { this->run_tsqr_test(129, 32); }

TYPED_TEST(TsqrTest, SmallSize) { this->run_tsqr_test(7, 4); }

TYPED_TEST(TsqrTest, FatBlockSize) { this->run_tsqr_test(1024, 32); }

TYPED_TEST(TsqrTest, TallAndSkinny) { this->run_tsqr_test(2048, 16); }

TYPED_TEST(TsqrTest, LargeSize) { this->run_tsqr_test(4096, 32); } 

TYPED_TEST(TsqrTest, AllOne) { this->run_tsqr_test(256, 32, true); }

TYPED_TEST(TsqrTest, AllOneSmall) { this->run_tsqr_test(96, 32, true); }