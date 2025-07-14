#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <memory>

#include "cublas_v2.h"
#include "gpu_handle_wrappers.h"
#include "gtest/gtest.h"
#include "matrix_ops.cuh"

template <typename T>
void cublas_syr2k(cublasHandle_t handle, size_t n, size_t k, T alpha, const T* A,
                  size_t lda, const T* B, size_t ldb, T beta, T* C, size_t ldc);

template <>
void cublas_syr2k<float>(cublasHandle_t handle, size_t n, size_t k, float alpha,
                         const float* A, size_t lda, const float* B,
                         size_t ldb, float beta, float* C, size_t ldc) {
    cublasSsyr2k(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, &alpha, A,
                 lda, B, ldb, &beta, C, ldc);
}

template <>
void cublas_syr2k<double>(cublasHandle_t handle, size_t n, size_t k,
                          double alpha, const double* A, size_t lda,
                          const double* B, size_t ldb, double beta, double* C,
                          size_t ldc) {
    cublasDsyr2k(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, &alpha, A,
                 lda, B, ldb, &beta, C, ldc);
}

template <typename T>
class Syr2kTest : public ::testing::Test {
   protected:
    void SetUp() override {
        handle_ = std::make_unique<common::CublasHandle>();
    }

    void TearDown() override { handle_.reset(); }

    void run_syr2k_test(size_t n, size_t k) {
        const size_t lda = n;
        const size_t ldb = n;
        const size_t ldc = n;
        const T alpha = 1.5;
        const T beta = 0.5;

        thrust::host_vector<T> h_A(n * k);
        thrust::host_vector<T> h_B(n * k);
        thrust::host_vector<T> h_C(n * n);

        for (size_t i = 0; i < n * k; ++i) {
            h_A[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
            h_B[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
        }
        for (size_t i = 0; i < n * n; ++i) {
            h_C[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
        }

        thrust::device_vector<T> d_A = h_A;
        thrust::device_vector<T> d_B = h_B;
        thrust::device_vector<T> d_C_result = h_C;
        thrust::device_vector<T> d_C_expected = h_C;

        matrix_ops::syr2k<T>(*handle_, n, k, alpha, d_A.data(), lda, d_B.data(),
                             ldb, beta, d_C_result.data(), ldc);

        cublas_syr2k<T>(handle_->get(), n, k, alpha,
                        thrust::raw_pointer_cast(d_A.data()), lda,
                        thrust::raw_pointer_cast(d_B.data()), ldb, beta,
                        thrust::raw_pointer_cast(d_C_expected.data()), ldc);

        thrust::host_vector<T> h_C_result = d_C_result;
        thrust::host_vector<T> h_C_expected = d_C_expected;

        double tolerance = std::is_same_v<T, float> ? 1e-4 : 1e-10;
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = j; i < n; ++i) {
                const size_t index = i + j * ldc;
                EXPECT_NEAR(h_C_expected[index], h_C_result[index],
                            tolerance * std::abs(h_C_expected[index]));
            }
        }
    }

    std::unique_ptr<common::CublasHandle> handle_;
};

using MyTypes = ::testing::Types<float, double>;

TYPED_TEST_SUITE(Syr2kTest, MyTypes);

TYPED_TEST(Syr2kTest, AlignedSize) { this->run_syr2k_test(64, 16); }

TYPED_TEST(Syr2kTest, UnalignedSize) { this->run_syr2k_test(129, 16); }

TYPED_TEST(Syr2kTest, SmallSize) { this->run_syr2k_test(7, 4); }

TYPED_TEST(Syr2kTest, VeryLargeSize) { this->run_syr2k_test(16384, 1024); }