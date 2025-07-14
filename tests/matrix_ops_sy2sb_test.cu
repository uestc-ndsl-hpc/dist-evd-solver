#include <gtest/gtest.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <memory>

#include "cublas_v2.h"
#include "gpu_handle_wrappers.h"
#include "gtest/gtest.h"
#include "matrix_ops.cuh"

// Helper function to initialize a symmetric matrix
template <typename T>
void make_symmetric(thrust::host_vector<T>& A, size_t n, size_t lda) {
    for (size_t j = 0; j < n; ++j) {
        for (size_t i = j; i < n; ++i) {
            T val = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
            A[i + j * lda] = val;
            A[j + i * lda] = val;
        }
    }
}

template <typename T>
class Sy2sbTest : public ::testing::Test {
   protected:
    void SetUp() override {
        handle_ = std::make_unique<common::CublasHandle>();
    }

    void TearDown() override { handle_.reset(); }

    void run_sy2sb_test(size_t n) {
        const size_t lda = n;
        const size_t b = 32;
        const size_t nb = b * 4;
        const size_t ldy = n;
        const size_t ldw = n;

        // 1. Create host symmetric matrix
        thrust::host_vector<T> h_A(n * n);
        make_symmetric(h_A, n, lda);

        // 2. Copy to device
        thrust::device_vector<T> d_A_orig = h_A;
        thrust::device_vector<T> d_B =
            h_A;  // B will be the output band matrix from sy2sb

        // 3. Allocate Y and W
        thrust::device_vector<T> d_Y(n * nb, 0.0);
        thrust::device_vector<T> d_W(n * nb, 0.0);

        // 4. Call sy2sb
        matrix_ops::sy2sb<T>(*handle_, n, d_B.data(), lda, d_Y.data(), ldy,
                             d_W.data(), ldw);

        // 5. Verification: Reconstruct A from B, Y, W
        // A_re = B - Y*W'*B - B*W*Y' + Y*W'*B*W*Y'
        thrust::device_vector<T> d_A_reconstructed(n * n);

        // Temp storage for GEMM results
        thrust::device_vector<T> d_T1(nb * n);   // W^T * B
        thrust::device_vector<T> d_T2(n * n);    // Y * T1
        thrust::device_vector<T> d_T3(n * nb);   // B * W
        thrust::device_vector<T> d_T4(n * n);    // T3 * Y^T
        thrust::device_vector<T> d_T5(nb * nb);  // T1 * W
        thrust::device_vector<T> d_T6(n * nb);   // Y * T5
        thrust::device_vector<T> d_T7(n * n);    // T6 * Y^T

        const T one = 1.0;
        const T zero = 0.0;
        const T neg_one = -1.0;

        // T1 = W^T * B
        matrix_ops::gemm(*handle_, nb, n, n, one, d_W.data(), ldw, true,
                         d_B.data(), lda, false, zero, d_T1.data(), nb);
        // T2 = Y * T1
        matrix_ops::gemm(*handle_, n, n, nb, one, d_Y.data(), ldy, false,
                         d_T1.data(), nb, false, zero, d_T2.data(), n);

        // T3 = B * W
        matrix_ops::gemm(*handle_, n, nb, n, one, d_B.data(), lda, false,
                         d_W.data(), ldw, false, zero, d_T3.data(), n);
        // T4 = T3 * Y^T
        matrix_ops::gemm(*handle_, n, n, nb, one, d_T3.data(), n, false,
                         d_Y.data(), ldy, true, zero, d_T4.data(), n);

        // T5 = T1 * W
        matrix_ops::gemm(*handle_, nb, nb, n, one, d_T1.data(), nb, false,
                         d_W.data(), ldw, false, zero, d_T5.data(), nb);
        // T6 = Y * T5
        matrix_ops::gemm(*handle_, n, nb, nb, one, d_Y.data(), ldy, false,
                         d_T5.data(), nb, false, zero, d_T6.data(), n);
        // T7 = T6 * Y^T
        matrix_ops::gemm(*handle_, n, n, nb, one, d_T6.data(), n, false,
                         d_Y.data(), ldy, true, zero, d_T7.data(), n);

        // A_reconstructed = B - T2 - T4 + T7
        // Using cublasgeam for C = alpha*A + beta*B
        // A_re = -T2
        if constexpr (std::is_same_v<T, float>) {
            cublasSgeam(handle_->get(), CUBLAS_OP_N, CUBLAS_OP_N, n, n,
                        &neg_one, thrust::raw_pointer_cast(d_T2.data()), n,
                        &zero, thrust::raw_pointer_cast(d_T2.data()), n,
                        thrust::raw_pointer_cast(d_A_reconstructed.data()), n);
        } else {
            cublasDgeam(handle_->get(), CUBLAS_OP_N, CUBLAS_OP_N, n, n,
                        &neg_one, thrust::raw_pointer_cast(d_T2.data()), n,
                        &zero, thrust::raw_pointer_cast(d_T2.data()), n,
                        thrust::raw_pointer_cast(d_A_reconstructed.data()), n);
        }

        // A_re = A_re - T4
        // A_re_new = 1 * A_re_old - 1 * T4
        if constexpr (std::is_same_v<T, float>) {
            cublasSgeam(handle_->get(), CUBLAS_OP_N, CUBLAS_OP_N, n, n, &one,
                        thrust::raw_pointer_cast(d_A_reconstructed.data()), n,
                        &neg_one, thrust::raw_pointer_cast(d_T4.data()), n,
                        thrust::raw_pointer_cast(d_A_reconstructed.data()), n);
        } else {
            cublasDgeam(handle_->get(), CUBLAS_OP_N, CUBLAS_OP_N, n, n, &one,
                        thrust::raw_pointer_cast(d_A_reconstructed.data()), n,
                        &neg_one, thrust::raw_pointer_cast(d_T4.data()), n,
                        thrust::raw_pointer_cast(d_A_reconstructed.data()), n);
        }

        // A_re = A_re + T7
        // A_re_new = 1 * A_re_old + 1 * T7
        if constexpr (std::is_same_v<T, float>) {
            cublasSgeam(handle_->get(), CUBLAS_OP_N, CUBLAS_OP_N, n, n, &one,
                        thrust::raw_pointer_cast(d_A_reconstructed.data()), n,
                        &one, thrust::raw_pointer_cast(d_T7.data()), n,
                        thrust::raw_pointer_cast(d_A_reconstructed.data()), n);
        } else {
            cublasDgeam(handle_->get(), CUBLAS_OP_N, CUBLAS_OP_N, n, n, &one,
                        thrust::raw_pointer_cast(d_A_reconstructed.data()), n,
                        &one, thrust::raw_pointer_cast(d_T7.data()), n,
                        thrust::raw_pointer_cast(d_A_reconstructed.data()), n);
        }
        // A_re = A_re + B
        // A_re_new = 1 * A_re_old + 1 * B
        if constexpr (std::is_same_v<T, float>) {
            cublasSgeam(handle_->get(), CUBLAS_OP_N, CUBLAS_OP_N, n, n, &one,
                        thrust::raw_pointer_cast(d_A_reconstructed.data()), n,
                        &one, thrust::raw_pointer_cast(d_B.data()), n,
                        thrust::raw_pointer_cast(d_A_reconstructed.data()), n);
        } else {
            cublasDgeam(handle_->get(), CUBLAS_OP_N, CUBLAS_OP_N, n, n, &one,
                        thrust::raw_pointer_cast(d_A_reconstructed.data()), n,
                        &one, thrust::raw_pointer_cast(d_B.data()), n,
                        thrust::raw_pointer_cast(d_A_reconstructed.data()), n);
        }

        // 6. Copy back to host
        thrust::host_vector<T> h_A_reconstructed = d_A_reconstructed;

        // 7. Compare
        double tolerance = std::is_same_v<T, float> ? 1e-4 : 1e-10;
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < n; ++i) {  // check full matrix
                const size_t index = i + j * lda;
                const size_t index_orig = i + j * lda;
                EXPECT_NEAR(h_A[index_orig], h_A_reconstructed[index],
                            tolerance * std::abs(h_A[index_orig]));
            }
        }
    }

    std::unique_ptr<common::CublasHandle> handle_;
};

using MyTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(Sy2sbTest, MyTypes);

TYPED_TEST(Sy2sbTest, Basic) { this->run_sy2sb_test(256); }

TYPED_TEST(Sy2sbTest, SmallerThanNb) { this->run_sy2sb_test(64); }

TYPED_TEST(Sy2sbTest, SmallSize) { this->run_sy2sb_test(129); }

TYPED_TEST(Sy2sbTest, NonDivisibleSize) { this->run_sy2sb_test(300); }