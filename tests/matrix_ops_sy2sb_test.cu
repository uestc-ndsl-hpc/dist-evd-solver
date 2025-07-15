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


        auto A = matrix_ops::create_symmetric_random<T>(n);
        auto lda = n, ldy = n, ldw = n;

        thrust::device_vector<T> d_Y(n * n, 0.0);
        thrust::device_vector<T> d_W(n * n, 0.0);

        matrix_ops::sy2sb<T>(*handle_, n, A.data(), lda, d_Y.data(), ldy,
                             d_W.data(), ldw);
    }

    std::unique_ptr<common::CublasHandle> handle_;
};

using MyTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(Sy2sbTest, MyTypes);

TYPED_TEST(Sy2sbTest, Basic) { this->run_sy2sb_test(256); }

TYPED_TEST(Sy2sbTest, SmallerThanNb) { this->run_sy2sb_test(64); }

TYPED_TEST(Sy2sbTest, SmallSize) { this->run_sy2sb_test(129); }

TYPED_TEST(Sy2sbTest, NonDivisibleSize) { this->run_sy2sb_test(3000); }