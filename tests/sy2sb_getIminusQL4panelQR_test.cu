#include <gpu_handle_wrappers.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstddef>

#include "matrix_ops.cuh"
#include "sy2sb_panelqr.cuh"

template <typename T>
class GetIminusQL4panelQRTest : public ::testing::Test {
   protected:
    void SetUp() override {
        handle_ = std::make_unique<common::CusolverDnHandle>();
    }

    void TearDown() override { handle_.reset(); }

    void run_test(size_t m, size_t n, size_t lda) {
        auto d_A = matrix_ops::create_uniform_random<T>(lda, n);
        auto A_ptr = d_A.data();
        EXPECT_NO_THROW(matrix_ops::internal::sy2sb::getIminusQL4panelQR<T>(
            *handle_, m, n, A_ptr, lda));
    }

    std::unique_ptr<common::CusolverDnHandle> handle_;
};

using MyTypes = ::testing::Types<float, double>;

TYPED_TEST_SUITE(GetIminusQL4panelQRTest, MyTypes);

TYPED_TEST(GetIminusQL4panelQRTest, Square) { this->run_test(16, 16, 16); }

TYPED_TEST(GetIminusQL4panelQRTest, SquareLda) { this->run_test(17, 17, 20); }

TYPED_TEST(GetIminusQL4panelQRTest, Tall) { this->run_test(32, 16, 32); }

TYPED_TEST(GetIminusQL4panelQRTest, TallLda) { this->run_test(32, 16, 40); }