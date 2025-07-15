#include <gpu_handle_wrappers.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <cstddef>

#include "matrix_ops.cuh"

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

template <typename T>
class IminusATest : public ::testing::Test {
   protected:
    void SetUp() override {
        handle_ = std::make_unique<common::CublasHandle>();
    }

    void TearDown() override { handle_.reset(); }

    void run_IminusA_test(size_t m, size_t n, size_t lda) {
        // col-major && leading dimension = lda
        auto A = matrix_ops::create_normal_random<T>(m, n, lda);

        thrust::host_vector<T> h_A = A;

        auto A_ptr = A.data();
        thrust::transform(
            thrust::device,
            thrust::make_zip_iterator(
                thrust::make_tuple(A_ptr, thrust::counting_iterator<size_t>(0))),
            thrust::make_zip_iterator(thrust::make_tuple(
                A_ptr + lda * n,
                thrust::counting_iterator<size_t>(lda * n))),
            A_ptr, identity_minus_A_functor<T>(m, n, lda));

        thrust::host_vector<T> h_result = A;

        thrust::host_vector<T> h_expected(lda * n);
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < m; ++i) {
                T val = -h_A[j * lda + i];
                if (i == j) {
                    val += 1.0;
                }
                h_expected[j * lda + i] = val;
            }
        }

        for (size_t i = 0; i < h_result.size(); ++i) {
            if (i % lda < m && i / lda < n) {
                ASSERT_NEAR(h_result[i], h_expected[i], 1e-5);
            }
        }

    }

    std::unique_ptr<common::CublasHandle> handle_;
};

using MyTypes = ::testing::Types<float, double>;

TYPED_TEST_SUITE(IminusATest, MyTypes);

TYPED_TEST(IminusATest, AlignedSize) { this->run_IminusA_test(64, 16, 64); }

TYPED_TEST(IminusATest, UnalignedSize) { this->run_IminusA_test(129, 16, 129); }

TYPED_TEST(IminusATest, AlignedSizeLda) { this->run_IminusA_test(64, 16, 70); }

TYPED_TEST(IminusATest, UnalignedSizeLda) {
    this->run_IminusA_test(129, 16, 135);
}