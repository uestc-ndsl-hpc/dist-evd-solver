#include <gtest/gtest.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

#include "cublas_v2.h"
#include "gpu_handle_wrappers.h"
#include "gtest/gtest.h"
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
class Sy2sbTest : public ::testing::Test {
   protected:
    void SetUp() override { handle_ = common::CublasHandle(); }

    void TearDown() override {}

    void run_sy2sb_test(size_t n) {
        auto A = matrix_ops::create_symmetric_random<T>(n);
        auto lda = n, ldy = n, ldw = n;

        thrust::device_vector<T> d_Y(n * n, (T)0.0f);
        thrust::device_vector<T> d_W(n * n, (T)0.0f);

        matrix_ops::sy2sb<T>(handle_, n, A.data(), lda, d_Y.data(), ldy,
                             d_W.data(), ldw);

        auto WYT = thrust::device_vector<T>(n * n, (T)0.0f);
        matrix_ops::gemm(handle_, n, n, n, (T)1.0f, d_W.data(), n, false,
                         d_Y.data(), n, true, (T)0.0f, WYT.data(), n);
        auto WYT_ptr = WYT.data();
        thrust::transform(
            thrust::device,
            thrust::make_zip_iterator(thrust::make_tuple(
                WYT_ptr, thrust::counting_iterator<size_t>(0))),
            thrust::make_zip_iterator(thrust::make_tuple(
                WYT_ptr + n * n, thrust::counting_iterator<size_t>(n * n))),
            WYT_ptr, identity_minus_A_functor<T>(n, n, lda));
        auto QTQ = thrust::device_vector<T>(n * n, (T)0.0f);
        matrix_ops::gemm(handle_, n, n, n, (T)1.0f, WYT_ptr, n, true,
                         WYT_ptr, n, false, (T)0.0f, QTQ.data(), n);
        auto QTQ_ptr = QTQ.data();
        thrust::transform(
            thrust::device,
            thrust::make_zip_iterator(thrust::make_tuple(
                QTQ_ptr, thrust::counting_iterator<size_t>(0))),
            thrust::make_zip_iterator(thrust::make_tuple(
                QTQ_ptr + n * n, thrust::counting_iterator<size_t>(n * n))),
            QTQ_ptr, identity_minus_A_functor<T>(n, n, n));
        // 计算 cublasXnrm2
        auto norm = (T)0.f;
        if constexpr (std::is_same_v<T, float>) {
            cublasSnrm2(handle_, n * n, QTQ.data().get(), 1, &norm);
            ASSERT_NEAR(norm / n, 0.0f, 1e-4);
        } else if constexpr (std::is_same_v<T, double>) {
            cublasDnrm2(handle_, n * n, QTQ.data().get(), 1, &norm);
            ASSERT_NEAR(norm / n, 0.0, 1e-10);
        }
       
    }

    common::CublasHandle handle_;
};

using MyTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(Sy2sbTest, MyTypes);

TYPED_TEST(Sy2sbTest, Basic) { this->run_sy2sb_test(256); }

TYPED_TEST(Sy2sbTest, SmallerThanNb) { this->run_sy2sb_test(64); }

TYPED_TEST(Sy2sbTest, SmallSize) { this->run_sy2sb_test(129); }

TYPED_TEST(Sy2sbTest, NonDivisibleSize) { this->run_sy2sb_test(3000); }