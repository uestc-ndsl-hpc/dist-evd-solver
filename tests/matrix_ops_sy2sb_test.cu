#include <gtest/gtest.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include <cstddef>

#include "cublas_v2.h"
#include "gpu_handle_wrappers.h"
#include "gtest/gtest.h"
#include "matrix_ops.cuh"
#include "matrix_ops_dist.cuh"

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
void computeWfromOriWY(common::CublasHandle& handle, size_t n, size_t b,
                       thrust::device_ptr<T> W, size_t ldw,
                       thrust::device_ptr<T> Y, size_t ldy) {
    auto work = thrust::device_vector<T>(n * b);
    for (auto i = 2 * b; i <= n; i += b) {
        matrix_ops::gemm(handle, i - b, b, n, (T)1.f, Y, ldy, true,
                         W + (i - b) * ldw, ldw, false, (T)0.f, work.data(), n);
        matrix_ops::gemm(handle, n, b, i - b, (T)-1.f, W, ldw, false,
                         work.data(), n, false, (T)1.f, W + (i - b) * ldw, ldw);
    }
}

template <typename T>
class Sy2sbTest : public ::testing::Test {
   protected:
    void SetUp() override { handle_ = common::CublasHandle(); }

    void TearDown() override {}

    void run_sy2sb_test(size_t n, size_t nb = 128, size_t b = 32) {
        auto A = matrix_ops::create_symmetric_random<T>(n, true);
        run_sy2sb_test(n, A, nb, b);
    }

    void run_dist_sy2sb_test(size_t n, size_t nb = 64, size_t b = 32,
                             size_t gpu_num = 1) {
        auto A_data = matrix_ops::create_symmetric_random<T>(n, true);

        auto lda = n, ldy = n, ldw = n;

        thrust::host_vector<T> A_h(A_data.begin(), A_data.end());
        thrust::host_vector<T> W_h(n * n);
        thrust::host_vector<T> Y_h(n * n);

        thrust::device_vector<T> d_Y(n * n, (T)0.0f);
        thrust::device_vector<T> d_W(n * n, (T)0.0f);
        thrust::device_vector<T> d_A(A_data.begin(), A_data.end());

        EXPECT_NO_THROW(matrix_ops::dist::sy2sb(handle_, n, A_h.data(), n,
                                                W_h.data(), n, Y_h.data(), n,
                                                nb, b, gpu_num));

        thrust::copy(Y_h.begin(), Y_h.end(), d_Y.begin());
        thrust::copy(W_h.begin(), W_h.end(), d_W.begin());
        thrust::copy(A_h.begin(), A_h.end(), d_A.begin());

        // compute W from oriW
        computeWfromOriWY(handle_, n, b, d_W.data(), ldw, d_Y.data(), ldy);

        // Q  = I - WYT QTBQ = A  QTQ = I
        auto transformation_matrix = thrust::device_vector<T>(n * n, (T)0.0f);
        matrix_ops::gemm(handle_, n, n, n, (T)1.0f, d_W.data(), n, false,
                         d_Y.data(), n, true, (T)0.0f,
                         transformation_matrix.data(), n);
        auto transformation_ptr = transformation_matrix.data();
        thrust::transform(
            thrust::device,
            thrust::make_zip_iterator(thrust::make_tuple(
                transformation_ptr, thrust::counting_iterator<size_t>(0))),
            thrust::make_zip_iterator(
                thrust::make_tuple(transformation_ptr + n * n,
                                   thrust::counting_iterator<size_t>(n * n))),
            transformation_ptr, identity_minus_A_functor<T>(n, n, lda));
        auto QTQ = thrust::device_vector<T>(n * n, (T)0.0f);
        matrix_ops::gemm(handle_, n, n, n, (T)1.0f, transformation_ptr, n, true,
                         transformation_ptr, n, false, (T)0.0f, QTQ.data(), n);
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

        // check QBQT = A_ori
        thrust::device_vector<T> d_QBQT(n * n, (T)0.0f);
        matrix_ops::gemm(handle_, n, n, n, (T)1.0f,
                         transformation_matrix.data(), n, false, d_A.data(), n,
                         false, (T)0.0f, d_QBQT.data(), n);
        matrix_ops::gemm(handle_, n, n, n, (T)-1.0f, d_QBQT.data(), n, false,
                         transformation_matrix.data(), n, true, (T)1.0f,
                         A_data.data(), n);
        if constexpr (std::is_same_v<T, float>) {
            cublasSnrm2(handle_, n * n, A_data.data().get(), 1, &norm);
            ASSERT_NEAR(norm / n, 0.0f, 1e-4);
        } else if constexpr (std::is_same_v<T, double>) {
            cublasDnrm2(handle_, n * n, A_data.data().get(), 1, &norm);
            ASSERT_NEAR(norm / n, 0.0, 1e-10);
        }
    }

    void run_sy2sb_test(size_t n, thrust::device_vector<T>& A, size_t nb = 128,
                        size_t b = 32) {
        // auto A = matrix_ops::create_symmetric_random<T>(n);
        auto lda = n, ldy = n, ldw = n;

        auto A_ori = A;

        thrust::device_vector<T> d_Y(n * n, (T)0.0f);
        thrust::device_vector<T> d_W(n * n, (T)0.0f);

        EXPECT_NO_THROW(matrix_ops::sy2sb<T>(handle_, n, A.data(), lda,
                                             d_Y.data(), ldy, d_W.data(), ldw,
                                             nb, b));

        // compute W from oriW
        computeWfromOriWY(handle_, n, b, d_W.data(), ldw, d_Y.data(), ldy);

        // Q  = I - WYT QTBQ = A  QTQ = I
        auto transformation_matrix = thrust::device_vector<T>(n * n, (T)0.0f);
        matrix_ops::gemm(handle_, n, n, n, (T)1.0f, d_W.data(), n, false,
                         d_Y.data(), n, true, (T)0.0f,
                         transformation_matrix.data(), n);
        auto transformation_ptr = transformation_matrix.data();
        thrust::transform(
            thrust::device,
            thrust::make_zip_iterator(thrust::make_tuple(
                transformation_ptr, thrust::counting_iterator<size_t>(0))),
            thrust::make_zip_iterator(
                thrust::make_tuple(transformation_ptr + n * n,
                                   thrust::counting_iterator<size_t>(n * n))),
            transformation_ptr, identity_minus_A_functor<T>(n, n, lda));
        auto QTQ = thrust::device_vector<T>(n * n, (T)0.0f);
        matrix_ops::gemm(handle_, n, n, n, (T)1.0f, transformation_ptr, n, true,
                         transformation_ptr, n, false, (T)0.0f, QTQ.data(), n);
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

        // check QBQT = A_ori
        thrust::device_vector<T> d_QBQT(n * n, (T)0.0f);
        matrix_ops::gemm(handle_, n, n, n, (T)1.0f,
                         transformation_matrix.data(), n, false, A.data(), n,
                         false, (T)0.0f, d_QBQT.data(), n);
        matrix_ops::gemm(handle_, n, n, n, (T)-1.0f, d_QBQT.data(), n, false,
                         transformation_matrix.data(), n, true, (T)1.0f,
                         A_ori.data(), n);
        if constexpr (std::is_same_v<T, float>) {
            cublasSnrm2(handle_, n * n, A_ori.data().get(), 1, &norm);
            ASSERT_NEAR(norm / n, 0.0f, 1e-4);
        } else if constexpr (std::is_same_v<T, double>) {
            cublasDnrm2(handle_, n * n, A_ori.data().get(), 1, &norm);
            ASSERT_NEAR(norm / n, 0.0, 1e-10);
        }
    }

    common::CublasHandle handle_;
};

using MyTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(Sy2sbTest, MyTypes);

TYPED_TEST(Sy2sbTest, Big) { this->run_sy2sb_test(16384); }

TYPED_TEST(Sy2sbTest, LittleBig) { this->run_sy2sb_test(4096); }

TYPED_TEST(Sy2sbTest, Basic) { this->run_sy2sb_test(256); }

TYPED_TEST(Sy2sbTest, Nb) { this->run_sy2sb_test(128); }

TYPED_TEST(Sy2sbTest, SmallNb) {
    this->run_sy2sb_test(64, (size_t)32, (size_t)16);
}

TYPED_TEST(Sy2sbTest, dist1gpu) { this->run_dist_sy2sb_test(64, 32, 16, 1); }

TYPED_TEST(Sy2sbTest, dist1gpuBasic) {
    this->run_dist_sy2sb_test(256, 32, 16, 1);
}

TYPED_TEST(Sy2sbTest, dist2gpuSmall) {
    this->run_dist_sy2sb_test(64, 32, 16, 2);
}

TYPED_TEST(Sy2sbTest, dist2gpuBasic) {
    this->run_dist_sy2sb_test(256, 32, 16, 2);
}