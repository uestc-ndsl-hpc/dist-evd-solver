#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cstddef>

#include "matrix_ops.cuh"

template <typename T>
class MatrixCopyTest : public ::testing::Test {
   protected:
    void run_d2d_copy_test(size_t m, size_t n, size_t src_ld, size_t dst_ld) {
        thrust::host_vector<T> h_src(src_ld * n);
        for (size_t i = 0; i < src_ld * n; ++i) {
            h_src[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
        }

        thrust::device_vector<T> d_src = h_src;
        thrust::device_vector<T> d_dst(dst_ld * n, 0);

        matrix_ops::matrix_copy<thrust::device_ptr<T>, thrust::device_ptr<T>, T>(
            d_src.data(), src_ld, d_dst.data(), dst_ld, m, n);

        thrust::host_vector<T> h_dst = d_dst;

        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < m; ++i) {
                EXPECT_EQ(h_src[j * src_ld + i], h_dst[j * dst_ld + i]);
            }
        }
    }

    void run_d2h_copy_test(size_t m, size_t n, size_t src_ld, size_t dst_ld) {
        thrust::host_vector<T> h_src(src_ld * n);
        for (size_t i = 0; i < src_ld * n; ++i) {
            h_src[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
        }

        thrust::device_vector<T> d_src = h_src;
        thrust::host_vector<T> h_dst(dst_ld * n, 0);

        matrix_ops::matrix_copy<thrust::device_ptr<T>, T*, T>(
            d_src.data(), src_ld, h_dst.data(), dst_ld, m, n);

        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < m; ++i) {
                EXPECT_EQ(h_src[j * src_ld + i], h_dst[j * dst_ld + i]);
            }
        }
    }
    
    void run_h2d_copy_test(size_t m, size_t n, size_t src_ld, size_t dst_ld) {
        thrust::host_vector<T> h_src(src_ld * n);
        for (size_t i = 0; i < src_ld * n; ++i) {
            h_src[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
        }

        thrust::device_vector<T> d_dst(dst_ld * n, 0);
        
        matrix_ops::matrix_copy<T*, thrust::device_ptr<T>, T>(h_src.data(),
                                                              src_ld, d_dst.data(),
                                                              dst_ld, m, n);

        thrust::host_vector<T> h_dst = d_dst;
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < m; ++i) {
                EXPECT_EQ(h_src[j * src_ld + i], h_dst[j * dst_ld + i]);
            }
        }
    }

    void run_h2h_copy_test(size_t m, size_t n, size_t src_ld, size_t dst_ld) {
        thrust::host_vector<T> h_src(src_ld * n);
        for (size_t i = 0; i < src_ld * n; ++i) {
            h_src[i] = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
        }

        thrust::host_vector<T> h_dst(dst_ld * n, 0);

        matrix_ops::matrix_copy<T*, T*, T>(h_src.data(), src_ld, h_dst.data(),
                                           dst_ld, m, n);
        
        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < m; ++i) {
                EXPECT_EQ(h_src[j * src_ld + i], h_dst[j * dst_ld + i]);
            }
        }
    }
};

using MyTypes = ::testing::Types<float, double>;

TYPED_TEST_SUITE(MatrixCopyTest, MyTypes);

TYPED_TEST(MatrixCopyTest, D2D) {
    this->run_d2d_copy_test(64, 32, 64, 64);
    this->run_d2d_copy_test(65, 33, 65, 65);
    this->run_d2d_copy_test(64, 32, 70, 80);
    this->run_d2d_copy_test(65, 33, 75, 85);
}

TYPED_TEST(MatrixCopyTest, D2H) {
    this->run_d2h_copy_test(64, 32, 64, 64);
    this->run_d2h_copy_test(65, 33, 65, 65);
    this->run_d2h_copy_test(64, 32, 70, 80);
    this->run_d2h_copy_test(65, 33, 75, 85);
}

TYPED_TEST(MatrixCopyTest, H2D) {
    this->run_h2d_copy_test(64, 32, 64, 64);
    this->run_h2d_copy_test(65, 33, 65, 65);
    this->run_h2d_copy_test(64, 32, 70, 80);
    this->run_h2d_copy_test(65, 33, 75, 85);
}

TYPED_TEST(MatrixCopyTest, H2H) {
    this->run_h2h_copy_test(64, 32, 64, 64);
    this->run_h2h_copy_test(65, 33, 65, 65);
    this->run_h2h_copy_test(64, 32, 70, 80);
    this->run_h2h_copy_test(65, 33, 75, 85);
} 