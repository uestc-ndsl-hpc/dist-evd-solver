#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "gpu_handle_wrappers.h"
#include "matrix_ops.cuh"
#include "sy2sb_panelqr.cuh"

namespace {

// 测试WY构造过程的正确性
class SY2SBWYConstructionTest : public ::testing::Test {
protected:
    void SetUp() override {
        cublasHandle = common::CublasHandle();
        cusolverHandle = common::CusolverDnHandle();
    }

    common::CublasHandle cublasHandle;
    common::CusolverDnHandle cusolverHandle;
};

// 测试I-WYT乘积的正确性
TEST_F(SY2SBWYConstructionTest, IWYTConstructionTest) {
    const size_t m = 8;  // 面板行数
    const size_t n = 4;  // 面板列数 (块大小b)
    
    // 创建测试矩阵A
    thrust::host_vector<double> h_A(m * n);
    for (size_t i = 0; i < m * n; ++i) {
        h_A[i] = static_cast<double>(i + 1) * 0.1;
    }
    
    thrust::device_vector<double> d_A = h_A;
    thrust::device_vector<double> d_R(m * n);
    thrust::device_vector<double> d_W(m * n);
    
    // 执行panelQR来构造W和R
    matrix_ops::internal::sy2sb::panelQR<double>(
        cublasHandle, cusolverHandle, m, n,
        d_A.data(), m,
        d_R.data(), m,
        d_W.data(), m
    );
    
    // 将W和A复制回host进行验证
    thrust::host_vector<double> h_W = d_W;
    thrust::host_vector<double> h_Y = d_A;  // Y就是变换后的A
    
    // 计算I - Y * W^T
    thrust::host_vector<double> h_I_YWT(m * m);
    
    // 初始化为单位矩阵
    for (size_t i = 0; i < m * m; ++i) {
        h_I_YWT[i] = 0.0;
    }
    for (size_t i = 0; i < m; ++i) {
        h_I_YWT[i + i * m] = 1.0;
    }
    
    // 计算Q = I - W * Y^T (经济型QR分解，Q是m×n)
    thrust::host_vector<double> h_Q(m * n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += h_W[i + k * m] * h_Y[j + k * m];
            }
            h_Q[i + j * m] = (i == j && i < n) ? 1.0 - sum : -sum;
        }
    }
    
    // 验证Q^T * Q = I (n×n单位矩阵)
    thrust::host_vector<double> h_QTQ(n * n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < m; ++k) {
                sum += h_Q[k + i * m] * h_Q[k + j * m];
            }
            h_QTQ[i + j * n] = sum;
        }
    }
    
    // 检查Q^T*Q是否接近n×n单位矩阵
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(h_QTQ[i + j * n], expected, 1e-8) 
                << "Q^T*Q[" << i << "," << j << "] should be " << expected 
                << " but got " << h_QTQ[i + j * n];
        }
    }
}

// 测试W和Y的维度是否正确
TEST_F(SY2SBWYConstructionTest, WYDimensionsTest) {
    const size_t m = 10;
    const size_t n = 3;
    
    thrust::device_vector<double> d_A(m * n, 1.0);
    thrust::device_vector<double> d_R(m * n);
    thrust::device_vector<double> d_W(m * n);
    
    EXPECT_NO_THROW({
        matrix_ops::internal::sy2sb::panelQR<double>(
            cublasHandle, cusolverHandle, m, n,
            d_A.data(), m,
            d_R.data(), m,
            d_W.data(), m
        );
    });
    
    // 检查W和Y的维度是否正确
    // W应该是m×n，Y也应该是m×n
    EXPECT_EQ(d_W.size(), m * n);
    EXPECT_EQ(d_A.size(), m * n);
}

// 测试W和Y的正交性验证
TEST_F(SY2SBWYConstructionTest, WYOrthogonalityTest) {
    const size_t m = 6;
    const size_t n = 3;
    
    // 创建随机测试矩阵
    thrust::host_vector<double> h_A(m * n);
    for (size_t i = 0; i < m * n; ++i) {
        h_A[i] = static_cast<double>(rand()) / RAND_MAX;
    }
    
    thrust::device_vector<double> d_A = h_A;
    thrust::device_vector<double> d_R(m * n);
    thrust::device_vector<double> d_W(m * n);
    
    // 执行panelQR
    matrix_ops::internal::sy2sb::panelQR<double>(
        cublasHandle, cusolverHandle, m, n,
        d_A.data(), m,
        d_R.data(), m,
        d_W.data(), m
    );
    
    // 将数据复制回host
    thrust::host_vector<double> h_W = d_W;
    thrust::host_vector<double> h_Y = d_A;
    
    // 计算I - Y * W^T
    thrust::host_vector<double> h_Q(m * m);
    
    // 初始化为单位矩阵
    for (size_t i = 0; i < m * m; ++i) {
        h_Q[i] = 0.0;
    }
    for (size_t i = 0; i < m; ++i) {
        h_Q[i + i * m] = 1.0;
    }
    
    // 计算Q = I - W * Y^T (经济型QR分解，Q是m×n)
    thrust::host_vector<double> h_Q_mn(m * n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += h_W[i + k * m] * h_Y[j + k * m];
            }
            h_Q_mn[i + j * m] = (i == j && i < n) ? 1.0 - sum : -sum;
        }
    }
    
    // 验证Q^T * Q = I (n×n单位矩阵)
    thrust::host_vector<double> h_QTQ(n * n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < m; ++k) {
                sum += h_Q_mn[k + i * m] * h_Q_mn[k + j * m];
            }
            h_QTQ[i + j * n] = sum;
        }
    }
    
    // 检查Q^T*Q是否接近n×n单位矩阵
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            EXPECT_NEAR(h_QTQ[i + j * n], expected, 1e-8) 
                << "Q^T*Q[" << i << "," << j << "] should be " << expected 
                << " but got " << h_QTQ[i + j * n];
        }
    }
}

// 测试I-WYT与TSQR Q矩阵的一致性
TEST_F(SY2SBWYConstructionTest, IWYTvsTSQRConsistencyTest) {
    const size_t m = 8;  // 矩阵行数
    const size_t n = 4;  // 矩阵列数
    
    // 创建随机测试矩阵
    thrust::host_vector<double> h_A(m * n);
    for (size_t i = 0; i < m * n; ++i) {
        h_A[i] = static_cast<double>(rand()) / RAND_MAX;
    }
    
    thrust::device_vector<double> d_A = h_A;
    thrust::device_vector<double> d_A_copy = h_A;  // 复制一份用于TSQR
    thrust::device_vector<double> d_R(m * n);
    thrust::device_vector<double> d_W(m * n);
    thrust::device_vector<double> d_Q_tsqr(m * m);
    thrust::device_vector<double> d_R_tsqr(n * n);
    
    // 方法1: 使用panelQR计算I-WYT形式的Q
    matrix_ops::internal::sy2sb::panelQR<double>(
        cublasHandle, cusolverHandle, m, n,
        d_A.data(), m,
        d_R.data(), m,
        d_W.data(), m
    );
    
    // 计算I-YWT
    thrust::host_vector<double> h_W = d_W;
    thrust::host_vector<double> h_Y = d_A;  // Y是变换后的A
    thrust::host_vector<double> h_Q_wy(m * m);
    
    // 初始化为单位矩阵
    for (size_t i = 0; i < m * m; ++i) {
        h_Q_wy[i] = 0.0;
    }
    for (size_t i = 0; i < m; ++i) {
        h_Q_wy[i + i * m] = 1.0;
    }
    
    // 计算Q = I - W * Y^T (经济型QR分解，Q是m×n)
    thrust::host_vector<double> h_Q_wy_mn(m * n);
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += h_W[i + k * m] * h_Y[j + k * m];
            }
            h_Q_wy_mn[i + j * m] = (i == j && i < n) ? 1.0 - sum : -sum;
        }
    }
    
    // 方法2: 使用TSQR计算Q矩阵 (经济型QR分解，Q是m×n)
    // 注意：tsqr会覆盖输入矩阵A，所以先复制到Q矩阵
    thrust::device_vector<double> d_Q_tsqr_input = d_A_copy;
    thrust::device_vector<double> d_R_tsqr_actual(n * n);
    matrix_ops::tsqr(
        cublasHandle, m, n,
        d_Q_tsqr_input.data(),
        d_R_tsqr_actual.data(), m, n
    );
    
    // 将结果复制到Q矩阵
    d_Q_tsqr = d_Q_tsqr_input;
    
    // 将TSQR的Q矩阵复制回host
    thrust::host_vector<double> h_Q_tsqr = d_Q_tsqr;
    
    // 验证两种方法得到的Q矩阵是否一致 (m×n矩阵)
    for (size_t i = 0; i < m * n; ++i) {
        EXPECT_NEAR(h_Q_wy_mn[i], h_Q_tsqr[i], 1e-8)
            << "I-WYT Q and TSQR Q differ at position " << i
            << ": I-WYT=" << h_Q_wy_mn[i] << ", TSQR=" << h_Q_tsqr[i];
    }
    
    // 额外验证：检查两个Q矩阵作用于原矩阵是否得到相同的R (n×n上三角矩阵)
    thrust::host_vector<double> h_A_host = h_A;
    thrust::host_vector<double> h_R_wy(n * n, 0.0);
    
    // 计算Q^T * A，应该得到上三角矩阵R
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum_wy = 0.0;
            for (size_t k = 0; k < m; ++k) {
                sum_wy += h_Q_wy_mn[k + i * m] * h_A_host[k + j * m];
            }
            h_R_wy[i + j * n] = sum_wy;
        }
    }
    
    // 验证I-YWT得到的R与TSQR得到的R是否一致
    thrust::host_vector<double> h_R_tsqr_actual = d_R_tsqr_actual;
    for (size_t i = 0; i < n * n; ++i) {
        EXPECT_NEAR(h_R_wy[i], h_R_tsqr_actual[i], 1e-8)
            << "R matrices from I-WYT and TSQR differ at position " << i;
    }
}

} // namespace