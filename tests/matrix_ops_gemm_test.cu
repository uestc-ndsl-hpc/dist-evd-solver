#include <gpu_handle_wrappers.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>

#include <matrix_ops.cuh>

TEST(MatrixGemmTest, AllOnesMatrixMultiplicationFloat) {
    // 初始化CUBLAS句柄
    common::CublasHandle cublasHandle;

    // 测试参数：2x3矩阵 × 3x4矩阵 = 2x4矩阵
    using T = float;
    const size_t m = 2, n = 4, k = 3;
    const T alpha = 1.0f, beta = 0.0f;

    // 创建全1矩阵（设备端）
    thrust::device_vector<T> A(m * k, 1.0f);
    thrust::device_vector<T> B(k * n, 1.0f);
    thrust::device_vector<T> C(m * n, 0.0f);

    // 执行GEMM运算: C = α*A*B + β*C
    matrix_ops::matrix_gemm(cublasHandle, m, n, k, alpha,
                            thrust::device_pointer_cast(A.data()), m, false,
                            thrust::device_pointer_cast(B.data()), k, false,
                            beta, thrust::device_pointer_cast(C.data()), m);

    EXPECT_NO_THROW(matrix_ops::print(thrust::device_pointer_cast(C.data()), m,
                                      n, "result matrix"));

    // 验证结果（全1矩阵相乘结果应为每个元素等于k）
    thrust::host_vector<T> h_C(C);
    for (size_t i = 0; i < m * n; ++i) {
        EXPECT_NEAR(h_C[i], k, 1e-5) << "矩阵乘法结果验证失败 at index " << i;
    }
}

TEST(MatrixGemmTest, AllOnesMatrixMultiplicationDouble) {
    // 初始化CUBLAS句柄
    common::CublasHandle cublasHandle;

    // 测试参数：2x3矩阵 × 3x4矩阵 = 2x4矩阵
    using T = double;
    const size_t m = 2, n = 4, k = 3;
    const T alpha = 1.0f, beta = 0.0f;

    // 创建全1矩阵（设备端）
    thrust::device_vector<T> A(m * k, 1.0f);
    thrust::device_vector<T> B(k * n, 1.0f);
    thrust::device_vector<T> C(m * n, 0.0f);

    // 执行GEMM运算: C = α*A*B + β*C
    matrix_ops::matrix_gemm(cublasHandle, m, n, k, alpha,
                            thrust::device_pointer_cast(A.data()), m, false,
                            thrust::device_pointer_cast(B.data()), k, false,
                            beta, thrust::device_pointer_cast(C.data()), m);

    EXPECT_NO_THROW(matrix_ops::print(thrust::device_pointer_cast(C.data()), m,
                                      n, "result matrix"));

    // 验证结果（全1矩阵相乘结果应为每个元素等于k）
    thrust::host_vector<T> h_C(C);
    for (size_t i = 0; i < m * n; ++i) {
        EXPECT_NEAR(h_C[i], k, 1e-5) << "矩阵乘法结果验证失败 at index " << i;
    }
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}