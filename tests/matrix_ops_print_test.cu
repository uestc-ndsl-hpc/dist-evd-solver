#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/system/cuda/execution_policy.h>  // 添加 CUDA 执行策略头文件

#include <cstddef>

#include "matrix_ops.cuh"

// 1. 创建一个测试夹具类
class MatrixPrintTest : public ::testing::Test {
   protected:
    // 在这里定义所有测试用例共享的成员变量
    size_t m = 2;
    size_t n = 3;

    void SetUp() override {
        // 测试前重置并初始化CUDA设备
        cudaDeviceReset();
    }

    void TearDown() override {
        // 测试后同步设备并重置
        cudaDeviceSynchronize();
        cudaDeviceReset();  // 确保测试后状态重置
    }
};

// 先测试 float 类型，看是否会影响后续的 double 类型测试
TEST_F(MatrixPrintTest, SequenceMatrixPrintFloat) {
    // 使用thrust创建设备向量
    thrust::device_vector<float> d_vec(m * n);
    thrust::sequence(d_vec.begin(), d_vec.end());

    // 打印矩阵
    EXPECT_NO_THROW(matrix_ops::print(d_vec, m, n, "SequenceMatrixPrintFloat"));
}

TEST_F(MatrixPrintTest, SequenceMatrixPrintDouble) {
    // 使用thrust创建设备向量
    thrust::device_vector<double> d_vec(m * n);
    thrust::sequence(d_vec.begin(), d_vec.end());

    // 打印矩阵
    EXPECT_NO_THROW(
        matrix_ops::print(d_vec, m, n, "SequenceMatrixPrintDouble"));
}

TEST_F(MatrixPrintTest, SequenceHostMatrixPrintFloat) {
    // 使用thrust创建设备向量
    thrust::device_vector<float> d_vec(m * n);
    thrust::sequence(d_vec.begin(), d_vec.end());

    auto h_vec = thrust::host_vector<float>(d_vec.begin(), d_vec.end());

    // 打印矩阵
    EXPECT_NO_THROW(
        matrix_ops::print(h_vec.data(), m, n, "SequenceHostMatrixPrintFloat"));
}

TEST_F(MatrixPrintTest, SequenceHostMatrixPrintDouble) {
    // 使用thrust创建设备向量
    thrust::device_vector<double> d_vec(m * n);
    thrust::sequence(d_vec.begin(), d_vec.end());

    auto h_vec = thrust::host_vector<double>(d_vec.begin(), d_vec.end());

    // 打印矩阵
    EXPECT_NO_THROW(
        matrix_ops::print(h_vec.data(), m, n, "SequenceHostMatrixPrintDouble"));
}
