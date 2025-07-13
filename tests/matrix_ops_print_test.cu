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
        cudaError_t err = cudaSetDevice(0);  // 使用设备 0
        ASSERT_EQ(err, cudaSuccess) << "无法设置 CUDA 设备: " << cudaGetErrorString(err);
    }
    
    void TearDown() override {
        // 测试后同步设备并重置
        cudaDeviceSynchronize();
        cudaDeviceReset();  // 确保测试后状态重置
        
        // 强制重置 Thrust 使用的设备
        thrust::cuda::par.on(0);
    }
};

// 先测试 float 类型，看是否会影响后续的 double 类型测试
TEST_F(MatrixPrintTest, SequenceMatrixPrintFloat) {
    // 重置设备以确保干净状态
    cudaDeviceReset();
    cudaError_t err = cudaSetDevice(0);
    ASSERT_EQ(err, cudaSuccess) << "无法设置 CUDA 设备: " << cudaGetErrorString(err);
    
    // 显式指定使用设备 0
    auto policy = thrust::cuda::par.on(0);
    
    // 创建独立的上下文
    float* d_data = nullptr;
    size_t bytes = m * n * sizeof(float);
    err = cudaMalloc(&d_data, bytes);
    ASSERT_EQ(err, cudaSuccess) << "无法分配设备内存: " << cudaGetErrorString(err);
    
    // 使用thrust创建设备向量
    thrust::device_vector<float> d_vec(m * n);
    thrust::sequence(policy, d_vec.begin(), d_vec.end());
    
    // 打印矩阵
    EXPECT_NO_THROW(
        matrix_ops::print(d_vec, m, n, "SequenceMatrixPrintFloat"));
    
    cudaDeviceSynchronize();
    
    // 重置 thrust 使用的设备
    thrust::cuda::par.on(0);
}

TEST_F(MatrixPrintTest, SequenceMatrixPrintDouble) {
    // 单独测试中重置设备
    cudaDeviceReset();
    cudaError_t err = cudaSetDevice(0);
    ASSERT_EQ(err, cudaSuccess) << "无法设置 CUDA 设备: " << cudaGetErrorString(err);
    
    // 显式指定使用设备 0
    auto policy = thrust::cuda::par.on(0);
    
    // 创建独立的上下文
    double* d_data = nullptr;
    size_t bytes = m * n * sizeof(double);
    err = cudaMalloc(&d_data, bytes);
    ASSERT_EQ(err, cudaSuccess) << "无法分配设备内存: " << cudaGetErrorString(err);
    
    // 使用thrust创建设备向量
    thrust::device_vector<double> d_vec(m * n);
    thrust::sequence(policy, d_vec.begin(), d_vec.end());
    
    // 打印矩阵
    EXPECT_NO_THROW(
        matrix_ops::print(d_vec, m, n, "SequenceMatrixPrintDouble"));

    cudaDeviceSynchronize();
    
    // 重置 thrust 使用的设备
    thrust::cuda::par.on(0);
}

