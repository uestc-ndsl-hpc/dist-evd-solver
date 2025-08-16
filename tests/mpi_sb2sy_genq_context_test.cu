#include <gtest/gtest.h>
#include <mpi.h>
#include <nccl.h>
#include <thrust/device_vector.h>

#include "matrix_ops_mpi.cuh"

using namespace matrix_ops::mpi;

class MpiSb2syGenQContextTest : public ::testing::Test {
   protected:
    void SetUp() override {
        // Initialize MPI (假设在测试环境中已经初始化)
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        // 设置测试参数
        n = 128;
        nb = 32;
        b = 8;
        lda = ldw = ldy = n;
        
        // 创建 MPI 配置
        mpi_config = std::make_unique<MpiConfig>(rank, size, 0, size);
    }

    void TearDown() override {
        // 清理资源
    }

    size_t n, nb, b, lda, ldw, ldy;
    std::unique_ptr<MpiConfig> mpi_config;
};

TEST_F(MpiSb2syGenQContextTest, ConstructorAndDestructor) {
    // 创建模拟的 Sy2sbResultBuffers
    Sy2sbResultBuffers<float> buffers;
    buffers.n = n;
    buffers.nb = nb;
    buffers.b = b;
    buffers.lda = lda;
    buffers.ldw = ldw;
    buffers.ldy = ldy;
    
    // 分配 GPU 内存
    size_t cols_per_process = n / mpi_config->size;
    size_t local_matrix_size = cols_per_process * n;
    
    buffers.A.resize(local_matrix_size);
    buffers.W.resize(local_matrix_size);
    buffers.Y.resize(local_matrix_size);
    
    // 初始化通信器为空（在实际使用中会被正确设置）
    buffers.nccl_comm = nullptr;
    buffers.sub_comm_groups.clear();
    buffers.sub_mpi_comms.clear();
    
    // 测试构造函数
    EXPECT_NO_THROW({
        MpiSb2syGenQContext<float> context(*mpi_config, buffers);
        
        // 验证基本参数
        EXPECT_EQ(context.n, n);
        EXPECT_EQ(context.nb, nb);
        EXPECT_EQ(context.b, b);
        EXPECT_EQ(context.lda, lda);
        EXPECT_EQ(context.ldw, ldw);
        EXPECT_EQ(context.ldy, ldy);
        
        // 验证分块信息
        EXPECT_EQ(context.cols_per_process, cols_per_process);
        EXPECT_EQ(context.start_col, mpi_config->rank * cols_per_process);
        EXPECT_EQ(context.local_matrix_size, local_matrix_size);
        
        // 验证 GPU 内存已分配
        EXPECT_EQ(context.gpu_W.size(), local_matrix_size);
        EXPECT_EQ(context.gpu_Y.size(), local_matrix_size);
        EXPECT_EQ(context.gpu_work.size(), 2 * n * nb);
        
        // 验证 NCCL 数据类型
        EXPECT_EQ(context.nccl_type, ncclFloat32);
        
        // 析构函数会在作用域结束时自动调用
    });
}

TEST_F(MpiSb2syGenQContextTest, DoubleTypeSupport) {
    // 测试 double 类型支持
    Sy2sbResultBuffers<double> buffers;
    buffers.n = n;
    buffers.nb = nb;
    buffers.b = b;
    buffers.lda = lda;
    buffers.ldw = ldw;
    buffers.ldy = ldy;
    
    size_t cols_per_process = n / mpi_config->size;
    size_t local_matrix_size = cols_per_process * n;
    
    buffers.A.resize(local_matrix_size);
    buffers.W.resize(local_matrix_size);
    buffers.Y.resize(local_matrix_size);
    
    buffers.nccl_comm = nullptr;
    buffers.sub_comm_groups.clear();
    buffers.sub_mpi_comms.clear();
    
    EXPECT_NO_THROW({
        MpiSb2syGenQContext<double> context(*mpi_config, buffers);
        
        // 验证 double 类型的 NCCL 数据类型
        EXPECT_EQ(context.nccl_type, ncclFloat64);
    });
}

TEST_F(MpiSb2syGenQContextTest, MoveSemantics) {
    // 测试移动语义是否正确工作
    Sy2sbResultBuffers<float> buffers;
    buffers.n = n;
    buffers.nb = nb;
    buffers.b = b;
    buffers.lda = lda;
    buffers.ldw = ldw;
    buffers.ldy = ldy;
    
    size_t cols_per_process = n / mpi_config->size;
    size_t local_matrix_size = cols_per_process * n;
    
    // 填充一些测试数据
    buffers.A.resize(local_matrix_size);
    buffers.W.resize(local_matrix_size, 1.0f);
    buffers.Y.resize(local_matrix_size, 2.0f);
    
    buffers.nccl_comm = nullptr;
    buffers.sub_comm_groups.clear();
    buffers.sub_mpi_comms.clear();
    
    // 记录原始大小
    size_t original_w_size = buffers.W.size();
    size_t original_y_size = buffers.Y.size();
    
    EXPECT_NO_THROW({
        MpiSb2syGenQContext<float> context(*mpi_config, buffers);
        
        // 验证数据已被移动
        EXPECT_EQ(context.gpu_W.size(), original_w_size);
        EXPECT_EQ(context.gpu_Y.size(), original_y_size);
        
        // 验证原始缓冲区已被清空（移动后）
        EXPECT_EQ(buffers.W.size(), 0);
        EXPECT_EQ(buffers.Y.size(), 0);
        EXPECT_EQ(buffers.nccl_comm, nullptr);
        EXPECT_TRUE(buffers.sub_comm_groups.empty());
        EXPECT_TRUE(buffers.sub_mpi_comms.empty());
    });
}

// 主函数
int main(int argc, char** argv) {
    // 初始化 MPI
    MPI_Init(&argc, &argv);
    
    // 初始化 Google Test
    ::testing::InitGoogleTest(&argc, argv);
    
    // 运行测试
    int result = RUN_ALL_TESTS();
    
    // 清理 MPI
    MPI_Finalize();
    
    return result;
}