#pragma once

#include <mpi.h>
#include <nccl.h>
#include <thrust/device_vector.h>

#include <cstddef>

#include "gpu_handle_wrappers.h"

namespace matrix_ops {
namespace mpi {

/**
 * @brief MPI 环境信息和配置类
 */
struct MpiConfig {
    int rank;
    int size;
    int local_gpu_id;
    int total_gpus;

    MpiConfig(int r, int s, int local_gpu, int total);

    MpiConfig(const MpiConfig& mpi_config)
        : rank(mpi_config.rank),
          size(mpi_config.size),
          local_gpu_id(mpi_config.local_gpu_id),
          total_gpus(mpi_config.total_gpus) {}
};

/**
 * @brief MPI sy2sb 算法的参数和资源管理类
 */

// 用于保存 sy2sb 计算结果缓冲区的结构体
template <typename T>
struct Sy2sbResultBuffers {
    // GPU 显存
    thrust::device_vector<T> A;
    thrust::device_vector<T> W;
    thrust::device_vector<T> Y;
    // 算法参数
    size_t n;
    size_t lda, ldw, ldy;
    size_t nb, b;
    // GPU 资源句柄
    common::CublasHandle cublas_handle;
    common::CusolverDnHandle cusolver_handle;
    cudaStream_t stream;
    // 通信器组
    ncclComm_t nccl_comm;                     // 主通信器
    std::vector<ncclComm_t> sub_comm_groups;  // 层次化子通信组
    std::vector<MPI_Comm> sub_mpi_comms;      // 对应的MPI子通信器
};

template <typename T>
class MpiSb2trContext {
   public:
    // MPI 配置
    MpiConfig mpi_config;

    // 算法参数
    size_t n;
    size_t b;

    size_t ldSubA, ldU;

    // 分块信息
    size_t cols_cur_node_process;

    // GPU 资源 (每个进程一个 GPU)
    cudaStream_t stream;

    // GPU 显存信息
    thrust::device_vector<T> gpu_A;

    T* gpu_subA;
    thrust::device_vector<T> gpu_U;

    // NVSHmem 内存信息, 使用NVShmem进行通信
    int* prePEWriteCom;
    int* nextPEWriteTailSweepProcRow;

    // GPU 内存信息, 用于同1PE内部的多趟BC之间进行通信
    thrust::device_vector<int> com;

    MpiSb2trContext(const MpiConfig& config, Sy2sbResultBuffers<T>& buffers);

    ~MpiSb2trContext();

    // 需要将sy2sb的Q复制到拷贝到tr2sb中
    void copyMatrixA2SubA();
};

template <typename T>
class MpiSy2sbContext {
   public:
    // MPI 配置
    MpiConfig mpi_config;

    // 算法参数
    size_t n;
    size_t lda, ldw, ldy;
    size_t nb, b;

    // 分块信息
    size_t cols_per_process;
    size_t start_col;
    size_t local_matrix_size;
    ncclDataType_t nccl_type;

    // 主机内存指针
    T* A_host;

    // GPU 资源 (每个进程一个 GPU)
    common::CublasHandle cublas_handle;
    common::CusolverDnHandle cusolver_handle;
    cudaStream_t stream;

    // GPU 内存
    thrust::device_vector<T> gpu_A;
    thrust::device_vector<T> gpu_W;
    thrust::device_vector<T> gpu_Y;
    thrust::device_vector<T> gpu_R;
    thrust::device_vector<T> gpu_Z;
    thrust::device_vector<T> gpu_work;
    thrust::device_vector<T> gpu_oriA;

    // NCCL 通信器
    ncclComm_t nccl_comm;

    // 层次化子通信组：支持流水线并行
    // sub_comm_groups[i] 表示从进程i开始到最后的通信组 [i, i+1, ..., size-1]
    std::vector<ncclComm_t> sub_comm_groups;

    // 对应的MPI子通信器：用于MPI_Barrier等同步操作
    std::vector<MPI_Comm> sub_mpi_comms;

    MpiSy2sbContext(const MpiConfig& config, size_t matrix_n, T* A,
                    size_t lda_val, T* W, size_t ldw_val, T* Y, size_t ldy_val,
                    size_t nb_val, size_t b_val);

    ~MpiSy2sbContext();

    // 释放主要的 GPU 缓冲区，将所有权转移给调用者
    Sy2sbResultBuffers<T> release_sy2sb_buffers();

    // 工具函数：计算给定列偏移对应的MPI进程
    size_t computeProcessForColumn(size_t col_offset) const;

    // 工具函数：判断给定列是否属于当前进程
    bool isLocalColumn(size_t col_offset) const;

    // 工具函数：获取本地列索引
    size_t getLocalColumnIndex(size_t global_col) const;

   private:
    void initGpuResources();
    void initCommunication();
    void allocateGpuMemory();
    void copyHostToGpu();
    void cleanup();
};

template <typename T>
class MpiSb2syGenQContext {
   public:
    // MPI 配置
    MpiConfig mpi_config;

    // 算法参数
    size_t n;
    size_t lda, ldw, ldy;
    size_t nb, b;

    // 分块信息
    size_t cols_per_process;
    size_t start_col;
    size_t local_matrix_size;
    ncclDataType_t nccl_type;
    std::vector<size_t> q_cols;

    // GPU 资源 (每个进程一个 GPU)
    common::CublasHandle cublas_handle;
    common::CusolverDnHandle cusolver_handle;
    cudaStream_t stream;

    // GPU 显存信息
    thrust::device_vector<T> gpu_W;
    thrust::device_vector<T> gpu_Y;
    thrust::device_vector<T> gpu_work;
    thrust::device_vector<T> gpu_Q;

    thrust::device_vector<T> gpu_W_rec;
    thrust::device_vector<T> gpu_Y_rec;

    // NCCL 通信器
    ncclComm_t nccl_comm;
    std::vector<ncclComm_t> sub_comm_groups;  // 层次化子通信组
    std::vector<MPI_Comm> sub_mpi_comms;      // 对应的MPI子通信器

    MpiSb2syGenQContext(const MpiConfig& config,
                        Sy2sbResultBuffers<T>& buffers);

    ~MpiSb2syGenQContext();

    void initializeQMatrix();
};

namespace internal {
// 内部函数声明
template <typename T>
void sy2sb_recursive_mpi(size_t recursive_depth, MpiSy2sbContext<T>& ctx);

}  // namespace internal

/**
 * @brief MPI 版本的 sy2sb 主函数
 */
template <typename T>
void sy2sb(const MpiConfig& mpi_config, size_t n, T* A, size_t lda, T* W,
           size_t ldw, T* Y, size_t ldy, size_t nb = 64, size_t b = 16);

/**
 * @brief MPI 版本的 sy2sb 主函数（使用预创建的上下文）
 */
template <typename T>
void sy2sb(MpiSy2sbContext<T>& ctx);

/**
 * @brief MPI 版本的 sb2sy GenQ 主函数
 */
template <typename T>
void sb2syGenQ(MpiSb2syGenQContext<T>& context);

}  // namespace mpi
}  // namespace matrix_ops
