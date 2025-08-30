#pragma once

#include <mpi.h>
#include <nccl.h>
#include <thrust/device_vector.h>

#include <cstddef>

#include "gpu_handle_wrappers.h"

namespace matrix_ops {
namespace mpi {

// 数据分布策略（1D 列方向）
enum class DistributionType {
    Blockwise,       // 连续列块（当前实现）
    BlockCyclic1D    // 1D 列方向 block-cyclic
};

/**
 * @brief MPI 环境信息和配置类
 */
struct MpiConfig {
    int rank;
    int size;
    int local_gpu_id;
    int total_gpus;

    MpiConfig(int r, int s, int local_gpu, int total);
};

/**
 * @brief MPI sy2sb 算法的参数和资源管理类
 */
template <typename T>
class MpiSy2sbContext {
   public:
    // MPI 配置
    MpiConfig mpi_config;

    // 算法参数
    size_t n;
    size_t lda, ldw, ldy;
    size_t nb, b;

    // 分布策略与分块信息
    DistributionType dist_type{DistributionType::Blockwise};
    size_t block_size_bs{0};  // block-cyclic 的块大小（默认使用 nb）
    size_t cols_per_process;
    size_t start_col;
    size_t local_matrix_size;
    size_t local_cols{0};     // block-cyclic 下当前进程拥有的列数
    ncclDataType_t nccl_type;

    // 主机内存指针
    T* A_host;
    T* W_host;
    T* Y_host;

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

    // 工具函数：计算给定列偏移对应的MPI进程
    size_t computeProcessForColumn(size_t col_offset) const;

    // 工具函数：判断给定列是否属于当前进程
    bool isLocalColumn(size_t col_offset) const;

    // 工具函数：获取本地列索引
    size_t getLocalColumnIndex(size_t global_col) const;

    // block-cyclic 辅助：列 j 的拥有者
    inline size_t ownerOfCol(size_t j) const {
        if (dist_type == DistributionType::Blockwise) {
            return j / cols_per_process;
        }
        // BlockCyclic1D
        auto bs = block_size_bs;
        return (j / bs) % static_cast<size_t>(mpi_config.size);
    }

    // block-cyclic 辅助：列 j 在本地打包后的列号
    inline size_t localColIndex(size_t j) const {
        if (dist_type == DistributionType::Blockwise) {
            return j - start_col;
        }
        // BlockCyclic1D
        auto bs = block_size_bs;
        return (j / bs) / static_cast<size_t>(mpi_config.size) * bs + (j % bs);
    }

    // 计算当前进程拥有的列数（block-cyclic）
    size_t computeLocalCols() const;

    // 设备指针映射：返回本地打包列 j 的起始指针（列主序）
    inline thrust::device_ptr<T> ptrLocalCol(thrust::device_vector<T>& buf,
                                             size_t j) {
        return buf.data() + localColIndex(j) * n;
    }

    // 设备指针映射：返回全局 (row, col) 在本地打包缓冲区中的指针
    inline thrust::device_ptr<T> ptrLocalRC(thrust::device_vector<T>& buf,
                                            size_t row, size_t col) {
        return buf.data() + localColIndex(col) * n + row;
    }

   private:
    void initGpuResources();
    void initCommunication();
    void allocateGpuMemory();
    void copyHostToGpu();
    void cleanup();
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

}  // namespace mpi
}  // namespace matrix_ops
