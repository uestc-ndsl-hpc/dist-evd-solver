#include <mpi.h>
#include <nccl.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <stdexcept>

#include "fmt/format.h"
#include "log.h"
#include "matrix_ops.cuh"
#include "matrix_ops_mpi.cuh"
#include "sy2sb_panelqr.cuh"

namespace matrix_ops {
namespace mpi {

// 构造函数实现
MpiConfig::MpiConfig(int r, int s, int local_gpu, int total)
    : rank(r), size(s), local_gpu_id(local_gpu), total_gpus(total) {}

// MpiSy2sbContext 类方法实现
template <typename T>
MpiSy2sbContext<T>::MpiSy2sbContext(const MpiConfig& config, size_t matrix_n,
                                    T* A, size_t lda_val, T* W, size_t ldw_val,
                                    T* Y, size_t ldy_val, size_t nb_val,
                                    size_t b_val)
    : mpi_config(config),
      n(matrix_n),
      lda(lda_val),
      ldw(ldw_val),
      ldy(ldy_val),
      nb(nb_val),
      b(b_val),
      A_host(A),
      W_host(W),
      Y_host(Y) {
    // 计算分块信息
    cols_per_process = n / mpi_config.size;
    start_col = mpi_config.rank * cols_per_process;
    local_matrix_size = cols_per_process * n;
    if constexpr (std::is_same_v<T, float>) {
        nccl_type = ncclFloat32;
    } else if constexpr (std::is_same_v<T, double>) {
        nccl_type = ncclFloat64;
    }

    initGpuResources();
    initCommunication();
    allocateGpuMemory();
}

template <typename T>
MpiSy2sbContext<T>::~MpiSy2sbContext() {
    cleanup();
}

// 工具函数：计算给定列偏移对应的MPI进程
template <typename T>
size_t MpiSy2sbContext<T>::computeProcessForColumn(size_t col_offset) const {
    return col_offset / cols_per_process;
}

// 工具函数：判断给定列是否属于当前进程
template <typename T>
bool MpiSy2sbContext<T>::isLocalColumn(size_t col_offset) const {
    return computeProcessForColumn(col_offset) ==
           static_cast<size_t>(mpi_config.rank);
}

// 工具函数：获取本地列索引
template <typename T>
size_t MpiSy2sbContext<T>::getLocalColumnIndex(size_t global_col) const {
    if (!isLocalColumn(global_col)) {
        throw std::out_of_range("Column is not local to this process");
    }
    return global_col - start_col;
}

template <typename T>
void MpiSy2sbContext<T>::initGpuResources() {
    // 设置当前进程使用的 GPU
    cudaSetDevice(mpi_config.local_gpu_id);

    // 创建 CUDA 流
    cudaStreamCreate(&stream);

    // 设置 cuBLAS 和 cuSOLVER 句柄的流
    cublasSetStream(cublas_handle, stream);
    cusolverDnSetStream(cusolver_handle, stream);
}

template <typename T>
void MpiSy2sbContext<T>::initCommunication() {
    // 在 MPI 环境中初始化层次化 NCCL 通信组
    // 获取所有进程的 GPU ID
    std::vector<int> all_gpu_ids(mpi_config.size);

    // 收集所有进程的本地 GPU ID
    MPI_Allgather(&mpi_config.local_gpu_id, 1, MPI_INT, all_gpu_ids.data(), 1,
                  MPI_INT, MPI_COMM_WORLD);

    // 创建层次化通信组：[0,1,2,3] -> [1,2,3] -> [2,3] -> [3]
    // 这样设计可以让早完成的进程开始下一轮计算，实现流水线并行

    // 1. 主通信组：所有进程 [0,1,2,3,...,size-1]
    ncclUniqueId main_nccl_id;
    if (mpi_config.rank == 0) {
        ncclGetUniqueId(&main_nccl_id);
    }
    MPI_Bcast(&main_nccl_id, sizeof(main_nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclResult_t nccl_result = ncclCommInitRank(&nccl_comm, mpi_config.size,
                                                main_nccl_id, mpi_config.rank);
    if (nccl_result != ncclSuccess) {
        throw std::runtime_error(
            fmt::format("Main NCCL initialization failed: {}",
                        ncclGetErrorString(nccl_result)));
    }

    // 2. 创建子通信组：每个进程参与从自己开始到最后的通信组
    // 进程i参与通信组 [i, i+1, ..., size-1]
    sub_comm_groups.resize(mpi_config.size);
    sub_mpi_comms.resize(mpi_config.size);  // 同时创建对应的MPI子通信器

    for (int start_rank = 0; start_rank < mpi_config.size; start_rank++) {
        // 所有进程都参与MPI_Comm_split，但只有部分进程会被分配到有效的通信器
        int color =
            (mpi_config.rank >= start_rank) ? start_rank : MPI_UNDEFINED;

        MPI_Comm_split(MPI_COMM_WORLD, color, mpi_config.rank,
                       &sub_mpi_comms[start_rank]);

        if (mpi_config.rank >= start_rank) {
            int sub_group_size = mpi_config.size - start_rank;
            int sub_rank = mpi_config.rank - start_rank;

            // 只有当子组大小大于1时才创建NCCL通信器
            // 单进程组不需要NCCL通信
            if (sub_group_size > 1) {
                // 生成该子组的NCCL ID
                ncclUniqueId sub_nccl_id;
                if (mpi_config.rank == start_rank) {
                    ncclGetUniqueId(&sub_nccl_id);
                }

                // 在子组内广播NCCL ID

                MPI_Bcast(&sub_nccl_id, sizeof(sub_nccl_id), MPI_BYTE, 0,
                          sub_mpi_comms[start_rank]);

                // 初始化子组NCCL通信器

                ncclResult_t sub_result =
                    ncclCommInitRank(&sub_comm_groups[start_rank],
                                     sub_group_size, sub_nccl_id, sub_rank);

                if (sub_result != ncclSuccess) {
                    throw std::runtime_error(fmt::format(
                        "Sub NCCL group {} initialization failed: {}",
                        start_rank, ncclGetErrorString(sub_result)));
                }

            } else {
                // 单进程组：不需要NCCL通信器
                sub_comm_groups[start_rank] = nullptr;
            }

        } else {
            sub_comm_groups[start_rank] = nullptr;
            // sub_mpi_comms[start_rank]
            // 已经在MPI_Comm_split中设置为MPI_COMM_NULL
        }
    }
}

template <typename T>
void MpiSy2sbContext<T>::allocateGpuMemory() {
    // 计算矩阵分布：每个进程负责 n/size 列（按列分块）
    if (n % mpi_config.size != 0) {
        throw std::runtime_error("Matrix size must be divisible by MPI size");
    }

    // 设置当前 GPU 设备
    cudaSetDevice(mpi_config.local_gpu_id);

    // 分配各个矩阵的 GPU 内存 (直接调用 resize)
    // A, W, Y, oriA: 存储本地矩阵块 (n × cols_per_process) 按列分块
    gpu_A.resize(local_matrix_size);
    gpu_W.resize(local_matrix_size);
    gpu_Y.resize(local_matrix_size);
    gpu_oriA.resize(local_matrix_size);  // 原始矩阵 A 的备份

    // R: 存储 QR 分解的上三角矩阵 (n × nb)
    gpu_R.resize(n * nb);

    // Z: 工作矩阵，用于 Householder 向量 (n × nb)
    gpu_Z.resize(n * nb);

    // work: 临时工作空间 (2 × n × nb)
    gpu_work.resize(2 * n * nb);

    // 初始化为 0
    thrust::fill(gpu_W.begin(), gpu_W.end(), T(0));
    thrust::fill(gpu_Y.begin(), gpu_Y.end(), T(0));
    thrust::fill(gpu_Z.begin(), gpu_Z.end(), T(0));

    // 复制主机数据到 GPU
    copyHostToGpu();
}

template <typename T>
void MpiSy2sbContext<T>::copyHostToGpu() {
    try {
        // 按列复制 A 矩阵的本地部分
        // 对于列主序存储，我们需要复制连续的列块
        thrust::copy(A_host + start_col * n,
                     A_host + start_col * n + local_matrix_size, gpu_A.begin());

        // 同时复制到 oriA 作为原始矩阵的备份
        thrust::copy(A_host + start_col * n,
                     A_host + start_col * n + local_matrix_size,
                     gpu_oriA.begin());
    } catch (const std::exception& e) {
        throw std::runtime_error(
            fmt::format("Failed to copy host data to GPU: {}", e.what()));
    }
}

template <typename T>
void MpiSy2sbContext<T>::cleanup() {
    // 销毁所有子 NCCL 通信器
    for (size_t i = 0; i < sub_comm_groups.size(); i++) {
        if (sub_comm_groups[i] != nullptr) {
            ncclCommDestroy(sub_comm_groups[i]);
        }
    }
    sub_comm_groups.clear();

    // 销毁所有子 MPI 通信器
    for (size_t i = 0; i < sub_mpi_comms.size(); i++) {
        if (sub_mpi_comms[i] != MPI_COMM_NULL) {
            MPI_Comm_free(&sub_mpi_comms[i]);
        }
    }
    sub_mpi_comms.clear();

    // 销毁主 NCCL 通信器
    if (nccl_comm != nullptr) {
        ncclCommDestroy(nccl_comm);
    }

    // 销毁 CUDA 流
    cudaStreamDestroy(stream);
}

namespace internal {


// 内部函数实现
template <typename T>
void sy2sb_recursive_mpi(size_t recursive_depth,
                         matrix_ops::mpi::MpiSy2sbContext<T>& ctx) {
    // compute recrusive offset and panel related resources
    auto recrusive_offset_finished = ctx.nb * recursive_depth;
    auto recrusive_offset = (ctx.nb + ctx.nb * ctx.n) * recursive_depth;
    auto gpu_index = ctx.computeProcessForColumn(recrusive_offset_finished);

    if (ctx.mpi_config.rank < gpu_index) {
        return;
    }

    auto mpi_comm = ctx.sub_mpi_comms[gpu_index];
    auto& handle = ctx.cublas_handle;
    auto& cusolver_handle = ctx.cusolver_handle;

    thrust::device_ptr<T> A, W, Y, Z, R, work_ptr;

    if (ctx.mpi_config.rank == gpu_index) {
        A = ctx.gpu_A.data() + recrusive_offset - ctx.start_col * ctx.n;
        W = ctx.gpu_W.data() + recrusive_offset - ctx.start_col * ctx.n;
        Y = ctx.gpu_Y.data() + recrusive_offset - ctx.start_col * ctx.n;
        R = ctx.gpu_R.data() + recrusive_offset_finished;
        Z = ctx.gpu_Z.data();
    }

    auto lda = ctx.n;
    auto ldw = ctx.n;
    auto ldr = ctx.n;
    auto ldy = ctx.n;
    auto ldz = ctx.n;
    auto ldwork = ctx.nb;

    // for-loop update with b panel
    for (auto i = ctx.b; i <= ctx.nb && i < ctx.n - recrusive_offset_finished;
         i += ctx.b) {
        thrust::device_ptr<T> panel_ptr, panel_W_ptr, panel_Y_ptr, panel_Z_ptr;
        auto panel_m = ctx.n - recrusive_offset_finished - i;
        auto panel_n = ctx.b;
        panel_ptr = A + i + (i - ctx.b) * lda;
        panel_W_ptr = W + i + (i - ctx.b) * ldw;
        panel_Y_ptr = Y + i + (i - ctx.b) * ldy;
        panel_Z_ptr = Z + i + (i - ctx.b) * ldz;

        // process for this panel do the work
        if (gpu_index == ctx.mpi_config.rank) {
            matrix_ops::internal::sy2sb::panelQR(handle, ctx.cusolver_handle,
                                                 panel_m, panel_n, panel_ptr,
                                                 lda, R, lda, panel_W_ptr, ldw);
            // copy panel data to panelY (using lda)
            matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                    thrust::device_ptr<T>, T>(
                panel_ptr, lda, panel_Y_ptr, ldy, panel_m, panel_n);

            // copy panelR data to panel (using lda)
            matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                    thrust::device_ptr<T>, T>(
                R, lda, panel_ptr, lda, panel_m, panel_n);
        }
    }

    // 递归终止条件检查 (参照分布式版本的位置)
    if (ctx.n <= ctx.nb + recrusive_offset_finished) {
        return;
    }

    // TODO: 在这里需要添加类似分布式版本的矩阵更新逻辑 (syr2k等)
    // 这是递归间的关键多进程操作
    // performInterRecursiveSyr2k(ctx, recursive_depth,
    // recursive_offset_finished);

    // 5. 递归调用下一层
    sy2sb_recursive_mpi(recursive_depth + 1, ctx);
}

}  // namespace internal

/**
 * @brief MPI 版本的 sy2sb 主函数
 */
template <typename T>
void sy2sb(const MpiConfig& mpi_config, size_t n, T* A, size_t lda, T* W,
           size_t ldw, T* Y, size_t ldy, size_t nb, size_t b) {
    // 检查分块兼容性（与分布式版本保持一致）
    if (n % b % mpi_config.size != 0) {
        throw std::runtime_error(
            "Matrix is not well divisible into MPI process panels");
    }

    // 创建 MPI sy2sb 上下文
    MpiSy2sbContext<T> ctx(mpi_config, n, A, lda, W, ldw, Y, ldy, nb, b);

    // 调用递归实现
    internal::sy2sb_recursive_mpi<T>(0, ctx);

    // 最后全局同步确保所有进程完成
    MPI_Barrier(MPI_COMM_WORLD);
}

}  // namespace mpi

// 显式模板实例化
template class matrix_ops::mpi::MpiSy2sbContext<float>;
template class matrix_ops::mpi::MpiSy2sbContext<double>;

template void matrix_ops::mpi::sy2sb<float>(
    const matrix_ops::mpi::MpiConfig& mpi_config, size_t n, float* A,
    size_t lda, float* W, size_t ldw, float* Y, size_t ldy, size_t nb,
    size_t b);

template void matrix_ops::mpi::sy2sb<double>(
    const matrix_ops::mpi::MpiConfig& mpi_config, size_t n, double* A,
    size_t lda, double* W, size_t ldw, double* Y, size_t ldy, size_t nb,
    size_t b);

}  // namespace matrix_ops
