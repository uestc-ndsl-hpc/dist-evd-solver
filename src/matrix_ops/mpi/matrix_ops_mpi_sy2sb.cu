#include <mpi.h>
#include <nccl.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstddef>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <sstream>

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
    // 默认使用现有 Blockwise 策略；支持通过环境变量预先切换为 BlockCyclic1D
    dist_type = DistributionType::Blockwise;
    block_size_bs = nb;  // block-cyclic 缺省使用 nb
    if (const char* dist_env = std::getenv("EVD_DIST")) {
        if (!std::strcmp(dist_env, "cyclic") ||
            !std::strcmp(dist_env, "blockcyclic") || !std::strcmp(dist_env, "bc")) {
            dist_type = DistributionType::BlockCyclic1D;
        }
    }

    // 计算分块信息（Blockwise）
    cols_per_process = n / mpi_config.size;
    start_col = mpi_config.rank * cols_per_process;
    local_matrix_size = cols_per_process * n;
    // 预计算当前策略下本地列数
    local_cols = computeLocalCols();
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
    if (dist_type == DistributionType::Blockwise) {
        return col_offset / cols_per_process;
    }
    return ownerOfCol(col_offset);
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
    // 与分布策略绑定
    if (dist_type == DistributionType::Blockwise) {
        return global_col - start_col;
    }
    return localColIndex(global_col);
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
    // 计算矩阵分布：
    // Blockwise 要求 n 能被 size 整除；BlockCyclic1D 无此硬性要求
    if (dist_type == DistributionType::Blockwise) {
        if (n % mpi_config.size != 0) {
            throw std::runtime_error(
                "Matrix size must be divisible by MPI size (blockwise)");
        }
    }

    // 设置当前 GPU 设备
    cudaSetDevice(mpi_config.local_gpu_id);

    // 分配各个矩阵的 GPU 内存 (直接调用 resize)
    // A, W, Y, oriA: 存储本地矩阵块，按当前分布策略打包为连续列
    size_t cols_owned = (dist_type == DistributionType::Blockwise)
                            ? cols_per_process
                            : local_cols;
    size_t local_elems = cols_owned * n;
    gpu_A.resize(local_elems);
    gpu_W.resize(local_elems);
    gpu_Y.resize(local_elems);
    gpu_oriA.resize(local_elems);  // 原始矩阵 A 的备份

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
        if (dist_type == DistributionType::Blockwise) {
            // 按列复制 A 矩阵的本地连续部分
            thrust::copy(A_host + start_col * n,
                         A_host + start_col * n + local_matrix_size,
                         gpu_A.begin());
            // 同时复制到 oriA 作为原始矩阵的备份
            thrust::copy(A_host + start_col * n,
                         A_host + start_col * n + local_matrix_size,
                         gpu_oriA.begin());
        } else {
            // BlockCyclic1D: 逐列打包复制
            auto bs = block_size_bs;
            size_t local_idx = 0;
            for (size_t j = 0; j < n; ++j) {
                if (ownerOfCol(j) == static_cast<size_t>(mpi_config.rank)) {
                    // 列 j 的本地列号
                    size_t lc = localColIndex(j);
                    // 源列起始指针（列主序）
                    const T* src = A_host + j * n;
                    // 目标列起始指针（列主序，打包）
                    thrust::device_ptr<T> dst = gpu_A.data() + lc * n;
                    // 拷贝 n 元素（整列）
                    thrust::copy(src, src + n, dst);
                    // 备份 oriA
                    thrust::device_ptr<T> dst_ori = gpu_oriA.data() + lc * n;
                    thrust::copy(src, src + n, dst_ori);
                    local_idx += 1;
                }
            }
        }
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

// 计算 block-cyclic (1D) 下本地列数
template <typename T>
size_t MpiSy2sbContext<T>::computeLocalCols() const {
    if (dist_type == DistributionType::Blockwise) {
        return cols_per_process;
    }
    auto P = static_cast<size_t>(mpi_config.size);
    auto r = static_cast<size_t>(mpi_config.rank);
    auto bs = block_size_bs > 0 ? block_size_bs : nb;
    size_t full_blocks = n / bs;
    size_t tail = n % bs;

    size_t base_blocks = full_blocks / P;
    size_t extra_blocks = (r < (full_blocks % P)) ? 1 : 0;
    size_t cols = (base_blocks + extra_blocks) * bs;
    // 处理尾块
    if (tail > 0 && (full_blocks % P) == r) {
        cols += tail;
    }
    return cols;
}

namespace internal {

// 前向声明：调试用 CUDA 同步打印
static inline void debug_cuda_sync(const char* where);

// 定义：调试用 CUDA 同步打印（放在相同命名空间内，避免链接问题）
static inline void debug_cuda_sync(const char* where) {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        util::MpiLogger::error("[CUDA] {}: {}", where,
                               cudaGetErrorString(err));
    }
}

/**
 * @brief Functor to copy the lower triangular part of a matrix to the upper
 * triangular part.
 *
 * @tparam T Data type of the matrix elements.
 */
template <typename T>
struct make_symmetric_functor {
    thrust::device_ptr<T> A_;
    size_t n_;
    size_t lda_;

    make_symmetric_functor(thrust::device_ptr<T> A, size_t n, size_t lda)
        : A_(A), n_(n), lda_(lda) {}

    __device__ void operator()(const size_t& k) const {
        size_t j = k % n_;  // row
        size_t i = k / n_;  // col
        if (j < i) {
            A_[j + i * lda_] = A_[i + j * lda_];
        }
    }
};

// 新提取的函数，用于执行面板QR分解和相关的数据复制
template <typename T>
void performPanelQrComputeWy(int rank, const common::CublasHandle& handle,
                             const common::CusolverDnHandle& cusolver_handle,
                             size_t gpu_index, size_t panel_m, size_t panel_n,
                             thrust::device_ptr<T> panel_ptr, size_t lda,
                             thrust::device_ptr<T> R, size_t ldr,
                             thrust::device_ptr<T> panel_W_ptr, size_t ldw,
                             thrust::device_ptr<T> panel_Y_ptr, size_t ldy,
                             MPI_Comm& comm) {
    if (rank == gpu_index) {
        // execute panel QR decomposition
        matrix_ops::internal::sy2sb::panelQR(handle, cusolver_handle, panel_m,
                                             panel_n, panel_ptr, lda, R, ldr,
                                             panel_W_ptr, ldw);
        // copy panel data to panelY (using lda)
        matrix_ops::matrix_copy<thrust::device_ptr<T>, thrust::device_ptr<T>,
                                T>(panel_ptr, lda, panel_Y_ptr, ldy, panel_m,
                                   panel_n);

        // copy panelR data to panel (using lda)
        matrix_ops::matrix_copy<thrust::device_ptr<T>, thrust::device_ptr<T>,
                                T>(R, lda, panel_ptr, lda, panel_m, panel_n);
    }
    MPI_Barrier(comm);
}

template <typename T>
void performComputeAw(matrix_ops::mpi::MpiSy2sbContext<T>& ctx, MPI_Comm& comm,
                      int rank, size_t gpu_index, size_t panel_m,
                      size_t panel_n, size_t i, size_t lda, size_t ldw,
                      size_t ldz, size_t recrusive_offset,
                      size_t recrusive_offset_finished) {
    if (ctx.dist_type == DistributionType::Blockwise) {
        auto rest_gpu_num = ctx.mpi_config.size - gpu_index;

        // single card
        if (rest_gpu_num == 1) {
            auto oriA_panel = ctx.gpu_oriA.data() + recrusive_offset -
                              ctx.start_col * ctx.n + i * lda + i;
            auto panel_W_ptr = ctx.gpu_W.data() + recrusive_offset -
                               ctx.start_col * ctx.n + i + (i - ctx.b) * ldw;
            auto panel_Z_ptr = ctx.gpu_Z.data() + i + (i - ctx.b) * ldz;
            matrix_ops::gemm(ctx.cublas_handle, panel_m, ctx.b, panel_m, (T)1,
                             oriA_panel, lda, false, panel_W_ptr, ldw, false,
                             (T)0, panel_Z_ptr, ldz);
        } else {
            if (ctx.mpi_config.rank == gpu_index) {
                // copy W to workspace
                auto panel_W_ptr = ctx.gpu_W.data() + recrusive_offset -
                                   ctx.start_col * ctx.n + i + (i - ctx.b) * ldw;
                matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                        thrust::device_ptr<T>, T>(
                    panel_W_ptr, ldw, ctx.gpu_work.data(), panel_m, panel_m,
                    ctx.b);
            }

            // 使用子通信组进行广播
            auto& sub_comm = ctx.sub_comm_groups[gpu_index];
            if (sub_comm != nullptr) {
                // root进程在子通信组中的rank是0
                ncclBcast(ctx.gpu_work.data().get(), ctx.b * panel_m,
                          ctx.nccl_type, 0, sub_comm, ctx.stream);
                cudaStreamSynchronize(ctx.stream);
            }

            auto oriA_panel =
                ctx.gpu_oriA.data() + i + recrusive_offset_finished;
            auto z_panel_rows = ctx.cols_per_process;

            if (gpu_index == ctx.mpi_config.rank) {
                oriA_panel = ctx.gpu_oriA.data() + recrusive_offset -
                             ctx.start_col * ctx.n + i + i * lda;
                z_panel_rows =
                    panel_m - ctx.cols_per_process * (rest_gpu_num - 1);
            }

            auto aw_panel = ctx.gpu_work.data() + ctx.n * ctx.nb;

            if (z_panel_rows > 0) {
                try {
                    matrix_ops::gemm(ctx.cublas_handle, z_panel_rows, ctx.b,
                                     panel_m, (T)1, oriA_panel, lda, true,
                                     ctx.gpu_work.data(), panel_m, false, (T)0,
                                     aw_panel, z_panel_rows);
                } catch (const std::exception& e) {
                    throw std::runtime_error(fmt::format(
                        "here aw gemm error exception: {}", e.what()));
                } catch (...) {
                    throw std::runtime_error(
                        "here aw gemm error: an unknown exception "
                        "occurred");
                }
            }

            std::vector<thrust::device_vector<T>> z_recv(rest_gpu_num - 1);

            // 使用子通信组进行Send/Recv
            if (sub_comm != nullptr) {
                ncclGroupStart();
                if (rank != gpu_index) {
                    // 在子通信组中，目标进程的rank是0
                    ncclSend(aw_panel.get(), ctx.cols_per_process * ctx.b,
                             ctx.nccl_type, 0, sub_comm, ctx.stream);
                } else {
                    for (auto gpu_offset = 1; gpu_offset < rest_gpu_num;
                         gpu_offset++) {
                        z_recv[gpu_offset - 1].resize(ctx.cols_per_process *
                                                      ctx.b);
                        // 在子通信组中，源进程的rank是 gpu_offset
                        ncclRecv(z_recv[gpu_offset - 1].data().get(),
                                 ctx.cols_per_process * ctx.b, ctx.nccl_type,
                                 gpu_offset, sub_comm, ctx.stream);
                    }
                }
                ncclGroupEnd();
            }

            cudaStreamSynchronize(ctx.stream);

            if (rank == gpu_index) {
                auto panel_Z_ptr = ctx.gpu_Z.data() + i + (i - ctx.b) * ldz;
                if (z_panel_rows > 0) {
                    matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                            thrust::device_ptr<T>, T>(
                        aw_panel, z_panel_rows, panel_Z_ptr, ldz, z_panel_rows,
                        ctx.b);
                }
                for (auto index = 1; index < rest_gpu_num; index++) {
                    auto row_finished = (index - 1) * ctx.cols_per_process +
                                        panel_m - ctx.cols_per_process *
                                                       (rest_gpu_num - 1);
                    matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                            thrust::device_ptr<T>, T>(
                        z_recv[index - 1].data(), ctx.cols_per_process,
                        panel_Z_ptr + row_finished, ldz,
                        ctx.cols_per_process, ctx.b);
                }
            }

            MPI_Barrier(comm);
        }
        return;
    }

    // BlockCyclic1D: 基于循环块按 bs 大小分块处理并由块拥有者计算对应 Z 的行块
    auto bs = ctx.block_size_bs;
    size_t tail_start = recrusive_offset_finished + i;
    size_t panel_rows = panel_m;  // = n - tail_start

    // root 复制 W 到工作区并全员广播（使用主通信器，覆盖所有参与者）
    if (ctx.mpi_config.rank == gpu_index) {
        auto panel_W_ptr = ctx.ptrLocalRC(ctx.gpu_W, tail_start,
                                          recrusive_offset_finished + i -
                                              ctx.b);
        matrix_ops::matrix_copy<thrust::device_ptr<T>, thrust::device_ptr<T>,
                                T>(panel_W_ptr, ldw, ctx.gpu_work.data(),
                                   panel_rows, panel_rows, ctx.b);
    }

    ncclBcast(ctx.gpu_work.data().get(), ctx.b * panel_rows, ctx.nccl_type,
              gpu_index, ctx.nccl_comm, ctx.stream);
    cudaStreamSynchronize(ctx.stream);

    // 每个循环块独立计算并装配到 owner 的 Z 中
    size_t num_blocks = (panel_rows + bs - 1) / bs;
    for (size_t t = 0; t < num_blocks; ++t) {
        size_t j0 = tail_start + t * bs;               // 该循环块的起始全局列
        size_t w = std::min(bs, ctx.n - j0);           // 块宽
        size_t owner = ctx.ownerOfCol(j0);             // 块拥有者（按循环块）
        bool i_am_owner = (ctx.mpi_config.rank == static_cast<int>(owner));

        // 拿到本地 A 子块起点（panel_rows x w），按列主序打包
        thrust::device_ptr<T> A_block;
        if (i_am_owner) {
            // 注意行起点应为 tail_start，而不是 j0
            A_block = ctx.ptrLocalRC(ctx.gpu_oriA, tail_start, j0);
        }

        // 计算 aw_block = A_block^T * W  => 尺寸 (w x b)，列主序，ld = w
        thrust::device_ptr<T> aw_block = ctx.gpu_work.data() + ctx.n * ctx.nb;
        if (i_am_owner) {
            if (w > 0) {
                matrix_ops::gemm(ctx.cublas_handle, w, ctx.b, panel_rows,
                                 (T)1, A_block, lda, true, ctx.gpu_work.data(),
                                 panel_rows, false, (T)0, aw_block, w);
            }
        }

        // 非 owner 不计算，直接同步
        cudaStreamSynchronize(ctx.stream);

        // 把结果送到 panel owner（gpu_index）并由其写入 Z 相应行偏移
        size_t row_offset = t * bs;  // 该块在 Z 中的起始行（按全局列顺序）
        if (ctx.mpi_config.rank != gpu_index) {
            if (i_am_owner && w > 0) {
                ncclSend(aw_block.get(), w * ctx.b, ctx.nccl_type, gpu_index,
                         ctx.nccl_comm, ctx.stream);
            }
        }

        if (ctx.mpi_config.rank == gpu_index) {
            // 在 owner 上：接收或直接使用本地计算的块，然后拷贝到 Z
            thrust::device_vector<T> recv_buf;  // 按需分配
            thrust::device_ptr<T> src_block = aw_block;
            if (!i_am_owner) {
                if (w > 0) {
                    recv_buf.resize(w * ctx.b);
                    ncclRecv(recv_buf.data().get(), w * ctx.b, ctx.nccl_type,
                             static_cast<int>(owner), ctx.nccl_comm,
                             ctx.stream);
                    src_block = recv_buf.data();
                }
            }

            auto panel_Z_ptr = ctx.ptrLocalRC(
                ctx.gpu_Z, tail_start, recrusive_offset_finished + i - ctx.b);
            if (w > 0) {
                matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                        thrust::device_ptr<T>, T>(
                    src_block, w, panel_Z_ptr + row_offset, ldz, w, ctx.b);
            }
        }

        cudaStreamSynchronize(ctx.stream);
        MPI_Barrier(comm);
    }
}

template <typename T>
void performInterRecursiveSyr2k(size_t recrusive_depth,
                                matrix_ops::mpi::MpiSy2sbContext<T>& ctx,
                                size_t gpu_index, thrust::device_ptr<T> A,
                                size_t lda, thrust::device_ptr<T> Y, size_t ldy,
                                thrust::device_ptr<T> Z, size_t ldz) {
    auto offset = (recrusive_depth + 1) * (ctx.nb + ctx.nb * ctx.n);
    auto tail_gpu_start_index =
        ctx.computeProcessForColumn((recrusive_depth + 1) * ctx.nb);
    auto rest_gpu_num = ctx.mpi_config.size - gpu_index;
    auto sub_matrix_n = ctx.n - (recrusive_depth + 1) * ctx.nb;
    MPI_Comm comm = (ctx.dist_type == DistributionType::BlockCyclic1D)
                        ? MPI_COMM_WORLD
                        : ctx.sub_mpi_comms[gpu_index];
    auto tail_start = (recrusive_depth + 1) * ctx.nb;

    // BlockCyclic1D: 采用循环块分配尾部更新，每个块的列由其拥有者本地更新
    if (ctx.dist_type == DistributionType::BlockCyclic1D) {
        // 1) 准备并广播 Y_tail 与 Z_tail（尺寸 sub_n x nb，按 ld=sub_n 打包）
        auto sub_n = sub_matrix_n;
        auto bs = ctx.block_size_bs;

        thrust::device_ptr<T> y_bcast = ctx.gpu_work.data();
        thrust::device_ptr<T> z_bcast = ctx.gpu_work.data() + ctx.n * ctx.nb;

        if (ctx.mpi_config.rank == gpu_index) {
            // 源在面板 owner 上：把 [row=tail_start: , col=panel_start] 打成连续
            auto y_src = ctx.ptrLocalRC(ctx.gpu_Y, tail_start, tail_start - ctx.nb);
            auto z_src = ctx.ptrLocalRC(ctx.gpu_Z, tail_start, tail_start - ctx.nb);
            matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                    thrust::device_ptr<T>, T>(
                y_src, ldy, y_bcast, sub_n, sub_n, ctx.nb);
            matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                    thrust::device_ptr<T>, T>(
                z_src, ldz, z_bcast, sub_n, sub_n, ctx.nb);
        }

        // 广播到所有进程
        ncclBcast(y_bcast.get(), sub_n * ctx.nb, ctx.nccl_type, gpu_index,
                  ctx.nccl_comm, ctx.stream);
        ncclBcast(z_bcast.get(), sub_n * ctx.nb, ctx.nccl_type, gpu_index,
                  ctx.nccl_comm, ctx.stream);
        cudaStreamSynchronize(ctx.stream);

        // 2) 遍历尾部按 bs 分块的列块，由块拥有者计算并就地更新 C(:,J)
        size_t num_blocks = (sub_n + bs - 1) / bs;
        for (size_t t = 0; t < num_blocks; ++t) {
            size_t j0 = tail_start + t * bs;    // 全局列起点
            size_t w = std::min(bs, ctx.n - j0);  // 块宽
            size_t owner = ctx.ownerOfCol(j0);
            bool i_am_owner = (ctx.mpi_config.rank == static_cast<int>(owner));

            if (w == 0) continue;

            if (i_am_owner) {
                // 本地 C(:,J) 指针（在 oriA 中更新）和 A 镜像
                auto c_block = ctx.ptrLocalRC(ctx.gpu_oriA, tail_start, j0);
                auto a_block = ctx.ptrLocalRC(ctx.gpu_A, tail_start, j0);

                // 选取 Y/J 和 Z/J 的行块（w x nb），并进行两个 GEMM 叠加
                size_t row_off = j0 - tail_start;  // 在打包 y/z 中的行偏移
                auto y_rows = y_bcast + row_off;   // (w x nb) with ld=sub_n
                auto z_rows = z_bcast + row_off;   // (w x nb) with ld=sub_n

                // C(:,J) -= Y * Z[J,:]^T
                matrix_ops::gemm(ctx.cublas_handle, sub_n, w, ctx.nb, T(-1),
                                 y_bcast, sub_n, false, z_rows, sub_n, true,
                                 T(1), c_block, lda);
                // C(:,J) -= Z * Y[J,:]^T
                matrix_ops::gemm(ctx.cublas_handle, sub_n, w, ctx.nb, T(-1),
                                 z_bcast, sub_n, false, y_rows, sub_n, true,
                                 T(1), c_block, lda);

                // 可选：保持 A 镜像一致（复制更新后的列块到 gpu_A）
                matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                        thrust::device_ptr<T>, T>(
                    c_block, lda, a_block, lda, sub_n, w);
            }

            cudaStreamSynchronize(ctx.stream);
            MPI_Barrier(comm);
        }

        return;
    }
    thrust::device_ptr<T> tail_matrix_ptr;
    if (ctx.mpi_config.rank == tail_gpu_start_index) {
        tail_matrix_ptr = ctx.gpu_oriA.data() + offset - ctx.start_col * ctx.n;
    }
    thrust::device_vector<T> z_send;
    if (ctx.mpi_config.rank == gpu_index) {
        z_send.resize(ctx.n * ctx.nb);
    }
    if (gpu_index == tail_gpu_start_index) {
        if (rest_gpu_num == 1) {
            if (ctx.mpi_config.rank == gpu_index) {
                matrix_ops::syr2k(ctx.cublas_handle, sub_matrix_n, ctx.nb,
                                  (T)(-1), Y + ctx.nb, ldy, Z + ctx.nb, ldz,
                                  (T)1, tail_matrix_ptr, lda);

                thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                                 thrust::make_counting_iterator<size_t>(
                                     sub_matrix_n * sub_matrix_n),
                                 make_symmetric_functor<T>(tail_matrix_ptr,
                                                           sub_matrix_n, lda));

                matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                        thrust::device_ptr<T>, T>(
                    tail_matrix_ptr, ctx.n, A + ctx.nb + ctx.nb * lda, ctx.n,
                    sub_matrix_n, sub_matrix_n);
            }
        } else {
            if (ctx.mpi_config.rank == gpu_index) {
                matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                        thrust::device_ptr<T>, T>(
                    Y + ctx.nb, lda, ctx.gpu_work.data(), sub_matrix_n,
                    sub_matrix_n, ctx.nb);
                matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                        thrust::device_ptr<T>, T>(
                    Z + ctx.nb, lda, z_send.data(), sub_matrix_n, sub_matrix_n,
                    ctx.nb);
            }
            MPI_Barrier(comm);

            auto z_bcast = ctx.gpu_Z.data();
            if (ctx.mpi_config.rank == gpu_index) {
                z_bcast = z_send.data();
            }
            auto& sub_comm = ctx.sub_comm_groups[gpu_index];
            if (sub_comm != nullptr) {
                ncclBcast(z_bcast.get(), sub_matrix_n * ctx.nb, ctx.nccl_type,
                          0, sub_comm, ctx.stream);
                ncclBcast(ctx.gpu_work.data().get(), sub_matrix_n * ctx.nb,
                          ctx.nccl_type, 0, sub_comm, ctx.stream);
            }
            cudaStreamSynchronize(ctx.stream);

            auto syr2k_panel_col = ctx.cols_per_process;
            auto& syr2k_panel_handle = ctx.cublas_handle;
            auto syr2k_panel_oriA_ptr =
                ctx.gpu_oriA.data() + (ctx.n - sub_matrix_n);
            auto dst_A_ptr = ctx.gpu_A.data() + (ctx.n - sub_matrix_n);
            auto zy_panel_offset =
                sub_matrix_n - (ctx.mpi_config.size - ctx.mpi_config.rank) *
                                   ctx.cols_per_process;
            if (ctx.mpi_config.rank == gpu_index) {
                syr2k_panel_col =
                    sub_matrix_n - (rest_gpu_num - 1) * ctx.cols_per_process;
                syr2k_panel_oriA_ptr = tail_matrix_ptr;
                dst_A_ptr = ctx.gpu_A.data() + offset - ctx.start_col * ctx.n;
                zy_panel_offset = 0;
            }

            matrix_ops::gemm(syr2k_panel_handle, sub_matrix_n, syr2k_panel_col,
                             ctx.nb, T(-1), z_bcast, sub_matrix_n, false,
                             ctx.gpu_work.data() + zy_panel_offset,
                             sub_matrix_n, true, T(1), syr2k_panel_oriA_ptr,
                             lda);
            matrix_ops::gemm(syr2k_panel_handle, sub_matrix_n, syr2k_panel_col,
                             ctx.nb, T(-1), ctx.gpu_work.data(), sub_matrix_n,
                             false, z_bcast + zy_panel_offset, sub_matrix_n,
                             true, T(1), syr2k_panel_oriA_ptr, lda);

            matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                    thrust::device_ptr<T>, T>(
                syr2k_panel_oriA_ptr, lda, dst_A_ptr, lda, sub_matrix_n,
                syr2k_panel_col);
        }
    } else {
        if (rest_gpu_num == 1) {
            if (ctx.mpi_config.rank == gpu_index) {
                matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                        thrust::device_ptr<T>, T>(
                    Y + ctx.nb, lda, ctx.gpu_work.data(), sub_matrix_n,
                    sub_matrix_n, ctx.nb);
                matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                        thrust::device_ptr<T>, T>(
                    Z + ctx.nb, lda, z_send.data(), sub_matrix_n, sub_matrix_n,
                    ctx.nb);
            }

            ncclGroupStart();
            if (ctx.mpi_config.rank == gpu_index) {
                ncclSend(ctx.gpu_work.data().get(), sub_matrix_n * ctx.nb,
                         ctx.nccl_type, tail_gpu_start_index, ctx.nccl_comm,
                         ctx.stream);
            } else if (ctx.mpi_config.rank == tail_gpu_start_index) {
                ncclRecv(ctx.gpu_work.data().get(), sub_matrix_n * ctx.nb,
                         ctx.nccl_type, gpu_index, ctx.nccl_comm, ctx.stream);
            }
            ncclGroupEnd();

            cudaStreamSynchronize(ctx.stream);

            ncclGroupStart();
            if (ctx.mpi_config.rank == gpu_index) {
                ncclSend(z_send.data().get(), sub_matrix_n * ctx.nb,
                         ctx.nccl_type, tail_gpu_start_index, ctx.nccl_comm,
                         ctx.stream);
            } else if (ctx.mpi_config.rank == tail_gpu_start_index) {
                ncclRecv(ctx.gpu_Z.data().get(), sub_matrix_n * ctx.nb,
                         ctx.nccl_type, gpu_index, ctx.nccl_comm, ctx.stream);
            }
            ncclGroupEnd();

            cudaStreamSynchronize(ctx.stream);

            if (ctx.mpi_config.rank == tail_gpu_start_index) {
                matrix_ops::syr2k(ctx.cublas_handle, sub_matrix_n, ctx.nb,
                                  (T)(-1), ctx.gpu_work.data(), sub_matrix_n,
                                  ctx.gpu_Z.data(), sub_matrix_n, (T)1,
                                  tail_matrix_ptr, lda);
                thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                                 thrust::make_counting_iterator<size_t>(
                                     sub_matrix_n * sub_matrix_n),
                                 make_symmetric_functor<T>(tail_matrix_ptr,
                                                           sub_matrix_n, lda));
                matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                        thrust::device_ptr<T>, T>(
                    tail_matrix_ptr, lda,
                    ctx.gpu_A.data() + offset - ctx.cols_per_process * ctx.n,
                    lda, sub_matrix_n, sub_matrix_n);
            }
        } else {
            if (ctx.mpi_config.rank == gpu_index) {
                matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                        thrust::device_ptr<T>, T>(
                    Y + ctx.nb, lda, ctx.gpu_work.data(), sub_matrix_n,
                    sub_matrix_n, ctx.nb);

                matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                        thrust::device_ptr<T>, T>(
                    Z + ctx.nb, lda, z_send.data(), sub_matrix_n, sub_matrix_n,
                    ctx.nb);
            }
            auto z_bcast = ctx.gpu_Z.data();
            if (ctx.mpi_config.rank == gpu_index) {
                z_bcast = z_send.data();
            }
            auto& sub_comm = ctx.sub_comm_groups[gpu_index];
            if (sub_comm != nullptr) {
                ncclBcast(z_bcast.get(), sub_matrix_n * ctx.nb, ctx.nccl_type,
                          0, sub_comm, ctx.stream);
                ncclBcast(ctx.gpu_work.data().get(), sub_matrix_n * ctx.nb,
                          ctx.nccl_type, 0, sub_comm, ctx.stream);
            }
            cudaStreamSynchronize(ctx.stream);
            if (ctx.mpi_config.rank != gpu_index) {
                auto syr2k_panel_col = ctx.cols_per_process;
                auto& syr2k_panel_handle = ctx.cublas_handle;
                auto syr2k_panel_oriA_ptr =
                    ctx.gpu_oriA.data() + (ctx.n - sub_matrix_n);
                auto dst_A_ptr = ctx.gpu_A.data() + (ctx.n - sub_matrix_n);

                auto zy_panel_offset =
                    sub_matrix_n - (ctx.mpi_config.size - ctx.mpi_config.rank) *
                                       ctx.cols_per_process;

                matrix_ops::gemm(
                    syr2k_panel_handle, sub_matrix_n, syr2k_panel_col, ctx.nb,
                    T(-1), z_bcast, sub_matrix_n, false,
                    ctx.gpu_work.data() + zy_panel_offset, sub_matrix_n, true,
                    T(1), syr2k_panel_oriA_ptr, lda);
                matrix_ops::gemm(syr2k_panel_handle, sub_matrix_n,
                                 syr2k_panel_col, ctx.nb, T(-1),
                                 ctx.gpu_work.data(), sub_matrix_n, false,
                                 z_bcast + zy_panel_offset, sub_matrix_n, true,
                                 T(1), syr2k_panel_oriA_ptr, lda);
                matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                        thrust::device_ptr<T>, T>(
                    syr2k_panel_oriA_ptr, lda, dst_A_ptr, lda, sub_matrix_n,
                    syr2k_panel_col);
            }
        }
    }
    MPI_Barrier(comm);
}

template <typename T>
void sy2sb_recursive_mpi(size_t recursive_depth,
                         matrix_ops::mpi::MpiSy2sbContext<T>& ctx) {
    // compute recrusive offset and panel related resources
    auto recrusive_offset_finished = ctx.nb * recursive_depth;
    auto recrusive_offset = (ctx.nb + ctx.nb * ctx.n) * recursive_depth;
    auto gpu_index = ctx.computeProcessForColumn(recrusive_offset_finished);

    if (ctx.dist_type != DistributionType::BlockCyclic1D &&
        ctx.mpi_config.rank < gpu_index) {
        return;
    }

    MPI_Comm mpi_comm = (ctx.dist_type == DistributionType::BlockCyclic1D)
                            ? MPI_COMM_WORLD
                            : ctx.sub_mpi_comms[gpu_index];
    auto& handle = ctx.cublas_handle;
    auto& cusolver_handle = ctx.cusolver_handle;

    thrust::device_ptr<T> A, W, Y, Z, R, work_ptr;

    // 调试：进入递归同步一次，捕获前序非法访问
    internal::debug_cuda_sync("enter sy2sb_recursive_mpi");

    if (ctx.mpi_config.rank == gpu_index) {
        if (ctx.dist_type == DistributionType::Blockwise) {
            A = ctx.gpu_A.data() + recrusive_offset - ctx.start_col * ctx.n;
            W = ctx.gpu_W.data() + recrusive_offset - ctx.start_col * ctx.n;
            Y = ctx.gpu_Y.data() + recrusive_offset - ctx.start_col * ctx.n;
        } else {
            // block-cyclic: A/W/Y 直接使用映射在循环中计算的 panel 指针
            A = nullptr;
            W = nullptr;
            Y = nullptr;
        }
        R = ctx.gpu_R.data() + recrusive_offset_finished;
        Z = ctx.gpu_Z.data();
        work_ptr = ctx.gpu_work.data();
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
        // 迭代开始时同步一次，便于定位上一轮留下的非法访问
        if (ctx.dist_type == DistributionType::BlockCyclic1D) {
            std::ostringstream oss;
            oss << "enter iter depth=" << recursive_depth << " i=" << i
                << " owner=" << gpu_index << " rank="
                << ctx.mpi_config.rank;
            internal::debug_cuda_sync(oss.str().c_str());
        }
        thrust::device_ptr<T> panel_ptr, panel_W_ptr, panel_Y_ptr, panel_Z_ptr;
        auto panel_m = ctx.n - recrusive_offset_finished - i;
        auto panel_n = ctx.b;
        if (ctx.dist_type == DistributionType::Blockwise) {
            panel_ptr = A + i + (i - ctx.b) * lda;
            panel_W_ptr = W + i + (i - ctx.b) * ldw;
            panel_Y_ptr = Y + i + (i - ctx.b) * ldy;
            panel_Z_ptr = Z + i + (i - ctx.b) * ldz;
        } else {
            // block-cyclic: 仅在面板拥有者上计算面板指针
            if (ctx.mpi_config.rank == gpu_index) {
                panel_ptr = ctx.ptrLocalRC(
                    ctx.gpu_A, recrusive_offset_finished + i,
                    recrusive_offset_finished + i - ctx.b);
                panel_W_ptr = ctx.ptrLocalRC(
                    ctx.gpu_W, recrusive_offset_finished + i,
                    recrusive_offset_finished + i - ctx.b);
                panel_Y_ptr = ctx.ptrLocalRC(
                    ctx.gpu_Y, recrusive_offset_finished + i,
                    recrusive_offset_finished + i - ctx.b);
                panel_Z_ptr = ctx.ptrLocalRC(
                    ctx.gpu_Z, recrusive_offset_finished + i,
                    recrusive_offset_finished + i - ctx.b);

                // 校验面板所属与本地列索引
                size_t panel_col = recrusive_offset_finished + i - ctx.b;
                size_t owner = ctx.ownerOfCol(panel_col);
                if (owner != gpu_index) {
                    util::MpiLogger::error(
                        "[cyclic] panel owner mismatch: depth={} i={} expect_owner={} calc_owner={} rank={}",
                        recursive_depth, i, gpu_index, owner,
                        ctx.mpi_config.rank);
                }
                size_t lc = ctx.localColIndex(panel_col);
                if (lc >= ctx.local_cols) {
                    util::MpiLogger::error(
                        "[cyclic] localColIndex OOB: depth={} i={} lc={} local_cols={} rank={} col={}",
                        recursive_depth, i, lc, ctx.local_cols,
                        ctx.mpi_config.rank, panel_col);
                }
            }
        }

        internal::debug_cuda_sync("before panelQR");

        // process for this panel do the work
        try {
            performPanelQrComputeWy<T>(ctx.mpi_config.rank, handle,
                                       ctx.cusolver_handle, gpu_index, panel_m,
                                       panel_n, panel_ptr, lda, R, ldr,
                                       panel_W_ptr, ldw, panel_Y_ptr, ldy,
                                       mpi_comm);
        } catch (const std::exception& e) {
            util::MpiLogger::error(
                "panelQR failed at depth={} i={} owner={} rank={} m={} n={} err={}",
                recursive_depth, i, gpu_index, ctx.mpi_config.rank, panel_m,
                panel_n, e.what());
            debug_cuda_sync("panelQR");
            throw;
        }

        internal::debug_cuda_sync("after panelQR");

        // compute AW distribution
        try {
            performComputeAw<T>(ctx, mpi_comm, ctx.mpi_config.rank, gpu_index,
                                panel_m, panel_n, i, lda, ldw, ldz,
                                recrusive_offset, recrusive_offset_finished);
        } catch (const std::exception& e) {
            util::MpiLogger::error(
                "performComputeAw failed at depth={} i={} owner={} rank={} m={} n={} err={}",
                recursive_depth, i, gpu_index, ctx.mpi_config.rank, panel_m,
                panel_n, e.what());
            debug_cuda_sync("performComputeAw");
            throw;
        }

        internal::debug_cuda_sync("after performComputeAw");

        // compute all b panel update
        if (ctx.mpi_config.rank == gpu_index) {
            if (i == ctx.b) {
                try {
                    // panel_tmp = panel_z^T * panel_z
                    matrix_ops::gemm(handle, ctx.b, ctx.b, panel_m, (T)1,
                                     panel_W_ptr, ldw, true, panel_Z_ptr, ldz,
                                     false, (T)0, work_ptr, ldwork);
                    // panel_z = panel_z - panel_y * panel_z^T * panel_z
                    matrix_ops::gemm(handle, panel_m, ctx.b, ctx.b, (T)(-0.5),
                                     panel_Y_ptr, ldy, false, work_ptr, ldwork,
                                     false, (T)1, panel_Z_ptr, ldz);
                } catch (...) {
                    throw std::runtime_error(
                        "Error during initial panel update in sy2sb");
                }
            } else {
                try {
                    // panel_tmp = (Z + i)^T * panel_w
                    auto Z_ip = (ctx.dist_type == DistributionType::Blockwise)
                                    ? (Z + i)
                                    : ctx.ptrLocalRC(ctx.gpu_Z,
                                                     recrusive_offset_finished +
                                                         i,
                                                     recrusive_offset_finished +
                                                         i);
                    auto Y_ip = (ctx.dist_type == DistributionType::Blockwise)
                                    ? (Y + i)
                                    : ctx.ptrLocalRC(ctx.gpu_Y,
                                                     recrusive_offset_finished +
                                                         i,
                                                     recrusive_offset_finished +
                                                         i);
                    auto A_dst = (ctx.dist_type == DistributionType::Blockwise)
                                     ? (A + i + i * lda)
                                     : ctx.ptrLocalRC(
                                           ctx.gpu_A,
                                           recrusive_offset_finished + i,
                                           recrusive_offset_finished + i);

                    matrix_ops::gemm(handle, i - ctx.b, ctx.b, panel_m, (T)1,
                                     Z_ip, ldz, true, panel_W_ptr, ldw, false,
                                     (T)0, work_ptr, ldwork);
                    // panel_z = panel_z - Y+i * panel_z^T * panel_w
                    matrix_ops::gemm(handle, panel_m, ctx.b, i - ctx.b,
                                     (T)(-1), Y_ip, ldy, false, work_ptr,
                                     ldwork, false, (T)1, panel_Z_ptr, ldz);
                    // panel_tmp = Y+i^T * panel_w
                    matrix_ops::gemm(handle, i - ctx.b, ctx.b, panel_m,
                                     (T)(1), Y_ip, ldy, true, panel_W_ptr, ldw,
                                     false, (T)0, work_ptr, ldwork);
                    // panel_z = panel_z - (Z + i) * Y+i^T * panel_w
                    matrix_ops::gemm(handle, panel_m, ctx.b, i - ctx.b,
                                     (T)(-1), Z_ip, ldz, false, work_ptr,
                                     ldwork, false, (T)1, panel_Z_ptr, ldz);
                    // panel_tmp = panel_w^T * panel_z
                    matrix_ops::gemm(handle, ctx.b, ctx.b, panel_m, (T)1,
                                     panel_W_ptr, ldw, true, panel_Z_ptr, ldz,
                                     false, (T)0, work_ptr, ldwork);
                    // panel_z = panel_z - 0.5 * panel_y * panel_w^T * panel_z
                    matrix_ops::gemm(handle, panel_m, ctx.b, ctx.b, (T)(-0.5),
                                     panel_Y_ptr, ldy, false, work_ptr, ldwork,
                                     false, (T)1, panel_Z_ptr, ldz);
                } catch (...) {
                    throw std::runtime_error("Error in gemm");
                }
            }
            if (i < ctx.nb) {
                auto Z_ip = (ctx.dist_type == DistributionType::Blockwise)
                                ? (Z + i)
                                : ctx.ptrLocalRC(ctx.gpu_Z,
                                                 recrusive_offset_finished + i,
                                                 recrusive_offset_finished +
                                                     i);
                auto Y_ip = (ctx.dist_type == DistributionType::Blockwise)
                                ? (Y + i)
                                : ctx.ptrLocalRC(ctx.gpu_Y,
                                                 recrusive_offset_finished + i,
                                                 recrusive_offset_finished +
                                                     i);
                auto A_dst = (ctx.dist_type == DistributionType::Blockwise)
                                 ? (A + i + i * lda)
                                 : ctx.ptrLocalRC(ctx.gpu_A,
                                                  recrusive_offset_finished + i,
                                                  recrusive_offset_finished +
                                                      i);

                matrix_ops::gemm(handle, panel_m, ctx.b, i, (T)(-1), Y_ip, ldy,
                                 false, Z_ip, ldz, true, (T)1, A_dst, lda);

                matrix_ops::gemm(handle, panel_m, ctx.b, i, (T)(-1), Z_ip, ldz,
                                 false, Y_ip, ldy, true, (T)1, A_dst, lda);
            }
        }
        MPI_Barrier(mpi_comm);
    }

    // recursive quit
    if (ctx.n <= ctx.nb + recrusive_offset_finished) {
        return;
    }

    try {
        performInterRecursiveSyr2k(recursive_depth, ctx, gpu_index, A, lda, Y,
                                   ldy, Z, ldz);
    } catch (const std::exception& e) {
        util::MpiLogger::error(
            "performInterRecursiveSyr2k failed at depth={} owner={} rank={} err={}",
            recursive_depth, gpu_index, ctx.mpi_config.rank, e.what());
        debug_cuda_sync("performInterRecursiveSyr2k");
        throw;
    }

    // recursive call
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

    // 可通过环境变量切换分布策略：EVD_DIST=cyclic 使用 1D block-cyclic，默认 blockwise
    if (const char* dist_env = std::getenv("EVD_DIST")) {
        if (!std::strcmp(dist_env, "cyclic") || !std::strcmp(dist_env, "blockcyclic") ||
            !std::strcmp(dist_env, "bc")) {
            ctx.dist_type = DistributionType::BlockCyclic1D;
            // block_size_bs 默认等于 nb
        }
    }

    // 调试输出：确认当前分布模式和块大小（只在 rank 0 打印）
    if (mpi_config.rank == 0) {
        const char* mode = (ctx.dist_type == DistributionType::BlockCyclic1D)
                               ? "cyclic"
                               : "blockwise";
        util::MpiLogger::println("[sy2sb] distribution={}, bs={}, nb={}, b={}",
                                 mode, ctx.block_size_bs, nb, b);
    }

    util::MpiLogger::tic("sy2sb mpi");
    // 调用递归实现
    internal::sy2sb_recursive_mpi<T>(0, ctx);
    util::MpiLogger::toc("sy2sb mpi");

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
