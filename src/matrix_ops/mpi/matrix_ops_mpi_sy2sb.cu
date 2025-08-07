#include <mpi.h>
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
    util::MpiLogger::println("Process {} starting MpiSy2sbContext constructor",
                             mpi_config.rank);

    // 计算分块信息
    cols_per_process = n / mpi_config.size;
    start_col = mpi_config.rank * cols_per_process;
    local_matrix_size = cols_per_process * n;

    util::MpiLogger::println("Process {} initializing GPU resources",
                             mpi_config.rank);
    initGpuResources();

    util::MpiLogger::println("Process {} initializing communication",
                             mpi_config.rank);
    initCommunication();

    util::MpiLogger::println("Process {} allocating GPU memory",
                             mpi_config.rank);
    allocateGpuMemory();

    util::MpiLogger::println("Process {} completed MpiSy2sbContext constructor",
                             mpi_config.rank);
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
    util::MpiLogger::println("Process {} starting initCommunication",
                             mpi_config.rank);

    // 在 MPI 环境中初始化层次化 NCCL 通信组
    // 获取所有进程的 GPU ID
    std::vector<int> all_gpu_ids(mpi_config.size);

    // 收集所有进程的本地 GPU ID
    MPI_Allgather(&mpi_config.local_gpu_id, 1, MPI_INT, all_gpu_ids.data(), 1,
                  MPI_INT, MPI_COMM_WORLD);

    util::MpiLogger::println("Process {} completed MPI_Allgather",
                             mpi_config.rank);

    // 创建层次化通信组：[0,1,2,3] -> [1,2,3] -> [2,3] -> [3]
    // 这样设计可以让早完成的进程开始下一轮计算，实现流水线并行

    // 1. 主通信组：所有进程 [0,1,2,3,...,size-1]
    ncclUniqueId main_nccl_id;
    if (mpi_config.rank == 0) {
        ncclGetUniqueId(&main_nccl_id);
    }
    MPI_Bcast(&main_nccl_id, sizeof(main_nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    util::MpiLogger::println("Process {} completed main NCCL ID broadcast",
                             mpi_config.rank);

    ncclResult_t nccl_result = ncclCommInitRank(&nccl_comm, mpi_config.size,
                                                main_nccl_id, mpi_config.rank);
    if (nccl_result != ncclSuccess) {
        throw std::runtime_error(
            fmt::format("Main NCCL initialization failed: {}",
                        ncclGetErrorString(nccl_result)));
    }

    util::MpiLogger::println("Process {} completed main NCCL comm init",
                             mpi_config.rank);

    // 2. 创建子通信组：每个进程参与从自己开始到最后的通信组
    // 进程i参与通信组 [i, i+1, ..., size-1]
    sub_comm_groups.resize(mpi_config.size);
    sub_mpi_comms.resize(mpi_config.size);  // 同时创建对应的MPI子通信器

    util::MpiLogger::println("Process {} starting sub-group creation loop",
                             mpi_config.rank);

    for (int start_rank = 0; start_rank < mpi_config.size; start_rank++) {
        util::MpiLogger::println(
            "Process {} processing sub-group starting from rank {}",
            mpi_config.rank, start_rank);

        // 所有进程都参与MPI_Comm_split，但只有部分进程会被分配到有效的通信器
        int color =
            (mpi_config.rank >= start_rank) ? start_rank : MPI_UNDEFINED;

        util::MpiLogger::println(
            "Process {} calling MPI_Comm_split for group {} with color={}",
            mpi_config.rank, start_rank, color);

        MPI_Comm_split(MPI_COMM_WORLD, color, mpi_config.rank,
                       &sub_mpi_comms[start_rank]);

        util::MpiLogger::println(
            "Process {} completed MPI_Comm_split for group {}", mpi_config.rank,
            start_rank);

        if (mpi_config.rank >= start_rank) {
            int sub_group_size = mpi_config.size - start_rank;
            int sub_rank = mpi_config.rank - start_rank;

            util::MpiLogger::println(
                "Process {} creating sub-group {}: size={}, sub_rank={}",
                mpi_config.rank, start_rank, sub_group_size, sub_rank);

            // 只有当子组大小大于1时才创建NCCL通信器
            // 单进程组不需要NCCL通信
            if (sub_group_size > 1) {
                // 生成该子组的NCCL ID
                ncclUniqueId sub_nccl_id;
                if (mpi_config.rank == start_rank) {
                    ncclGetUniqueId(&sub_nccl_id);
                    util::MpiLogger::println(
                        "Process {} generated NCCL ID for group {}",
                        mpi_config.rank, start_rank);
                }

                // 在子组内广播NCCL ID
                util::MpiLogger::println(
                    "Process {} starting NCCL ID broadcast for group {}",
                    mpi_config.rank, start_rank);
                MPI_Bcast(&sub_nccl_id, sizeof(sub_nccl_id), MPI_BYTE, 0,
                          sub_mpi_comms[start_rank]);

                util::MpiLogger::println(
                    "Process {} completed sub-group NCCL ID broadcast for "
                    "group {}",
                    mpi_config.rank, start_rank);

                // 初始化子组NCCL通信器
                util::MpiLogger::println(
                    "Process {} initializing NCCL comm for group {}",
                    mpi_config.rank, start_rank);
                ncclResult_t sub_result =
                    ncclCommInitRank(&sub_comm_groups[start_rank],
                                     sub_group_size, sub_nccl_id, sub_rank);

                if (sub_result != ncclSuccess) {
                    throw std::runtime_error(fmt::format(
                        "Sub NCCL group {} initialization failed: {}",
                        start_rank, ncclGetErrorString(sub_result)));
                }

                util::MpiLogger::println(
                    "Process {} completed NCCL initialization for group {}",
                    mpi_config.rank, start_rank);
            } else {
                // 单进程组：不需要NCCL通信器
                sub_comm_groups[start_rank] = nullptr;
                util::MpiLogger::println(
                    "Process {} skipping NCCL creation for single-process "
                    "group {}",
                    mpi_config.rank, start_rank);
            }

            util::MpiLogger::println(
                "Process {} joined sub-group starting from rank {}, "
                "sub_rank={}, group_size={}",
                mpi_config.rank, start_rank, sub_rank, sub_group_size);
        } else {
            sub_comm_groups[start_rank] = nullptr;
            // sub_mpi_comms[start_rank]
            // 已经在MPI_Comm_split中设置为MPI_COMM_NULL
            util::MpiLogger::println(
                "Process {} not participating in sub-group {}", mpi_config.rank,
                start_rank);
        }
    }

    util::MpiLogger::println("Process {} completed initCommunication",
                             mpi_config.rank);
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

// 函数声明
template <typename T>
void performInterRecursiveSyr2k(matrix_ops::mpi::MpiSy2sbContext<T>& ctx,
                                size_t recursive_depth,
                                size_t recursive_offset_finished);

template <typename T>
void pipelineParallelScheduler(matrix_ops::mpi::MpiSy2sbContext<T>& ctx,
                               size_t current_phase, size_t total_phases);

template <typename T>
void hierarchicalBarrier(matrix_ops::mpi::MpiSy2sbContext<T>& ctx,
                         size_t min_participating_rank,
                         const std::string& operation_name);

// 内部函数实现
template <typename T>
void sy2sb_recursive_mpi(size_t recursive_depth,
                         matrix_ops::mpi::MpiSy2sbContext<T>& ctx) {
    // 计算递归偏移量
    size_t recursive_offset_finished = ctx.nb * recursive_depth;

    // 设置当前设备
    cudaSetDevice(ctx.mpi_config.local_gpu_id);

    util::MpiLogger::println(
        "Process {} 递归层数={}, recursive_offset_finished={}, 终止条件检查: "
        "{} <= {}+{} "
        "= {}",
        ctx.mpi_config.rank, recursive_depth, recursive_offset_finished, ctx.n,
        ctx.nb, recursive_offset_finished,
        ctx.n <= ctx.nb + recursive_offset_finished);

    // 面板循环处理，每次处理 b 列 (集成流水线并行优化)
    for (size_t i = ctx.b;
         i <= ctx.nb && i < (ctx.n - recursive_offset_finished); i += ctx.b) {
        size_t panel_m = ctx.n - recursive_offset_finished - i;
        size_t panel_n = ctx.b;

        // 计算当前面板在全局矩阵中的列偏移
        size_t global_panel_col = recursive_offset_finished + (i - ctx.b);

        util::MpiLogger::println(
            "Process {} entering panel loop: i={}, global_panel_col={}, "
            "panel_m={}, panel_n={}",
            ctx.mpi_config.rank, i, global_panel_col, panel_m, panel_n);

        // 流水线并行调度：检查是否可以并行执行
        size_t current_phase = (i - ctx.b) / ctx.b;
        size_t total_phases = ctx.nb / ctx.b;
        pipelineParallelScheduler(ctx, current_phase, total_phases);

        // 1. 面板分解 (Panel QR) - 只有拥有该面板的进程执行
        util::MpiLogger::println(
            "Process {} calling performPanelQR for global_panel_col={}",
            ctx.mpi_config.rank, global_panel_col);
        performPanelQR(ctx, global_panel_col, panel_m, panel_n, i,
                       recursive_offset_finished);

        // 2. MPI 通信和数据交换 - 使用层次化通信组
        exchangeDataMPI(ctx, global_panel_col, panel_m, panel_n, i);

        // 3. 矩阵更新 - 参照分布式版本的 WY 更新逻辑
        updateMatricesMPI(ctx, i, panel_m, panel_n, recursive_offset_finished);

        // 4. 更新 A 矩阵 (对应分布式版本的 i < nb 部分)
        if (i < ctx.nb) {
            updateAMatrixMPI(ctx, i, panel_m, panel_n,
                             recursive_offset_finished);
        }

        // 阶段完成同步：使用层次化Barrier优化
        size_t owner_rank = ctx.computeProcessForColumn(global_panel_col);
        hierarchicalBarrier(ctx, owner_rank,
                            fmt::format("phase-{}-complete", current_phase));
    }

    // 递归终止条件检查 (参照分布式版本的位置)
    if (ctx.n <= ctx.nb + recursive_offset_finished) {
        util::MpiLogger::println("递归终止：{} <= {} + {} = {}", ctx.n, ctx.nb,
                                 recursive_offset_finished,
                                 ctx.nb + recursive_offset_finished);
        return;
    }

    // TODO: 在这里需要添加类似分布式版本的矩阵更新逻辑 (syr2k等)
    // 这是递归间的关键多进程操作
    performInterRecursiveSyr2k(ctx, recursive_depth, recursive_offset_finished);

    util::MpiLogger::println("需要继续递归到下一层：recursive_depth={} -> {}",
                             recursive_depth, recursive_depth + 1);

    // 5. 递归调用下一层
    sy2sb_recursive_mpi(recursive_depth + 1, ctx);
}

template <typename T>
void performPanelQR(matrix_ops::mpi::MpiSy2sbContext<T>& ctx,
                    size_t global_panel_col, size_t panel_m, size_t panel_n,
                    size_t i, size_t recursive_offset_finished) {
    // 确定哪个进程拥有这个面板
    size_t owner_rank = ctx.computeProcessForColumn(global_panel_col);

    // 检查当前面板是否在本进程的列范围内
    if (ctx.isLocalColumn(global_panel_col)) {
        // 计算本地面板指针
        size_t local_col = ctx.getLocalColumnIndex(global_panel_col);
        auto panel_ptr = ctx.gpu_A.data() + local_col * ctx.n + i;
        auto panel_W_ptr = ctx.gpu_W.data() + local_col * ctx.n + i;
        auto panel_Y_ptr = ctx.gpu_Y.data() + local_col * ctx.n + i;
        auto R_ptr = ctx.gpu_R.data() + recursive_offset_finished;

        util::MpiLogger::println(
            "Panel is local: global_col={}, local_col={}, performing QR",
            global_panel_col, local_col);

        // 执行面板QR分解
        matrix_ops::internal::sy2sb::panelQR(
            ctx.cublas_handle, ctx.cusolver_handle, panel_m, panel_n, panel_ptr,
            ctx.n,              // 使用 n 作为 lda
            R_ptr, ctx.n,       // R 矩阵的 leading dimension
            panel_W_ptr, ctx.n  // W 矩阵的 leading dimension
        );

        // 复制面板数据到 Y 矩阵 (保存 Q 的表示)
        matrix_ops::matrix_copy<thrust::device_ptr<T>, thrust::device_ptr<T>,
                                T>(panel_ptr, ctx.n, panel_Y_ptr, ctx.n,
                                   panel_m, panel_n);

        // 复制 R 数据回面板位置
        matrix_ops::matrix_copy<thrust::device_ptr<T>, thrust::device_ptr<T>,
                                T>(R_ptr, ctx.n, panel_ptr, ctx.n, panel_m,
                                   panel_n);

        // 打印 Y 和 R 数据
        util::MpiLogger::println("Q {}:", i / ctx.b);
        matrix_ops::print(panel_Y_ptr, panel_m, panel_n, ctx.n, "Q");
        util::MpiLogger::println("R {}", i / ctx.b);
        matrix_ops::print(panel_Y_ptr, panel_m, panel_n, ctx.n, "R");

        util::MpiLogger::println("Panel QR completed successfully");
    }

    // 使用层次化Barrier：只有参与的进程需要等待
    // 这样可以让早完成的进程开始下一阶段工作
    hierarchicalBarrier(ctx, owner_rank, "Panel-QR");

    util::MpiLogger::println("Panel QR sync complete");
}

template <typename T>
void exchangeDataMPI(matrix_ops::mpi::MpiSy2sbContext<T>& ctx,
                     size_t global_panel_col, size_t panel_m, size_t panel_n,
                     size_t i) {
    // 使用层次化通信组进行更高效的数据广播
    // 这样可以实现流水线并行：早完成的进程可以开始下一轮计算

    auto nccl_type = std::is_same_v<T, float> ? ncclFloat32 : ncclFloat64;

    // 确定哪个进程拥有这个面板
    size_t owner_rank = ctx.computeProcessForColumn(global_panel_col);

    // 选择合适的通信组：从拥有面板的进程开始的子组
    // 例如：如果进程1拥有面板，则使用子组[1,2,3]进行通信
    ncclComm_t selected_comm = (owner_rank < ctx.sub_comm_groups.size() &&
                                ctx.sub_comm_groups[owner_rank] != nullptr)
                                   ? ctx.sub_comm_groups[owner_rank]
                                   : ctx.nccl_comm;

    // 计算在选定通信组中的根进程rank (相对于子组)
    int sub_root = 0;  // 在子组中，owner总是rank 0

    util::MpiLogger::println(
        "Using sub-communication group starting from rank {}, current process "
        "participating: {}",
        owner_rank, ctx.mpi_config.rank >= owner_rank);

    // 只有参与该子通信组的进程才进行通信
    if (ctx.mpi_config.rank >= owner_rank) {
        // 广播 W 面板数据
        if (ctx.mpi_config.rank == owner_rank) {
            size_t local_col = ctx.getLocalColumnIndex(global_panel_col);
            auto panel_W_ptr = ctx.gpu_W.data() + local_col * ctx.n + i;
            ncclBcast(panel_W_ptr.get(), panel_m * panel_n, nccl_type, sub_root,
                      selected_comm, ctx.stream);
        } else {
            // 接收到临时缓冲区
            auto work_W_ptr = ctx.gpu_work.data();
            ncclBcast(work_W_ptr.get(), panel_m * panel_n, nccl_type, sub_root,
                      selected_comm, ctx.stream);
        }

        // 广播 Y 面板数据
        if (ctx.mpi_config.rank == owner_rank) {
            size_t local_col = ctx.getLocalColumnIndex(global_panel_col);
            auto panel_Y_ptr = ctx.gpu_Y.data() + local_col * ctx.n + i;
            ncclBcast(panel_Y_ptr.get(), panel_m * panel_n, nccl_type, sub_root,
                      selected_comm, ctx.stream);
        } else {
            // 接收到临时缓冲区
            auto work_Y_ptr = ctx.gpu_work.data() + ctx.n * ctx.nb;
            ncclBcast(work_Y_ptr.get(), panel_m * panel_n, nccl_type, sub_root,
                      selected_comm, ctx.stream);
        }

        cudaStreamSynchronize(ctx.stream);
    }

    // 不参与该子组的进程可以继续做其他工作
    // 例如：当进程1拥有面板时，进程0可以开始准备下一轮计算
}

// 多进程并行计算 A*W (类似分布式版本的 computeAwMgpu)
template <typename T>
void computeAwMPI(matrix_ops::mpi::MpiSy2sbContext<T>& ctx, size_t i,
                  size_t panel_m, size_t panel_n,
                  size_t recursive_offset_finished, size_t global_panel_col) {
    // 每个进程处理自己的本地矩阵块
    // 类似分布式版本中每个GPU处理自己的矩阵块

    auto nccl_type = std::is_same_v<T, float> ? ncclFloat32 : ncclFloat64;
    size_t owner_rank = ctx.computeProcessForColumn(global_panel_col);

    // 获取当前面板的W数据指针
    thrust::device_ptr<T> panel_W_ptr;
    if (ctx.mpi_config.rank == owner_rank) {
        size_t local_col = ctx.getLocalColumnIndex(global_panel_col);
        panel_W_ptr = ctx.gpu_W.data() + local_col * ctx.n + i;
    } else {
        // 使用从广播中接收到的W数据
        panel_W_ptr = ctx.gpu_work.data();
    }

    // 计算本地矩阵块的 A*W
    auto local_oriA_ptr = ctx.gpu_oriA.data() + i + recursive_offset_finished;
    auto local_aw_ptr =
        ctx.gpu_work.data() + ctx.n * ctx.nb;  // 使用work空间的后半部分

    // 执行 GEMM: local_aw = local_oriA * panel_W
    matrix_ops::gemm(ctx.cublas_handle, ctx.cols_per_process, panel_n, panel_m,
                     T(1.0), local_oriA_ptr, ctx.n, false, panel_W_ptr, ctx.n,
                     false, T(0.0), local_aw_ptr, ctx.cols_per_process);

    // 将结果收集到拥有面板的进程
    if (ctx.mpi_config.rank == owner_rank) {
        // 接收其他进程的AW结果
        // TODO: 实现MPI gather操作来收集所有进程的AW结果
    } else {
        // 发送AW结果到拥有面板的进程
        // TODO: 实现MPI send操作
    }
}

// 单进程的WY更新逻辑 (类似分布式版本面板循环内的gemm序列)
template <typename T>
void updateWYLogic(matrix_ops::mpi::MpiSy2sbContext<T>& ctx, size_t i,
                   size_t panel_m, size_t panel_n,
                   size_t recursive_offset_finished, size_t global_panel_col) {
    // 只有拥有当前面板的进程执行WY更新逻辑
    if (!ctx.isLocalColumn(global_panel_col)) {
        return;
    }

    size_t local_col = ctx.getLocalColumnIndex(global_panel_col);
    auto panel_W_ptr = ctx.gpu_W.data() + local_col * ctx.n + i;
    auto panel_Y_ptr = ctx.gpu_Y.data() + local_col * ctx.n + i;
    auto panel_Z_ptr = ctx.gpu_Z.data() + i + (i - ctx.b) * ctx.n;
    auto work_ptr = ctx.gpu_work.data();

    if (i == ctx.b) {
        // 第一个面板的处理逻辑
        // panel_tmp = panel_W^T * panel_Z
        matrix_ops::gemm(ctx.cublas_handle, panel_n, panel_n, panel_m, T(1.0),
                         panel_W_ptr, ctx.n, true, panel_Z_ptr, ctx.n, false,
                         T(0.0), work_ptr, panel_n);

        // panel_Z = panel_Z - 0.5 * panel_Y * panel_tmp
        matrix_ops::gemm(ctx.cublas_handle, panel_m, panel_n, panel_n, T(-0.5),
                         panel_Y_ptr, ctx.n, false, work_ptr, panel_n, false,
                         T(1.0), panel_Z_ptr, ctx.n);
    } else {
        // 后续面板的处理逻辑 (参照分布式版本的复杂gemm序列)
        auto Z_prev = ctx.gpu_Z.data() + i;
        auto Y_prev = ctx.gpu_Y.data() + i;

        // panel_tmp = Z_prev^T * panel_W
        matrix_ops::gemm(ctx.cublas_handle, i - ctx.b, panel_n, panel_m, T(1.0),
                         Z_prev, ctx.n, true, panel_W_ptr, ctx.n, false, T(0.0),
                         work_ptr, i - ctx.b);

        // panel_Z = panel_Z - Y_prev * panel_tmp
        matrix_ops::gemm(ctx.cublas_handle, panel_m, panel_n, i - ctx.b,
                         T(-1.0), Y_prev, ctx.n, false, work_ptr, i - ctx.b,
                         false, T(1.0), panel_Z_ptr, ctx.n);

        // 继续其他gemm操作...
        // (这里需要参照分布式版本的完整gemm序列)
    }
}

template <typename T>
void updateMatricesMPI(matrix_ops::mpi::MpiSy2sbContext<T>& ctx, size_t i,
                       size_t panel_m, size_t panel_n,
                       size_t recursive_offset_finished) {
    size_t global_panel_col = recursive_offset_finished + (i - ctx.b);

    // 1. 多进程并行计算 A*W
    computeAwMPI(ctx, i, panel_m, panel_n, recursive_offset_finished,
                 global_panel_col);

    // 2. 单进程的WY更新逻辑
    updateWYLogic(ctx, i, panel_m, panel_n, recursive_offset_finished,
                  global_panel_col);
}

template <typename T>
void updateAMatrixMPI(matrix_ops::mpi::MpiSy2sbContext<T>& ctx, size_t i,
                      size_t panel_m, size_t panel_n,
                      size_t recursive_offset_finished) {
    // 对应分布式版本中 if (i < nb) 部分的两个 gemm 操作
    // 这些是单进程操作，只在拥有相应面板的进程上执行

    size_t global_panel_col = recursive_offset_finished + (i - ctx.b);
    if (!ctx.isLocalColumn(global_panel_col)) {
        return;  // 其他进程不需要执行这些操作
    }

    size_t local_col = ctx.getLocalColumnIndex(global_panel_col);
    auto panel_ptr = ctx.gpu_A.data() + local_col * ctx.n + i;
    auto Y_prev = ctx.gpu_Y.data() + i;
    auto Z_prev = ctx.gpu_Z.data() + i;

    // 第一个gemm操作
    matrix_ops::gemm(ctx.cublas_handle, panel_m, ctx.b, i, T(-1.0), Y_prev,
                     ctx.n, false, ctx.gpu_R.data(), ctx.n, false, T(1.0),
                     panel_ptr, ctx.n);

    // 第二个gemm操作
    matrix_ops::gemm(ctx.cublas_handle, panel_m, ctx.b, i, T(-1.0), Z_prev,
                     ctx.n, false, ctx.gpu_R.data(), ctx.n, false, T(1.0),
                     panel_ptr, ctx.n);
}

// 递归间的syr2k操作 (多进程并行，使用层次化通信优化)
template <typename T>
void performInterRecursiveSyr2k(matrix_ops::mpi::MpiSy2sbContext<T>& ctx,
                                size_t recursive_depth,
                                size_t recursive_offset_finished) {
    auto nccl_type = std::is_same_v<T, float> ? ncclFloat32 : ncclFloat64;
    size_t sub_matrix_n = ctx.n - recursive_offset_finished - ctx.nb;

    if (sub_matrix_n <= 0) return;

    // 使用层次化通信组进行更高效的数据分发
    // 每个子组可以独立进行数据交换，减少通信开销

    // 1. 准备Z和W数据用于syr2k
    auto local_Z_ptr = ctx.gpu_Z.data() + ctx.nb;  // 跳过前nb行
    auto local_W_ptr = ctx.gpu_work.data();

    // 2. 使用层次化AllReduce：从大组到小组逐步聚合
    // 这样可以减少通信延迟，提高带宽利用率
    for (int group_start = 0; group_start < ctx.mpi_config.size;
         group_start++) {
        if (ctx.mpi_config.rank >= group_start &&
            ctx.sub_comm_groups[group_start] != nullptr) {
            util::MpiLogger::println(
                "Process {} participating in sub-group {} AllReduce",
                ctx.mpi_config.rank, group_start);

            // 在该子组内进行AllReduce
            ncclAllReduce(local_Z_ptr.get(), local_Z_ptr.get(),
                          sub_matrix_n * ctx.nb, nccl_type, ncclSum,
                          ctx.sub_comm_groups[group_start], ctx.stream);

            ncclAllReduce(local_W_ptr.get(), local_W_ptr.get(),
                          sub_matrix_n * ctx.nb, nccl_type, ncclSum,
                          ctx.sub_comm_groups[group_start], ctx.stream);

            cudaStreamSynchronize(ctx.stream);
            break;  // 只参与一个合适的子组
        }
    }

    // 3. 执行本地的syr2k操作
    auto tail_matrix_ptr =
        ctx.gpu_A.data() + (ctx.n - sub_matrix_n) * ctx.cols_per_process;

    matrix_ops::syr2k(ctx.cublas_handle, sub_matrix_n, ctx.nb, T(-1),
                      local_W_ptr, sub_matrix_n, local_Z_ptr, sub_matrix_n,
                      T(1), tail_matrix_ptr, ctx.n);

    // 4. 确保对称性 (类似分布式版本的make_symmetric_functor)
    // TODO: 实现make_symmetric操作

    // 5. 复制结果到oriA矩阵
    matrix_ops::matrix_copy<thrust::device_ptr<T>, thrust::device_ptr<T>, T>(
        tail_matrix_ptr, ctx.n,
        ctx.gpu_oriA.data() + (ctx.n - sub_matrix_n) * ctx.cols_per_process,
        ctx.n, sub_matrix_n, ctx.cols_per_process);

    util::MpiLogger::println(
        "Process {} completed syr2k with hierarchical communication",
        ctx.mpi_config.rank);
}

// 新增：流水线并行调度函数
template <typename T>
void pipelineParallelScheduler(matrix_ops::mpi::MpiSy2sbContext<T>& ctx,
                               size_t current_phase, size_t total_phases) {
    // 根据当前阶段和进程rank决定是否可以开始下一阶段的工作

    size_t min_participating_rank = current_phase % ctx.mpi_config.size;

    if (ctx.mpi_config.rank < min_participating_rank) {
        // 该进程已经完成当前阶段，可以开始准备下一阶段
        util::MpiLogger::println(
            "Process {} starting next phase while others finish current phase",
            ctx.mpi_config.rank);

        // 在这里可以添加预取数据、预分配内存等优化操作
        // 例如：预计算下一轮需要的矩阵块

    } else {
        // 该进程仍需参与当前阶段
        util::MpiLogger::println("Process {} continuing current phase {}",
                                 ctx.mpi_config.rank, current_phase);
    }
}

// 新增：层次化子通信组Barrier函数
template <typename T>
void hierarchicalBarrier(matrix_ops::mpi::MpiSy2sbContext<T>& ctx,
                         size_t min_participating_rank,
                         const std::string& operation_name) {
    // 添加调试信息
    util::MpiLogger::println(
        "Process {} checking barrier for {}: min_rank={}, rank>={}, size={}, "
        "comm_null={}",
        ctx.mpi_config.rank, operation_name, min_participating_rank,
        ctx.mpi_config.rank >= min_participating_rank, ctx.sub_mpi_comms.size(),
        min_participating_rank < ctx.sub_mpi_comms.size()
            ? (ctx.sub_mpi_comms[min_participating_rank] == MPI_COMM_NULL)
            : true);

    // 只有参与该阶段的进程需要同步
    if (ctx.mpi_config.rank >= min_participating_rank &&
        min_participating_rank < ctx.sub_mpi_comms.size() &&
        ctx.sub_mpi_comms[min_participating_rank] != MPI_COMM_NULL) {
        util::MpiLogger::println(
            "Process {} syncing with sub-group [{},...,{}] after {}",
            ctx.mpi_config.rank, min_participating_rank,
            ctx.mpi_config.size - 1, operation_name);
        MPI_Barrier(ctx.sub_mpi_comms[min_participating_rank]);
        util::MpiLogger::println("Process {} completed barrier for {}",
                                 ctx.mpi_config.rank, operation_name);
    } else if (ctx.mpi_config.rank < min_participating_rank) {
        // 早完成的进程可以继续下一阶段工作
        util::MpiLogger::println(
            "Process {} skipping barrier for {} (already finished this phase)",
            ctx.mpi_config.rank, operation_name);
    } else {
        // 调试：其他情况
        util::MpiLogger::println(
            "Process {} cannot participate in barrier for {} (comm not "
            "available)",
            ctx.mpi_config.rank, operation_name);
    }
}

}  // namespace internal

/**
 * @brief MPI 版本的 sy2sb 主函数
 */
template <typename T>
void sy2sb(const MpiConfig& mpi_config, size_t n, T* A, size_t lda, T* W,
           size_t ldw, T* Y, size_t ldy, size_t nb, size_t b) {
    util::MpiLogger::println("Process {} entering sy2sb function",
                             mpi_config.rank);

    // 检查分块兼容性（与分布式版本保持一致）
    if (n % b % mpi_config.size != 0) {
        throw std::runtime_error(
            "Matrix is not well divisible into MPI process panels");
    }

    util::MpiLogger::println("Process {} creating MPI sy2sb context",
                             mpi_config.rank);

    // 创建 MPI sy2sb 上下文
    MpiSy2sbContext<T> ctx(mpi_config, n, A, lda, W, ldw, Y, ldy, nb, b);

    util::MpiLogger::println("Process {} calling recursive implementation",
                             mpi_config.rank);

    // 调用递归实现
    internal::sy2sb_recursive_mpi<T>(0, ctx);

    util::MpiLogger::println("Process {} completed recursive implementation",
                             mpi_config.rank);

    // 最终同步：确保所有进程完成
    // 使用层次化同步：从小组到大组逐步同步，最后全局同步
    util::MpiLogger::println(
        "Process {} starting final hierarchical synchronization",
        mpi_config.rank);

    // 先在各自的子组内同步
    for (int group_start = mpi_config.size - 1; group_start >= 0;
         group_start--) {
        if (mpi_config.rank >= group_start &&
            group_start < ctx.sub_mpi_comms.size() &&
            ctx.sub_mpi_comms[group_start] != MPI_COMM_NULL) {
            util::MpiLogger::println(
                "Process {} syncing with sub-group [{},...,{}]",
                mpi_config.rank, group_start, mpi_config.size - 1);
            MPI_Barrier(ctx.sub_mpi_comms[group_start]);
            break;  // 只参与一个子组同步
        }
    }

    // 最后全局同步确保所有进程完成
    util::MpiLogger::println("Process {} final global synchronization",
                             mpi_config.rank);
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
