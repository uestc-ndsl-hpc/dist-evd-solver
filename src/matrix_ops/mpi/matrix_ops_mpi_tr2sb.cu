#include <mpi.h>
#include <nccl.h>
#include <thrust/device_vector.h>
#include <thrust/universal_vector.h>

#include <cstddef>

#include "log.h"
#include "matrix_ops.cuh"
#include "matrix_ops_mpi.cuh"

namespace matrix_ops {
namespace mpi {

// MpiTr2sbGenQContext 类方法实现
template <typename T>
MpiTr2sbGenQContext<T>::MpiTr2sbGenQContext(const MpiConfig& config,
                                            Sy2sbResultBuffers<T>& buffers,
                                            Tr2sbBuffers<T>& tr2sbBuffer,
                                            thrust::host_vector<T>& U_h)
    : mpi_config(config),
      n(buffers.n),
      U(std::move(U_h)),
      b(buffers.b),
      gpu_Q(std::move(tr2sbBuffer.Q)),
      ldQ(tr2sbBuffer.ldQ),
      stream(buffers.stream) {
    // 计算分块信息
    cols_per_process = n / mpi_config.size;

    // Initialize q_cols vector with proper size
    q_cols.resize(mpi_config.size);
    std::fill(q_cols.begin(), q_cols.end(), cols_per_process);

    // 只有rank==0才会使用
    ldU = n + mpi_config.total_gpus * (2 * b);

    ldSubU = cols_per_process + 2 * b;

    sweepCount = (n - 1 - 1 + (U_LEN_PROC_1TIME - 1)) / (U_LEN_PROC_1TIME);

    lastSweepUCount = n - ((sweepCount - 1) * U_LEN_PROC_1TIME + 1) - 1;

    // // 设置当前进程使用的 GPU
    cudaSetDevice(mpi_config.local_gpu_id);
}

template <typename T>
void MpiTr2sbGenQContext<T>::initializeQMatrix() {
    // Pre-calculate base offset for current process
    size_t base_offset = 0;
    for (int i = 0; i < mpi_config.rank; i++) {
        base_offset += q_cols[i];
    }

    // Copy member variables to local variables for device lambda
    auto local_n = n;
    auto local_base_offset = base_offset;
    auto gpu_Q_ptr = thrust::raw_pointer_cast(gpu_Q.data());

    thrust::for_each(
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(q_cols[mpi_config.rank] * n),
        [=] __device__(size_t idx) {
            auto row = idx % local_n;
            auto col = idx / local_n;
            if (local_base_offset + col == row) {
                gpu_Q_ptr[idx] = static_cast<T>(1.0);
            } else {
                gpu_Q_ptr[idx] = static_cast<T>(0.0);
            }
        });
}

template <typename T>
MpiTr2sbGenQContext<T>::~MpiTr2sbGenQContext() {
    // 确保所有CUDA操作完成
    cudaDeviceSynchronize();

    // 销毁 CUDA 流
    if (stream != nullptr) {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }
}

// 添加函数的说明
template <typename T>
void bcBack(MpiTr2sbGenQContext<T>& ctx) {
    int sweepCount = ctx.sweepCount;

    // 计算最后1趟有多少个u
    // 最后1趟的起始位置
    int lastSweepUCount = ctx.lastSweepUCount;

    ssize_t shareDyMem =
        U_COL_EXRTERN_COUNT * U_LEN_PROC_1TIME * sizeof(T);  // 动态共享内存大小
    cudaFuncSetAttribute(matrix_ops::tr2sb::BC_kernel_computerQ_1Col_V8_10<T>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shareDyMem);

    dim3 dimBlock(32, MAX_WARP_COUNT, 1);

    int numBlocksPerSm = 0;

    int numThreads = 32 * MAX_WARP_COUNT;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, ctx.mpi_config.local_gpu_id);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocksPerSm, matrix_ops::tr2sb::BC_kernel_computerQ_1Col_V8_10<T>,
        numThreads, shareDyMem);

    util::MpiLogger::println(
        "BC-back numBlocksPerSm: {}, deviceProp.multiProcessorCount: {}",
        numBlocksPerSm, deviceProp.multiProcessorCount);

    int blockNum = numBlocksPerSm * deviceProp.multiProcessorCount;

    int perBlockN = ctx.cols_per_process / blockNum;
    int largeBlockNum = ctx.cols_per_process % blockNum;

    dim3 dimGrid(blockNum, 1, 1);

    size_t n = ctx.n;
    size_t ldsubU = U_LEN_PROC_1TIME;

    // 进行内存的分配
    int n_assignment = (n + U_COL_EXRTERN_COUNT - 1) / U_COL_EXRTERN_COUNT *
                       U_COL_EXRTERN_COUNT;

    int device = -1;
    cudaGetDevice(&device);
    util::MpiLogger::println(
        "BC-BACK device {}, U_COL_EXRTERN_COUNT : {}, MAX_WARP_COUNT: {}, "
        "shareDyMem:{}",
        device, U_COL_EXRTERN_COUNT, MAX_WARP_COUNT, shareDyMem);

    ctx.subU_rev.resize(n_assignment * ldsubU);
    ctx.gpu_subU.resize(n_assignment * ldsubU);

    auto p_subU_rev = thrust::raw_pointer_cast(ctx.subU_rev.data());
    auto p_subU = thrust::raw_pointer_cast(ctx.subU.data());

    auto p_gpu_subU = thrust::raw_pointer_cast(ctx.gpu_subU.data());

    ctx.gpu_Q.resize((n + 2 * U_LEN_PROC_1TIME) * ctx.cols_per_process);
    auto Q = thrust::raw_pointer_cast(ctx.gpu_Q.data());
    size_t ldQ = n + 2 * U_LEN_PROC_1TIME;

    auto stream = ctx.stream;

    size_t base = 0;
    for (int sweepIndex = 0, tmp = lastSweepUCount; sweepIndex < sweepCount;
         sweepIndex++, tmp += U_LEN_PROC_1TIME) {
        // 广播数据
        int tmp2 = (tmp + U_COL_EXRTERN_COUNT - 1) / U_COL_EXRTERN_COUNT *
                   U_COL_EXRTERN_COUNT;

        // 复制数据
        if (0 == ctx.mpi_config.rank) {
            memcpy(p_subU_rev, p_subU + base * ldsubU,
                   tmp2 * ldsubU * sizeof(T));
        }

        MPI_Bcast(p_subU_rev, tmp2 * ldsubU,
                  std::is_same_v<T, float> ? MPI_FLOAT : MPI_DOUBLE, 0,
                  MPI_COMM_WORLD);

        cudaMemcpy(p_gpu_subU, p_subU_rev, tmp2 * ldsubU * sizeof(T),
                   cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();

        void* kernelArgs[] = {(void*)&n,
                              (void*)&perBlockN,
                              (void*)&largeBlockNum,
                              (void*)&sweepCount,
                              (void*)&lastSweepUCount,
                              (void*)&sweepIndex,
                              (void*)&p_gpu_subU,
                              (void*)&Q,
                              (void*)&ldQ};

        cudaLaunchCooperativeKernel(
            (void*)matrix_ops::tr2sb::BC_kernel_computerQ_1Col_V8_10<T>,
            dimGrid, dimBlock, kernelArgs, shareDyMem, stream);

        base += tmp2;
    }

    cudaDeviceSynchronize();

    return;
}

// 将BC生成的U矩阵拷贝到条带化矩阵当中
template <typename T>
void copyU2banMatrix(MpiTr2sbGenQContext<T>& ctx) {
    size_t n = ctx.n;
    size_t b = ctx.b;

    // size_t ldU = ctx.ldU;

    // 计算总共有多少个U
    // 计算总共有多少趟BC
    int sweepCount = ctx.sweepCount;

    // 计算最后1趟BC的位置
    int lastSweepUCount = ctx.lastSweepUCount;

    long countU = 0;
    for (int i = 0, tmp = lastSweepUCount; i < sweepCount;
         i++, tmp += U_LEN_PROC_1TIME) {
        int tmp2 = (tmp + U_COL_EXRTERN_COUNT - 1) / U_COL_EXRTERN_COUNT *
                   U_COL_EXRTERN_COUNT;
        countU += tmp2;
    }

    // 只有rank==0才会进入
    ctx.subU.resize(countU * U_LEN_PROC_1TIME);

    // 将数据从dU中拷贝到条带化u矩阵中
    auto pSubU_base = thrust::raw_pointer_cast(ctx.subU.data());

    auto p_u = thrust::raw_pointer_cast(ctx.U.data());

    auto cols_perPE = ctx.cols_per_process;

    size_t ldSubU = ctx.ldSubU;

    // 1.3 连续存储
    for (int i = 0; i < sweepCount; i++) {
        // 1.3.2 存储u的起始位置
        int base = (sweepCount - i - 1) * U_LEN_PROC_1TIME + 1;
        int uCount =
            i * U_LEN_PROC_1TIME + lastSweepUCount;  // 这个是计算有多少列的U

        for (int j = 0; j < uCount; j++) {
            for (int k = 0; k < U_COUNT; k++) {
                size_t rowIndexS = base + k * b;

                // 只拷贝合法的u
                if (rowIndexS > n - 1) {
                    break;
                }

                int colB1 = b;
                if (sweepCount - 1 == i) {
                    colB1 = 1;
                }

                // 计算在U的什么位置--也就是在那个rank的U上
                int localRank = (rowIndexS - colB1) / cols_perPE;

                memcpy(pSubU_base + k * b,
                       p_u + (localRank * ldSubU * n) + rowIndexS -
                           (localRank * cols_perPE) + j * ldSubU,
                       b * sizeof(T));
            }

            pSubU_base += U_LEN_PROC_1TIME;

            // 下1列的u的起始位置
            base += 1;
        }

        // 填充到U_COL_EXRTERN_COUNT的整数个u
        int tmp2 = (uCount + U_COL_EXRTERN_COUNT - 1) / U_COL_EXRTERN_COUNT *
                   U_COL_EXRTERN_COUNT;
        pSubU_base += (tmp2 - uCount) * U_LEN_PROC_1TIME;
    }

    // 是否可以理解为释放内存
    ctx.U.resize(0);
}

template <typename T>
void tr2sbGenQ(MpiTr2sbGenQContext<T>& ctx) {
    if (0 == ctx.mpi_config.rank) {
        util::MpiLogger::tic("BC-Back_U_bandnize");
        matrix_ops::mpi::copyU2banMatrix(ctx);
        util::MpiLogger::toc("BC-Back_U_bandnize");
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // 调用核函数进行处理
    util::MpiLogger::tic("BC-Back_compute");
    matrix_ops::mpi::bcBack(ctx);
    util::MpiLogger::toc("BC-Back_compute");

    return;
}

}  // namespace mpi

// 显式模板实例化
template class matrix_ops::mpi::MpiTr2sbGenQContext<float>;
template class matrix_ops::mpi::MpiTr2sbGenQContext<double>;

template void matrix_ops::mpi::tr2sbGenQ<float>(
    matrix_ops::mpi::MpiTr2sbGenQContext<float>& context);

template void matrix_ops::mpi::tr2sbGenQ<double>(
    matrix_ops::mpi::MpiTr2sbGenQContext<double>& context);

}  // namespace matrix_ops
