#include <mpi.h>
#include <nccl.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstddef>
#include <exception>

#include "log.h"
#include "matrix_ops.cuh"
#include "matrix_ops_mpi.cuh"

namespace matrix_ops {
namespace mpi {

// MpiSb2syGenQContext 类方法实现
template <typename T>
MpiSb2syGenQContext<T>::MpiSb2syGenQContext(const MpiConfig& config,
                                            Sy2sbResultBuffers<T>& buffers)
    : mpi_config(config),
      n(buffers.n),
      lda(buffers.lda),
      ldw(buffers.ldw),
      ldy(buffers.ldy),
      nb(buffers.nb),
      b(buffers.b),
      cublas_handle(std::move(buffers.cublas_handle)),
      cusolver_handle(std::move(buffers.cusolver_handle)),
      stream(buffers.stream),
      gpu_W(std::move(buffers.W)),
      gpu_Y(std::move(buffers.Y)),
      nccl_comm(buffers.nccl_comm),
      sub_comm_groups(std::move(buffers.sub_comm_groups)),
      sub_mpi_comms(std::move(buffers.sub_mpi_comms)) {
    // 计算分块信息
    cols_per_process = n / mpi_config.size;
    start_col = mpi_config.rank * cols_per_process;
    local_matrix_size = cols_per_process * n;

    // Initialize q_cols vector with proper size
    q_cols.resize(mpi_config.size);
    std::fill(q_cols.begin(), q_cols.end(), cols_per_process);

    // 设置 NCCL 数据类型
    if constexpr (std::is_same_v<T, float>) {
        nccl_type = ncclFloat32;
    } else if constexpr (std::is_same_v<T, double>) {
        nccl_type = ncclFloat64;
    }

    // 设置当前进程使用的 GPU
    cudaSetDevice(mpi_config.local_gpu_id);

    gpu_work.resize(local_matrix_size);

    // 这里可以考虑一下设计, 方便负载均衡
    gpu_Q.resize(q_cols[mpi_config.rank] * (n + U_LEN_PROC_1TIME));
    gpu_W_rec.resize(local_matrix_size);
    gpu_Y_rec.resize(local_matrix_size);

    // Initialize Q matrix as identity matrix
    initializeQMatrix();

    // 清空原始缓冲区的引用，避免重复释放
    buffers.nccl_comm = nullptr;
    buffers.sub_comm_groups.clear();
    buffers.sub_mpi_comms.clear();
    buffers.stream = nullptr;
}

template <typename T>
void MpiSb2syGenQContext<T>::initializeQMatrix() {
    // Pre-calculate base offset for current process
    size_t base_offset = 0;
    for (int i = 0; i < mpi_config.rank; i++) {
        base_offset += q_cols[i];
    }

    // Copy member variables to local variables for device lambda
    auto local_base_offset = base_offset;
    auto gpu_Q_ptr = thrust::raw_pointer_cast(gpu_Q.data());
    auto ldQ = U_LEN_PROC_1TIME + n;
    auto local_q_cols = q_cols[mpi_config.rank];
    auto local_n = n;  // capture n as a plain value for device lambda

    thrust::for_each(
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(local_q_cols * ldQ),
        [=] __device__(size_t idx) {
            auto row = idx % ldQ;
            auto col = idx / ldQ;
            // 检查边界：确保不访问超出矩阵范围的内存
            if (row < local_n && col < local_q_cols) {
                if (local_base_offset + col == row) {
                    gpu_Q_ptr[idx] = static_cast<T>(1.0);
                } else {
                    gpu_Q_ptr[idx] = static_cast<T>(0.0);
                }
            } else {
                // 对于超出矩阵范围的填充区域，设置为0
                gpu_Q_ptr[idx] = static_cast<T>(0.0);
            }
        });
}

template <typename T>
MpiSb2syGenQContext<T>::~MpiSb2syGenQContext() {
    // 确保所有CUDA操作完成
    cudaDeviceSynchronize();

    // 销毁所有子 NCCL 通信器
    for (size_t i = 0; i < sub_comm_groups.size(); i++) {
        if (sub_comm_groups[i] != nullptr) {
            ncclCommDestroy(sub_comm_groups[i]);
            sub_comm_groups[i] = nullptr;
        }
    }
    sub_comm_groups.clear();

    // 销毁主 NCCL 通信器
    if (nccl_comm != nullptr) {
        ncclCommDestroy(nccl_comm);
        nccl_comm = nullptr;
    }

    // 销毁所有子 MPI 通信器
    if (!sub_mpi_comms.empty()) {
        for (int start_rank = 0;
             start_rank < mpi_config.size &&
             start_rank < static_cast<int>(sub_mpi_comms.size());
             start_rank++) {
            if (mpi_config.rank >= start_rank) {
                // 检查通信器是否有效且不是MPI_COMM_NULL
                if (sub_mpi_comms[start_rank] != MPI_COMM_NULL) {
                    try {
                        MPI_Comm_free(&sub_mpi_comms[start_rank]);
                    } catch (...) {
                        // 如果出现任何异常，设置为NULL并继续
                        sub_mpi_comms[start_rank] = MPI_COMM_NULL;
                    }
                }
            }
        }
        sub_mpi_comms.clear();
    }

    // 销毁 CUDA 流
    if (stream != nullptr) {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }
}

template <typename T>
void sb2syGenQ(MpiSb2syGenQContext<T>& ctx) {
    auto W = ctx.gpu_W.data() + ctx.mpi_config.rank * ctx.cols_per_process;
    auto Y = ctx.gpu_Y.data() + ctx.mpi_config.rank * ctx.cols_per_process;
    auto work = ctx.gpu_work.data();
    auto m = ctx.n - ctx.mpi_config.rank * ctx.cols_per_process;
    auto b = ctx.b;
    auto ldW = ctx.ldw;
    auto ldY = ctx.ldy;

    auto nk = ctx.cols_per_process;

    // Extract type-dependent constants
    T done, dzero, dnegone;
    cudaDataType_t cuda_type;
    cublasComputeType_t compute_type;

    // Create compute stream for overlapping communication and computation
    cudaStream_t compute_stream;
    cudaStreamCreate(&compute_stream);

    util::MpiLogger::tic("sb2syGenW");

    if constexpr (std::is_same_v<T, float>) {
        done = 1.0f;
        dzero = 0.0f;
        dnegone = -1.0f;
        cuda_type = CUDA_R_32F;
        compute_type = CUBLAS_COMPUTE_32F;
    } else {
        done = 1.0;
        dzero = 0.0;
        dnegone = -1.0;
        cuda_type = CUDA_R_64F;
        compute_type = CUBLAS_COMPUTE_64F;
    }

    for (auto col_wk = b; col_wk < nk; col_wk *= 2) {
        cublasGemmStridedBatchedEx(
            ctx.cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, col_wk, col_wk, m - b,
            &done, Y.get() + b, cuda_type, ldY, 2 * col_wk * ldY,
            W.get() + b + col_wk * ldW, cuda_type, ldW, 2 * col_wk * ldW,
            &dzero, work.get(), cuda_type, m, 2 * col_wk * m, nk / (2 * col_wk),
            compute_type, CUBLAS_GEMM_DEFAULT);

        cublasGemmStridedBatchedEx(
            ctx.cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m - b, col_wk, col_wk,
            &dnegone, W.get() + b, cuda_type, ldW, 2 * col_wk * ldW, work.get(),
            cuda_type, m, 2 * col_wk * m, &done, W.get() + b + col_wk * ldW,
            cuda_type, ldW, 2 * col_wk * ldW, nk / (2 * col_wk), compute_type,
            CUBLAS_GEMM_DEFAULT);
    }

    util::MpiLogger::toc("sb2syGenW");

    MPI_Barrier(MPI_COMM_WORLD);

    util::MpiLogger::tic("sb2syGenQT");

    auto ldQ = ctx.n + U_LEN_PROC_1TIME;

    // (I - WYT)Q 左乘, 是一个依次的发送到乘的故事 就是 Q - W (YTQ)
    for (auto i = 0; i < ctx.mpi_config.size; ++i) {
        auto wy_gpu_id = i;
        if (wy_gpu_id == ctx.mpi_config.local_gpu_id) {
            matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                    thrust::device_ptr<T>, T>(
                Y, ctx.ldy, ctx.gpu_Y_rec.data(), m, m, ctx.cols_per_process);
            matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                    thrust::device_ptr<T>, T>(
                W, ctx.ldw, ctx.gpu_W_rec.data(), m, m, ctx.cols_per_process);
        }
        // 计算发送方进程的 m 值，确保所有进程使用相同的广播数据量
        auto sender_m = ctx.n - wy_gpu_id * ctx.cols_per_process;

        // Start W matrix broadcast (asynchronous)
        ncclBcast(ctx.gpu_W_rec.data().get(), sender_m * ctx.cols_per_process,
                  ctx.nccl_type, wy_gpu_id, ctx.nccl_comm, ctx.stream);

        // Prepare W data in communication stream (asynchronous)
        thrust::fill(thrust::cuda::par.on(ctx.stream), ctx.gpu_work.begin(),
                     ctx.gpu_work.end(), static_cast<T>(0.0));
        matrix_ops::matrix_copy<thrust::device_ptr<T>, thrust::device_ptr<T>,
                                T>(
            ctx.gpu_W_rec.data(), sender_m,
            ctx.gpu_work.data() + wy_gpu_id * ctx.cols_per_process, ctx.n,
            sender_m, ctx.cols_per_process);

        // Record event when W data is ready
        cudaEvent_t w_data_ready;
        cudaEventCreate(&w_data_ready);
        cudaEventRecord(w_data_ready, ctx.stream);

        // Start Y matrix broadcast (asynchronous) - can overlap with W
        // computation
        ncclBcast(ctx.gpu_Y_rec.data().get(), sender_m * ctx.cols_per_process,
                  ctx.nccl_type, wy_gpu_id, ctx.nccl_comm, ctx.stream);

        // Wait for W data and perform YTQ GEMM in compute stream
        cudaStreamWaitEvent(compute_stream, w_data_ready, 0);

        // Save original cublas stream and set to compute stream
        cudaStream_t original_stream;
        cublasGetStream(ctx.cublas_handle, &original_stream);
        cublasSetStream(ctx.cublas_handle, compute_stream);

        try {
            // 优化: 只对非零块进行计算
            auto work_block =
                ctx.gpu_work.data() + wy_gpu_id * ctx.cols_per_process;
            auto q_block = ctx.gpu_Q.data() + wy_gpu_id * ctx.cols_per_process;
            matrix_ops::gemm(ctx.cublas_handle, ctx.cols_per_process,
                             ctx.q_cols[ctx.mpi_config.rank], sender_m, T(1.0),
                             work_block, ctx.n, true, q_block, ldQ, false,
                             T(0.0), ctx.gpu_W_rec.data(), ctx.n);
        } catch (std::exception& e) {
            throw std::runtime_error(
                fmt::format("CUBLAS 错误: 无法计算第 {} 卡的 YTQ 矩阵 {}",
                            wy_gpu_id, e.what()));
        }

        // Wait for Y broadcast to complete in communication stream
        cudaStreamSynchronize(ctx.stream);

        // Prepare Y data in communication stream
        thrust::fill(thrust::cuda::par.on(ctx.stream), ctx.gpu_work.begin(),
                     ctx.gpu_work.end(), static_cast<T>(0.0));
        matrix_ops::matrix_copy<thrust::device_ptr<T>, thrust::device_ptr<T>,
                                T>(
            ctx.gpu_Y_rec.data(), sender_m,
            ctx.gpu_work.data() + wy_gpu_id * ctx.cols_per_process, ctx.n,
            sender_m, ctx.cols_per_process);

        // Record event when Y data is ready
        cudaEvent_t y_data_ready;
        cudaEventCreate(&y_data_ready);
        cudaEventRecord(y_data_ready, ctx.stream);

        // Wait for Y data and perform WYTQ GEMM in compute stream
        cudaStreamWaitEvent(compute_stream, y_data_ready, 0);

        try {
            // 优化: 只对非零块进行计算
            auto work_block =
                ctx.gpu_work.data() + wy_gpu_id * ctx.cols_per_process;
            auto w_rec_block = ctx.gpu_W_rec.data();
            auto q_block = ctx.gpu_Q.data() + wy_gpu_id * ctx.cols_per_process;
            matrix_ops::gemm(ctx.cublas_handle, static_cast<size_t>(sender_m),
                             ctx.cols_per_process,
                             ctx.q_cols[ctx.mpi_config.rank], T(-1.0),
                             work_block, ctx.n, false, w_rec_block, ctx.n,
                             false, T(1.0), q_block, ldQ);
        } catch (std::exception& e) {
            throw std::runtime_error(
                fmt::format("CUBLAS 错误: 无法计算第 {} 卡的 WYTQ 矩阵 {}",
                            wy_gpu_id, e.what()));
        }

        // Restore original cublas stream
        cublasSetStream(ctx.cublas_handle, original_stream);

        // Clean up events
        cudaEventDestroy(w_data_ready);
        cudaEventDestroy(y_data_ready);
    }

    util::MpiLogger::toc("sb2syGenQT");

    // Clean up compute stream
    cudaStreamDestroy(compute_stream);

    return;
}

}  // namespace mpi

// 显式模板实例化
template class matrix_ops::mpi::MpiSb2syGenQContext<float>;
template class matrix_ops::mpi::MpiSb2syGenQContext<double>;

template void matrix_ops::mpi::sb2syGenQ<float>(
    matrix_ops::mpi::MpiSb2syGenQContext<float>& context);

template void matrix_ops::mpi::sb2syGenQ<double>(
    matrix_ops::mpi::MpiSb2syGenQContext<double>& context);

}  // namespace matrix_ops