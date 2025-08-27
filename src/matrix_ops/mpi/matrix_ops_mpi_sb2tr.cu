#include <mpi.h>
#include <nccl.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "log.h"
#include "matrix_ops.cuh"
#include "matrix_ops_mpi.cuh"

namespace matrix_ops {
namespace mpi {

// MpiTr2sbGenQContext 类方法实现
template <typename T>
MpiSb2trContext<T>::MpiSb2trContext(const MpiConfig& config,
                                    Sy2sbResultBuffers<T>& buffers)
    : mpi_config(config),
      n(buffers.n),
      b(buffers.b),
      gpu_A(std::move(buffers.A)),
      stream(buffers.stream) {
    // 计算分块信息
    cols_cur_node_process = n / mpi_config.size;

    util::MpiLogger::tic("NVShmem_init");
    // Initialize NVShmem
    nvshmemx_init_attr_t attr;
    MPI_Comm comm = MPI_COMM_WORLD;
    attr.mpi_comm = &comm;

    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    util::MpiLogger::toc("NVShmem_init");

    // 设置当前进程使用的 GPU
    // cudaSetDevice(mpi_config.local_gpu_id);

    // 这里可以考虑一下设计, 方便负载均衡
    ldSubA = 2 * b;
    // util::MpiLogger::tic("NVShmem_malloc");

    gpu_subA = (T*)nvshmem_malloc(ldSubA * cols_cur_node_process * sizeof(T));

    ldU = cols_cur_node_process + 2 * b;
    gpu_U.resize(ldU * n);

    util::MpiLogger::tic("BC");

    util::MpiLogger::tic("BC_copyMatrixA2SubA");
    // cudaDeviceSynchronize();
    // Initialize subA matrix as band matrix for matrix A
    copyMatrixA2SubA();
    // cudaDeviceSynchronize();
    util::MpiLogger::toc("BC_copyMatrixA2SubA");

    // 进行NVSHmem资源分配
    prePEWriteCom = (int*)nvshmem_malloc(n * sizeof(int));
    nextPEWriteTailSweepProcRow = (int*)nvshmem_malloc(sizeof(int));
    // util::MpiLogger::toc("NVShmem_malloc");

    util::MpiLogger::tic("BC_cudaMemset");
    cudaMemset(prePEWriteCom, 0, n * sizeof(int));
    cudaMemset(nextPEWriteTailSweepProcRow, 0, sizeof(int));
    util::MpiLogger::toc("BC_cudaMemset");

    com.resize(n);
}
template <typename T>
void MpiSb2trContext<T>::copyMatrixA2SubA() {
    // Todo
    dim3 blockDimcpydA2dSubA(32, 32);
    dim3 gridDimcpydA2dSubA((2 * b + 31) / 32,
                            (cols_cur_node_process + 31) / 32);

    // 将 thrust::device_ptr 转换为原始指针
    auto dA = thrust::raw_pointer_cast(gpu_A.data());
    auto dSubA = gpu_subA;

    matrix_ops::sb2tr::kernel_bugle_chasing_cpydA2dSubA<<<
        gridDimcpydA2dSubA, blockDimcpydA2dSubA>>>(
        n, b, cols_cur_node_process, mpi_config.rank, dA, n, dSubA, ldSubA);

    return;
}

template <typename T>
MpiSb2trContext<T>::~MpiSb2trContext() {
    // 确保所有CUDA操作完成
    cudaDeviceSynchronize();

    // 销毁 CUDA 流
    if (stream != nullptr) {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }

    nvshmem_free(prePEWriteCom);
    nvshmem_free(nextPEWriteTailSweepProcRow);
    nvshmem_free(gpu_subA);

    // 销毁NVShmem环境
    // nvshmem_finalize();
}

template <typename T>
void sb2tr(MpiSb2trContext<T>& ctx) {
    // auto n = ctx.n;
    auto n = static_cast<int>(ctx.n);
    auto b = static_cast<int>(ctx.b);

    auto ns = static_cast<int>(ctx.cols_cur_node_process);

    auto dSubA = ctx.gpu_subA;
    auto ldSubA = static_cast<int>(ctx.ldSubA);
    auto dU = thrust::raw_pointer_cast(ctx.gpu_U.data());
    auto ldU = static_cast<int>(ctx.ldU);

    int PEIndex = ctx.mpi_config.rank;
    int PENum = ctx.mpi_config.size;

    auto com = thrust::raw_pointer_cast(ctx.com.data());

    auto prePEWriteCom = ctx.prePEWriteCom;

    auto nextPEWriteTailSweepProcRow = ctx.nextPEWriteTailSweepProcRow;

    // 调用核函数进行处理
    matrix_ops::sb2tr::sb2tr(n, b, ns, dSubA, ldSubA, dU, ldU, PEIndex, PENum,
                             com, prePEWriteCom, nextPEWriteTailSweepProcRow);

    return;
}

}  // namespace mpi

// 显式模板实例化
template class matrix_ops::mpi::MpiSb2trContext<float>;
template class matrix_ops::mpi::MpiSb2trContext<double>;

template void matrix_ops::mpi::sb2tr<float>(
    matrix_ops::mpi::MpiSb2trContext<float>& context);

template void matrix_ops::mpi::sb2tr<double>(
    matrix_ops::mpi::MpiSb2trContext<double>& context);

}  // namespace matrix_ops