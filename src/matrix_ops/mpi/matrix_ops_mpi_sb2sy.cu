#include <mpi.h>
#include <nccl.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cstddef>

#include "log.h"
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
      gpu_W(std::move(buffers.W)),
      gpu_Y(std::move(buffers.Y)),
      nccl_comm(buffers.nccl_comm),
      sub_comm_groups(std::move(buffers.sub_comm_groups)),
      sub_mpi_comms(std::move(buffers.sub_mpi_comms)) {
    
    // 计算分块信息
    cols_per_process = n / mpi_config.size;
    start_col = mpi_config.rank * cols_per_process;
    local_matrix_size = cols_per_process * n;
    
    // 设置 NCCL 数据类型
    if constexpr (std::is_same_v<T, float>) {
        nccl_type = ncclFloat32;
    } else if constexpr (std::is_same_v<T, double>) {
        nccl_type = ncclFloat64;
    }
    
    // 设置当前进程使用的 GPU
    cudaSetDevice(mpi_config.local_gpu_id);
    
    // 分配工作空间
    gpu_work.resize(2 * n * nb);
    thrust::fill(gpu_work.begin(), gpu_work.end(), T(0));
    
    // 清空原始缓冲区的通信器引用，避免重复释放
    buffers.nccl_comm = nullptr;
    buffers.sub_comm_groups.clear();
    buffers.sub_mpi_comms.clear();
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
}

template <typename T>
void sy2sbGenQ(MpiSb2syGenQContext<T>& context) {
    util::MpiLogger::println("(还)原(SBR)神启动");
    return;
}

}  // namespace mpi

// 显式模板实例化
template class matrix_ops::mpi::MpiSb2syGenQContext<float>;
template class matrix_ops::mpi::MpiSb2syGenQContext<double>;

template void matrix_ops::mpi::sy2sbGenQ<float>(
    matrix_ops::mpi::MpiSb2syGenQContext<float>& context);

template void matrix_ops::mpi::sy2sbGenQ<double>(
    matrix_ops::mpi::MpiSb2syGenQContext<double>& context);

}  // namespace matrix_ops