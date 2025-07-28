#include <nccl.h>
#include <omp.h>
#include <thrust/device_vector.h>

#include <vector>

#include "fmt/base.h"
#include "fmt/format.h"
#include "gpu_handle_wrappers.h"
#include "matrix_ops.cuh"

// 添加标准的错误检查宏，这对于调试至关重要
#define CUDACHECK(cmd)                                                    \
    do {                                                                  \
        cudaError_t e = cmd;                                              \
        if (e != cudaSuccess) {                                           \
            fmt::println(stderr, "Fatal: CUDA error in line {} - {}: {}", \
                         __LINE__, #cmd, cudaGetErrorString(e));          \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)
#define NCCLCHECK(cmd)                                                    \
    do {                                                                  \
        ncclResult_t r = cmd;                                             \
        if (r != ncclSuccess) {                                           \
            fmt::println(stderr, "Fatal: NCCL error in line {} - {}: {}", \
                         __LINE__, #cmd, ncclGetErrorString(r));          \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

int main() {
    auto m = 16;
    auto n = 16;
    auto gpu_num = 2;
    auto n4gpu = n / gpu_num;
    auto root_gpu = 0;
    auto dev_list = std::vector<int>{0, 1};

    // --- 1. 初始化阶段 (在并行区域之外，串行执行) ---

    CUDACHECK(cudaSetDevice(root_gpu));
    thrust::device_vector<float> b_bcast(m * n, 1.0f);

    std::vector<thrust::device_vector<float>> gpu_data_dist(gpu_num);
    std::vector<thrust::device_vector<float>> ans_data_dist(gpu_num);
    std::vector<thrust::device_vector<float>> buffer(gpu_num);

    std::vector<cudaStream_t> streams(gpu_num);
    std::vector<ncclComm_t> comms(gpu_num);
    std::vector<common::CublasHandle> handles(gpu_num);

    // 将所有初始化工作集中在一个循环中，更清晰
    for (int i = 0; i < gpu_num; ++i) {
        auto gpu_id = dev_list[i];
        CUDACHECK(cudaSetDevice(gpu_id));

        // 创建资源
        CUDACHECK(cudaStreamCreate(&streams[i]));
        handles[i] = common::CublasHandle();  // 假设构造函数创建 handle
        cublasSetStream(handles[i], streams[i]);

        // 分配并初始化设备内存
        gpu_data_dist[i].resize(m * n4gpu);
        thrust::fill(gpu_data_dist[i].begin(), gpu_data_dist[i].end(), 1.0f);

        buffer[i].resize(m * n);

        // 结果维度是 (n4gpu, n) = (8, 16)
        ans_data_dist[i].resize(n4gpu * n);
        thrust::fill(ans_data_dist[i].begin(), ans_data_dist[i].end(), 0.0f);
    }

    NCCLCHECK(ncclCommInitAll(comms.data(), gpu_num, dev_list.data()));

    // --- 2. 并行执行阶段 ---
#pragma omp parallel num_threads(gpu_num)
    {
        auto i = omp_get_thread_num();
        auto gpu_id = dev_list[i];
        CUDACHECK(cudaSetDevice(gpu_id));

        // a. 根GPU准备广播数据
        if (gpu_id == root_gpu) {
            CUDACHECK(cudaMemcpyAsync(
                buffer[i].data().get(), b_bcast.data().get(),
                m * n * sizeof(float), cudaMemcpyDeviceToDevice, streams[i]));
        }

        // b. 执行集体广播
        NCCLCHECK(ncclBcast(buffer[i].data().get(), m * n, ncclFloat, root_gpu,
                            comms[i], streams[i]));

        // c. 执行 GEMM 计算
        // op(A) = gpu_data_dist^T, 维度 (n4gpu, m) = (8, 16)
        // op(B) = buffer, 维度 (m, n) = (16, 16)
        // C = ans_data_dist, 维度 (n4gpu, n) = (8, 16)
        // cublas的m,n,k分别对应结果C的行、列和内积维度
        matrix_ops::gemm(handles[i], n4gpu, n, m, 1.f, gpu_data_dist[i].data(),
                         m, true, buffer[i].data(), m, false, 0.f,
                         ans_data_dist[i].data(), n4gpu);

        // d. 同步流，等待所有在该流上的异步操作完成
        CUDACHECK(cudaStreamSynchronize(streams[i]));

        // e. 现在可以安全地打印结果了
#pragma omp critical
        {
            fmt::println("GPU {} finished all async tasks.", gpu_id);
            matrix_ops::print(ans_data_dist[i], n4gpu, n,
                              fmt::format("GPU {} ans matrix", gpu_id));
        }
    }

    // --- 3. 清理阶段 ---
    fmt::println("Cleaning up resources.");
    for (int i = 0; i < gpu_num; ++i) {
        auto gpu_id = dev_list[i];
        CUDACHECK(cudaSetDevice(gpu_id));
        CUDACHECK(cudaStreamDestroy(streams[i]));
    }
    // NCCL comms 应该在所有 stream 操作完成后销毁
    for (int i = 0; i < gpu_num; ++i) {
        NCCLCHECK(ncclCommDestroy(comms[i]));
    }

    return 0;
}