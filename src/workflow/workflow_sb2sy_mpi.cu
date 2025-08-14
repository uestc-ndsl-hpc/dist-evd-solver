#include <cuda_runtime.h>
#include <mpi.h>

#include "gpu_handle_wrappers.h"
#include "log.h"
#include "matrix_ops.cuh"
#include "matrix_ops_mpi.cuh"

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

template <typename T>
void run_workflow_sb2sy_mpi(size_t n, bool validate, int num_gpus, size_t nb,
                            size_t b, bool debug) {
    // Initialize MPI environment
    int provided;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        // Use regular logger before getting rank
        util::Logger::error("MPI does not support MPI_THREAD_MULTIPLE");
        MPI_Finalize();
        return;
    }

    // Get MPI rank and size
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Initialize MPI logger with rank information
    util::MpiLogger::init(util::Logger::is_verbose(), rank);

    if (rank == 0) {
        util::MpiLogger::print_environment_info();
        util::MpiLogger::println("Starting dist-evd-solver MPI version");
        util::MpiLogger::println("MPI initialized with {} processes", size);
    }

    // Initialize GPU devices for each MPI process
    int total_gpus;
    cudaError_t cuda_err = cudaGetDeviceCount(&total_gpus);
    if (cuda_err != cudaSuccess) {
        util::MpiLogger::error("Failed to get CUDA device count: {}",
                               cudaGetErrorString(cuda_err));
        MPI_Finalize();
        return;
    }

    // Bind each MPI process to specific GPU(s)
    // Strategy: round-robin assignment of GPUs to MPI processes
    int local_gpu_id = rank % total_gpus;
    cuda_err = cudaSetDevice(local_gpu_id);
    if (cuda_err != cudaSuccess) {
        util::MpiLogger::error("Failed to set CUDA device {}: {}", local_gpu_id,
                               cudaGetErrorString(cuda_err));
        MPI_Finalize();
        return;
    }

    // Verify GPU assignment
    int current_device;
    cudaGetDevice(&current_device);
    util::MpiLogger::println("Bound to GPU {}", current_device);

    // Synchronize all processes after GPU initialization
    MPI_Barrier(MPI_COMM_WORLD);

    // Warm-up cuBLAS to avoid initialization overhead in timing
    {
        if (util::MpiLogger::is_verbose()) {
            util::MpiLogger::println("--- Performing cuBLAS warm-up ---");
        }

        auto handle = common::CublasHandle();
        const int warmup_size = 8192;
        const int num_warmup_iterations = 3;

        // Allocate device memory for warm-up
        thrust::device_vector<T> a_d(warmup_size * warmup_size);
        thrust::device_vector<T> b_d(warmup_size * warmup_size);
        thrust::device_vector<T> c_d(warmup_size * warmup_size);

        // Initialize with simple values
        thrust::fill(a_d.begin(), a_d.end(), T(1.0));
        thrust::fill(b_d.begin(), b_d.end(), T(1.0));
        thrust::fill(c_d.begin(), c_d.end(), T(0.0));

        // Perform multiple GEMM operations for thorough warm-up
        for (int i = 0; i < num_warmup_iterations; ++i) {
            if (util::MpiLogger::is_verbose()) {
                util::MpiLogger::println("  Warm-up iteration {}/{}", i + 1,
                                         num_warmup_iterations);
            }

            try {
                matrix_ops::gemm(handle, warmup_size, warmup_size, warmup_size,
                                 T(1.0), a_d.data(), warmup_size, b_d.data(),
                                 warmup_size, T(0.0), c_d.data(), warmup_size);

                // Synchronize to ensure operation completes
                cudaError_t cuda_err = cudaDeviceSynchronize();
                if (cuda_err != cudaSuccess) {
                    util::MpiLogger::error(
                        "CUDA synchronization failed during warm-up: {}",
                        cudaGetErrorString(cuda_err));
                    MPI_Finalize();
                    return;
                }
            } catch (const std::exception& e) {
                util::MpiLogger::error(
                    "cuBLAS warm-up GEMM operation failed: {}", e.what());
                MPI_Finalize();
                return;
            }
        }

        if (util::MpiLogger::is_verbose()) {
            util::MpiLogger::println(
                "--- cuBLAS warm-up completed successfully ---");
        }

        // Device vectors will be automatically freed when they go out of scope
    }

    // Generate initial matrix A (symmetric) on rank 0
    auto A_h = thrust::host_vector<T>(n * n);
    if (rank == 0) {
        auto handle = common::CublasHandle();
        {
            auto A_d = matrix_ops::create_symmetric_random<T>(n, true);
            thrust::copy(A_d.begin(), A_d.end(), A_h.begin());
        }
    }

    // 广播矩阵数据到所有进程
    MPI_Bcast(A_h.data(), n * n,
              std::is_same_v<T, float> ? MPI_FLOAT : MPI_DOUBLE, 0,
              MPI_COMM_WORLD);

    // debug 打印原始 A
    if (debug) {
        if (rank == 0) {
            matrix_ops::print(A_h.data(), n, n,
                              fmt::format("Original A matrix before sy2sb"));
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // 创建工作数组
    auto W_h = thrust::host_vector<T>(n * n, 0.0);
    auto Y_h = thrust::host_vector<T>(n * n, 0.0);

    // 创建 MPI 配置
    matrix_ops::mpi::MpiConfig mpi_config(rank, size, current_device,
                                          total_gpus);

    matrix_ops::mpi::Sy2sbResultBuffers<T> sy2sb_result_buffers;

    // 使用额外的作用域确保上下文在 MPI_Finalize() 之前析构
    {
        // 创建 MPI sy2sb 上下文
        util::MpiLogger::tic("sy2sb_mpi_context_creation");
        matrix_ops::mpi::MpiSy2sbContext<T> sy2sb_context(
            mpi_config, n, A_h.data(), n, W_h.data(), n, Y_h.data(), n, nb, b);
        util::MpiLogger::toc("sy2sb_mpi_context_creation");

        // 执行 MPI sy2sb 算法
        util::MpiLogger::tic("sy2sb_mpi_computation");
        matrix_ops::mpi::sy2sb<T>(sy2sb_context);
        util::MpiLogger::toc("sy2sb_mpi_computation");

        // 保留 A, W, Y 缓冲区，释放其他所有资源
        sy2sb_result_buffers = std::move(sy2sb_context.release_sy2sb_buffers());
    }

    if (debug) {
        // 计算每个进程的数据大小
        size_t cols_per_process = n / size;
        size_t local_data_size = cols_per_process * n;

        auto B_h_print = thrust::host_vector<T>(n * n);
        auto B_h_partial = thrust::host_vector<T>(sy2sb_result_buffers.A);

        // 将大家的部分 A 数据收集到 rank 0
        MPI_Gather(B_h_partial.data(), local_data_size,
                   std::is_same_v<T, float> ? MPI_FLOAT : MPI_DOUBLE,
                   B_h_print.data(), local_data_size,
                   std::is_same_v<T, float> ? MPI_FLOAT : MPI_DOUBLE, 0,
                   MPI_COMM_WORLD);

        auto B_d = thrust::device_vector<T>(B_h_print);
        thrust::for_each(thrust::make_counting_iterator<size_t>(0),
                         thrust::make_counting_iterator<size_t>(n * n),
                         make_symmetric_functor<T>(B_d.data(), n, n));

        // 只在 rank 0 打印收集到的矩阵
        if (rank == 0) {
            matrix_ops::print(B_d.data(), n, n,
                              fmt::format("Collected A matrix after sy2sb"));
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    {
        // 创建 MPI sb2sy GenQ 上下文
        util::MpiLogger::tic("sb2sy_genQ_mpi_context_creation");
        matrix_ops::mpi::MpiSb2syGenQContext<T> sb2sy_genQ_context(
            mpi_config, sy2sb_result_buffers);
        util::MpiLogger::toc("sb2sy_genQ_mpi_context_creation");

        // 执行 MPI sb2sy GenQ 算法
        util::MpiLogger::tic("sb2sy_genQ_mpi_computation");
        matrix_ops::mpi::sb2syGenQ<T>(sb2sy_genQ_context);
        util::MpiLogger::toc("sb2sy_genQ_mpi_computation");

        if (debug) {
            for (auto i = 0; i < mpi_config.size; ++i) {
                if (i == rank) {
                    matrix_ops::print(
                        sb2sy_genQ_context.gpu_Q, n,
                        sb2sy_genQ_context.q_cols[rank],
                        fmt::format("[rank:{}] Q matrix before sb2sy GenQ",
                                    rank));
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
        }

        // 调试打印 sb2sy 结果
        if (debug) {
            for (auto i = 0; i < mpi_config.size; ++i) {
                if (i == rank) {
                    matrix_ops::print(
                        sb2sy_genQ_context.gpu_W, n, n / mpi_config.size,
                        fmt::format("[rank:{}] W matrix after sb2sy", rank));
                    matrix_ops::print(
                        sb2sy_genQ_context.gpu_Y, n, n / mpi_config.size,
                        fmt::format("[rank:{}] Y matrix after sb2sy", rank));
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
        }
    }

    // 在上下文析构之后再调用 MPI_Finalize
    MPI_Finalize();
}

// Explicit template instantiation
template void run_workflow_sb2sy_mpi<float>(size_t n, bool validate,
                                            int num_gpus, size_t nb, size_t b,
                                            bool debug);
template void run_workflow_sb2sy_mpi<double>(size_t n, bool validate,
                                             int num_gpus, size_t nb, size_t b,
                                             bool debug);