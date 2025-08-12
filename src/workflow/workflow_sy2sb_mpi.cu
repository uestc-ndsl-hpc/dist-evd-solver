#include <cuda_runtime.h>
#include <mpi.h>

#include "gpu_handle_wrappers.h"
#include "log.h"
#include "matrix_ops.cuh"
#include "matrix_ops_mpi.cuh"

template <typename T>
void run_workflow_sy2sb_mpi(size_t n, bool validate, int num_gpus, size_t nb,
                            size_t b) {
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
                util::MpiLogger::println("  Warm-up iteration {}/{}", i + 1, num_warmup_iterations);
            }
            
            try {
                matrix_ops::gemm(handle, warmup_size, warmup_size, warmup_size,
                                T(1.0), a_d.data(), warmup_size,
                                b_d.data(), warmup_size, T(0.0),
                                c_d.data(), warmup_size);
                
                // Synchronize to ensure operation completes
                cudaError_t cuda_err = cudaDeviceSynchronize();
                if (cuda_err != cudaSuccess) {
                    util::MpiLogger::error("CUDA synchronization failed during warm-up: {}",
                                           cudaGetErrorString(cuda_err));
                    MPI_Finalize();
                    return;
                }
            } catch (const std::exception& e) {
                util::MpiLogger::error("cuBLAS warm-up GEMM operation failed: {}", e.what());
                MPI_Finalize();
                return;
            }
        }
        
        if (util::MpiLogger::is_verbose()) {
            util::MpiLogger::println("--- cuBLAS warm-up completed successfully ---");
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

    // TODO: Perform symmetric-to-band (sy2sb) reduction using MPI

    // 广播矩阵数据到所有进程
    MPI_Bcast(A_h.data(), n * n,
              std::is_same_v<T, float> ? MPI_FLOAT : MPI_DOUBLE, 0,
              MPI_COMM_WORLD);

    // 创建工作数组
    auto W_h = thrust::host_vector<T>(n * n, 0.0);
    auto Y_h = thrust::host_vector<T>(n * n, 0.0);

    // 创建 MPI 配置
    matrix_ops::mpi::MpiConfig mpi_config(rank, size, current_device,
                                          total_gpus);

    // 执行 MPI sy2sb 算法
    util::MpiLogger::tic("sy2sb_mpi_computation");

    matrix_ops::mpi::sy2sb<T>(mpi_config, n, A_h.data(), n, W_h.data(), n,
                              Y_h.data(), n, nb, b);

    util::MpiLogger::toc("sy2sb_mpi_computation");

    // TODO: Validate results if requested
    if (validate) {
        // TODO: Implement validation logic
        
    }

    // TODO: Cleanup and finalize MPI
    MPI_Finalize();
}

// Explicit template instantiation
template void run_workflow_sy2sb_mpi<float>(size_t n, bool validate,
                                            int num_gpus, size_t nb, size_t b);
template void run_workflow_sy2sb_mpi<double>(size_t n, bool validate,
                                             int num_gpus, size_t nb, size_t b);