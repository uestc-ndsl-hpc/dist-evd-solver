#include <cuda_runtime.h>
#include <mpi.h>

#include "gpu_handle_wrappers.h"
#include "log.h"
#include "matrix_ops.cuh"

template <typename T>
void run_workflow_sy2sb_mpi(size_t n, bool validate, int num_gpus = 1) {
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
        util::MpiLogger::error("Failed to set CUDA device {}: {}", 
                              local_gpu_id, cudaGetErrorString(cuda_err));
        MPI_Finalize();
        return;
    }

    // Verify GPU assignment
    int current_device;
    cudaGetDevice(&current_device);
    util::MpiLogger::println("Bound to GPU {}", current_device);

    // Synchronize all processes after GPU initialization
    MPI_Barrier(MPI_COMM_WORLD);

    // TODO: Warm-up cuBLAS to avoid initialization overhead in timing
    {
        if (util::MpiLogger::is_verbose() && n <= 128) {
            util::MpiLogger::println("--- Performing cuBLAS warm-up ---");
        }
        // TODO: Implement cuBLAS warm-up
    }

    // Generate initial matrix A (symmetric) on rank 0
    auto A_h = thrust::host_vector<T>(n * n);
    if (rank == 0) {
        auto handle = common::CublasHandle();
        {
            auto A_d = matrix_ops::create_symmetric_random<T>(n, true);
            thrust::copy(A_d.begin(), A_d.end(), A_h.begin());
        }
        
        if (util::MpiLogger::is_verbose() && n <= 256) {
            matrix_ops::print(A_h.data(), n, n, n, "Initial matrix A");
        }
    }

    // TODO: Distribute matrix A across MPI processes

    // TODO: Perform symmetric-to-band (sy2sb) reduction using MPI

    // TODO: Gather results back to root process

    // TODO: Validate results if requested
    if (validate) {
        // TODO: Implement validation logic
    }

    // TODO: Cleanup and finalize MPI
    MPI_Finalize();
}

// Explicit template instantiation
template void run_workflow_sy2sb_mpi<float>(size_t n, bool validate,
                                            int num_gpus);
template void run_workflow_sy2sb_mpi<double>(size_t n, bool validate,
                                             int num_gpus);