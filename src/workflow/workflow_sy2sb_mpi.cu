#include <mpi.h>
#include <cuda_runtime.h>
#include "log.h"

template <typename T>
void run_workflow_sy2sb_mpi(size_t n, bool validate, int num_gpus = 1) {
    // Initialize MPI environment
    int provided;
    MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    if (provided < MPI_THREAD_MULTIPLE) {
        util::Logger::error("MPI does not support MPI_THREAD_MULTIPLE");
        MPI_Finalize();
        return;
    }
    
    // Get MPI rank and size
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        util::Logger::println("MPI initialized with {} processes", size);
    }
    
    // Initialize GPU devices for each MPI process
    int total_gpus;
    cudaError_t cuda_err = cudaGetDeviceCount(&total_gpus);
    if (cuda_err != cudaSuccess) {
        util::Logger::error("Failed to get CUDA device count: {}", 
                           cudaGetErrorString(cuda_err));
        MPI_Finalize();
        return;
    }
    
    // Bind each MPI process to specific GPU(s)
    // Strategy: round-robin assignment of GPUs to MPI processes
    int local_gpu_id = rank % total_gpus;
    cuda_err = cudaSetDevice(local_gpu_id);
    if (cuda_err != cudaSuccess) {
        util::Logger::error("Rank {}: Failed to set CUDA device {}: {}", 
                           rank, local_gpu_id, cudaGetErrorString(cuda_err));
        MPI_Finalize();
        return;
    }
    
    // Verify GPU assignment
    int current_device;
    cudaGetDevice(&current_device);
    util::Logger::println("Rank {} bound to GPU {}", rank, current_device);
    
    // Synchronize all processes after GPU initialization
    MPI_Barrier(MPI_COMM_WORLD);
    
    // TODO: Warm-up cuBLAS to avoid initialization overhead in timing
    {
        if (util::Logger::is_verbose() && n <= 128) {
            util::Logger::println("--- Performing cuBLAS warm-up ---");
        }
        // TODO: Implement cuBLAS warm-up
    }
    
    // TODO: Allocate matrices on GPU(s)
    
    // TODO: Generate initial matrix A (symmetric)
    
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
template void run_workflow_sy2sb_mpi<float>(size_t n, bool validate, int num_gpus);
template void run_workflow_sy2sb_mpi<double>(size_t n, bool validate, int num_gpus);