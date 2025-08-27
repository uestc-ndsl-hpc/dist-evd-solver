#include <cuda_runtime.h>
#include <mkl_lapacke.h>
#include <mpi.h>
#include <thrust/universal_vector.h>

#include <cstddef>
#include <thread>

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
void bc_back_parallal(
    matrix_ops::mpi::MpiConfig& mpi_config,
    matrix_ops::mpi::Sy2sbResultBuffers<T>& sy2sb_result_buffers,
    matrix_ops::mpi::Tr2sbBuffers<T>& tr2sbBuffer,
    thrust::universal_vector<T>& U_h, bool debug, int rank) {
    util::MpiLogger::tic("BC_Back");
    cudaSetDevice(rank);
    // 进行BC Back
    {
        // 创建 MPI tr2sb GenQ 上下文
        util::MpiLogger::tic("tr2sb_genQ_mpi_context_creation");
        matrix_ops::mpi::MpiTr2sbGenQContext<T> tr2sb_genQ_context(
            mpi_config, sy2sb_result_buffers, tr2sbBuffer, U_h);
        util::MpiLogger::toc("tr2sb_genQ_mpi_context_creation");

        // 执行 MPI tr2sb GenQ 算法
        util::MpiLogger::tic("tr2sb_genQ_mpi_computation");
        matrix_ops::mpi::tr2sbGenQ<T>(tr2sb_genQ_context);
        util::MpiLogger::toc("tr2sb_genQ_mpi_computation");

        tr2sbBuffer.Q = std::move(tr2sb_genQ_context.gpu_Q);
    }

    util::MpiLogger::toc("BC_Back");

    return;
}

template <typename T>
void run_workflow_tr2sb_mpi(size_t n, bool validate, int num_gpus, size_t nb,
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

    MPI_Comm shmcomm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &shmcomm);
    int shmrank, shmsize;
    MPI_Comm_rank(shmcomm, &shmrank);
    MPI_Comm_size(shmcomm, &shmsize);
    MPI_Win zwin;
    T* z_shm = nullptr;
    MPI_Aint z_bytes =
        (shmrank == 0) ? (MPI_Aint)n * (MPI_Aint)n * (MPI_Aint)sizeof(T) : 0;
    MPI_Win_allocate_shared(z_bytes, sizeof(T), MPI_INFO_NULL, shmcomm,
                            (void**)&z_shm, &zwin);
    if (shmrank != 0) {
        MPI_Aint sz;
        int disp;
        void* base = nullptr;
        MPI_Win_shared_query(zwin, 0, &sz, &disp, &base);
        z_shm = static_cast<T*>(base);
    }
    if (z_shm && shmrank == 0) {
        cudaHostRegister(z_shm, (size_t)n * (size_t)n * sizeof(T),
                         cudaHostRegisterPortable);
    }

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
    thrust::host_vector<T> A_h;
    T* A_ptr = nullptr;
    if (rank == 0) {
        auto handle = common::CublasHandle();
        A_h.resize(n * n);
        {
            auto A_d = matrix_ops::create_symmetric_random<T>(n, true);
            thrust::copy(A_d.begin(), A_d.end(), A_h.begin());
        }
        A_ptr = thrust::raw_pointer_cast(A_h.data());
    }

    // debug 打印原始 A
    if (debug) {
        if (rank == 0) {
            matrix_ops::print(A_h.data(), n, n,
                              fmt::format("Original A matrix before sy2sb"));
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // 创建工作数组
    T* W_ptr = nullptr;
    T* Y_ptr = nullptr;

    // 创建 MPI 配置
    matrix_ops::mpi::MpiConfig mpi_config(rank, size, current_device,
                                          total_gpus);

    matrix_ops::mpi::Sy2sbResultBuffers<T> sy2sb_result_buffers;

    util::MpiLogger::tic("SBR+SBR Back");

    // 使用额外的作用域确保上下文在 MPI_Finalize() 之前析构
    {
        // 创建 MPI sy2sb 上下文
        util::MpiLogger::tic("sy2sb_mpi_context_creation");
        matrix_ops::mpi::MpiSy2sbContext<T> sy2sb_context(
            mpi_config, n, A_ptr, n, W_ptr, n, Y_ptr, n, nb, b);
        util::MpiLogger::toc("sy2sb_mpi_context_creation");

        MPI_Barrier(MPI_COMM_WORLD);

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

    matrix_ops::mpi::Tr2sbBuffers<T> tr2sbBuffers;

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
                    util::MpiLogger::println(
                        "Q matrix after sb2sy GenQ  size is {}",
                        sb2sy_genQ_context.gpu_Q.size());
                    matrix_ops::print(
                        sb2sy_genQ_context.gpu_Q, n,
                        sb2sy_genQ_context.q_cols[rank], n + 256,
                        fmt::format("[rank:{}] Q matrix after sb2sy GenQ",
                                    rank));
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
        }

        tr2sbBuffers.Q = std::move(sb2sy_genQ_context.gpu_Q);
        tr2sbBuffers.ldQ = sb2sy_genQ_context.n + U_LEN_PROC_1TIME;
    }

    util::MpiLogger::toc("SBR+SBR Back");

    thrust::universal_vector<T> S_h;
    thrust::universal_vector<T> E_h;
    thrust::universal_vector<T> subU_h;

    if (rank == 0) {
        S_h.resize(n);
        E_h.resize(n);
        subU_h.resize((n + total_gpus * 2 * b) * n);

    } else {
        S_h.resize(n / total_gpus);
        E_h.resize(n / total_gpus);
        subU_h.resize((n / total_gpus + 2 * b) * n);
    }

    util::MpiLogger::tic("BC");

    {
        // 创建 MPI sb2tr 上下文
        util::MpiLogger::tic("sb2tr_mpi_context_creation");
        matrix_ops::mpi::MpiSb2trContext<T> tr2sb_context(mpi_config,
                                                          sy2sb_result_buffers);
        util::MpiLogger::toc("sb2tr_mpi_context_creation");

        // 执行 MPI sb2tr 算法
        util::MpiLogger::tic("sb2tr_mpi_computation");
        matrix_ops::mpi::sb2tr<T>(tr2sb_context);
        util::MpiLogger::toc("sb2tr_mpi_computation");

        util::MpiLogger::tic("Gather_S&E_for_DC");
        auto p_dSubA = thrust::device_pointer_cast(tr2sb_context.gpu_subA);
        auto p_S_h = thrust::raw_pointer_cast(S_h.data());
        matrix_ops::matrix_copy<thrust::device_ptr<T>, T*, T>(
            p_dSubA, tr2sb_context.ldSubA, p_S_h, (size_t)1, (size_t)1,
            tr2sb_context.cols_cur_node_process);

        MPI_Gather(p_S_h, tr2sb_context.cols_cur_node_process,
                   std::is_same_v<T, float> ? MPI_FLOAT : MPI_DOUBLE, p_S_h,
                   tr2sb_context.cols_cur_node_process,
                   std::is_same_v<T, float> ? MPI_FLOAT : MPI_DOUBLE, 0,
                   MPI_COMM_WORLD);

        p_dSubA = thrust::device_pointer_cast(tr2sb_context.gpu_subA + 1);
        auto p_E_h = thrust::raw_pointer_cast(E_h.data());
        matrix_ops::matrix_copy<thrust::device_ptr<T>, T*, T>(
            p_dSubA, tr2sb_context.ldSubA, p_E_h, (size_t)1, (size_t)1,
            tr2sb_context.cols_cur_node_process);

        MPI_Gather(p_E_h, tr2sb_context.cols_cur_node_process,
                   std::is_same_v<T, float> ? MPI_FLOAT : MPI_DOUBLE, p_E_h,
                   tr2sb_context.cols_cur_node_process,
                   std::is_same_v<T, float> ? MPI_FLOAT : MPI_DOUBLE, 0,
                   MPI_COMM_WORLD);
        util::MpiLogger::toc("Gather_S&E_for_DC");

        util::MpiLogger::tic("Gather_U_for_BC_Back");
        auto p_subU_h = thrust::raw_pointer_cast(subU_h.data());
        auto p_gpu_U = thrust::raw_pointer_cast(tr2sb_context.gpu_U.data());
        cudaMemcpy(p_subU_h, p_gpu_U, tr2sb_context.ldU * n,
                   cudaMemcpyDeviceToHost);
        util::MpiLogger::toc("Gather_U_for_BC_Back");
    }
    util::MpiLogger::toc("BC");

    MPI_Barrier(MPI_COMM_WORLD);

    {
        if (rank == 0) {
            // S_h/E_h 仍是你前面 Gather 得到的
            T* S = thrust::raw_pointer_cast(S_h.data());
            T* E = thrust::raw_pointer_cast(E_h.data());
            // 直接写入 z_shm（行主序，ld = n）
            util::MpiLogger::tic("LAPACKE_MKL_DC");
            // if constexpr (std::is_same_v<T, double>) {
            //     LAPACKE_dstedc(LAPACK_ROW_MAJOR, 'I', n, S, E, z_shm, n);
            // } else {
            //     LAPACKE_sstedc(LAPACK_ROW_MAJOR, 'I', n, S, E, z_shm, n);
            // }
            util::MpiLogger::toc("LAPACKE_MKL_DC");
            // 保证对共享段的写入可见
            MPI_Win_sync(zwin);
        }
        // 节点内同步一次，确保大家都能读到完整 Z
        MPI_Barrier(shmcomm);
    }

    if (rank == 0) {
        std::thread back_thread(bc_back_parallal<T>, std::ref(mpi_config),
                                std::ref(sy2sb_result_buffers),
                                std::ref(tr2sbBuffers), std::ref(subU_h), debug,
                                rank);
        back_thread.join();
    } else {
        bc_back_parallal<T>(mpi_config, sy2sb_result_buffers, tr2sbBuffers,
                            subU_h, debug, rank);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    tr2sbBuffers.ldQ = n;

    auto finalQ = thrust::device_vector<T>(n / total_gpus * n);
    util::MpiLogger::tic("FinalGEMM");
    {
        auto gpu_Z = thrust::device_vector<T>(n / total_gpus * n);
        auto& gpu_QT = tr2sbBuffers.Q;
        common::CublasHandle handle;

        auto gpu_Q_ptr = gpu_QT.data();

        for (size_t i = 0; i < total_gpus; ++i) {
            auto cpy_name = fmt::format("FinalGEMM copy {}", i);
            auto gemm_name = fmt::format("FinalGEMM gemm {}", i);
            auto z_shm_ptr = z_shm + i * n * n / total_gpus;

            util::MpiLogger::tic(cpy_name);
            thrust::copy(z_shm_ptr, z_shm_ptr + n * n / total_gpus,
                         gpu_Z.data());
            util::MpiLogger::toc(cpy_name);

            util::MpiLogger::tic(gemm_name);
            matrix_ops::gemm(
                handle, n / total_gpus, n / total_gpus, n, T(1.0), gpu_Z.data(),
                n / total_gpus, gpu_Q_ptr, n, T(0.0),
                finalQ.data() + i * n / total_gpus, n);
            util::MpiLogger::toc(gemm_name);
        }
    }
    util::MpiLogger::toc("FinalGEMM");

    tr2sbBuffers.Q.resize(0);
    // 在上下文析构之后再调用 MPI_Finalize
    MPI_Finalize();
}

// Explicit template instantiation
template void run_workflow_tr2sb_mpi<float>(size_t n, bool validate,
                                            int num_gpus, size_t nb, size_t b,
                                            bool debug);
template void run_workflow_tr2sb_mpi<double>(size_t n, bool validate,
                                             int num_gpus, size_t nb, size_t b,
                                             bool debug);