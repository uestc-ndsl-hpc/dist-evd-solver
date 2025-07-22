#include "gpu_handle_wrappers.h"
#include "log.h"
#include "matrix_ops.cuh"
#include "matrix_ops_dist.cuh"
#include "workflow.cuh"

template <typename T>
void run_workflow_sy2sb_dist(size_t n, bool validate) {
    // TODO: replace Warm-up cuBLAS to avoid initialization overhead in timing
    // with cublasXt
    {
        if (util::Logger::is_verbose() && n <= 128) {
            util::Logger::println("--- Performing cuBLAS warm-up ---");
        }
        const int n_warmup = 16384;
        thrust::device_vector<T> d_A(n_warmup * n_warmup, 1.0);
        thrust::device_vector<T> d_B(n_warmup * n_warmup, 1.0);
        thrust::device_vector<T> d_C(n_warmup * n_warmup, 0.0);
        T alpha = 1.0;
        T beta = 0.0;
        common::CublasHandle handle;
        for (int i = 0; i < 10; i++) {
            if constexpr (std::is_same_v<T, float>) {
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_warmup,
                            n_warmup, n_warmup, (const float*)&alpha,
                            (const float*)thrust::raw_pointer_cast(d_A.data()),
                            n_warmup,
                            (const float*)thrust::raw_pointer_cast(d_B.data()),
                            n_warmup, (const float*)&beta,
                            (float*)thrust::raw_pointer_cast(d_C.data()),
                            n_warmup);
            } else {
                cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_warmup,
                            n_warmup, n_warmup, (const double*)&alpha,
                            (const double*)thrust::raw_pointer_cast(d_A.data()),
                            n_warmup,
                            (const double*)thrust::raw_pointer_cast(d_B.data()),
                            n_warmup, (const double*)&beta,
                            (double*)thrust::raw_pointer_cast(d_C.data()),
                            n_warmup);
            }
            cudaDeviceSynchronize();
        }
        util::Logger::println("--- Warm-up finished ---");
    }
    util::Logger::println("--- Running Sy2Sb Workflow ---");
    // 1. Generate a random symmetric matrix

    auto A_h = thrust::host_vector<T>(n * n);

    // 2. Run the workflow
    auto handle = common::CublasHandle();
    {
        auto A_d = matrix_ops::create_symmetric_random<T>(n, true);
        thrust::copy(A_d.begin(), A_d.end(), A_h.begin());
    }

    auto Y_h = thrust::host_vector<T>(n * n);
    auto W_h = thrust::host_vector<T>(n * n);

    matrix_ops::dist::sy2sb(handle, n, A_h.data(), n, W_h.data(), n, Y_h.data(),
                            n, 32, 16, 2);

    if (util::Logger::is_verbose() && n <= 128) {
        matrix_ops::print(A_h.data(), n, n, n, "A");
        matrix_ops::print(W_h.data(), n, n, n, "W");
        matrix_ops::print(Y_h.data(), n, n, n, "Y");
    }
}

template void run_workflow_sy2sb_dist<float>(size_t n, bool validate);
template void run_workflow_sy2sb_dist<double>(size_t n, bool validate);