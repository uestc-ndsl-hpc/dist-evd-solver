#include "gpu_handle_wrappers.h"
#include "log.h"
#include "matrix_ops.cuh"
#include "workflow.cuh"

template <typename T>
void run_workflow_sy2sb(size_t n, bool validate) {
    // Warm-up cuBLAS to avoid initialization overhead in timing.
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
    auto A = matrix_ops::create_symmetric_random<T>(n, true);

    // 2. Run the workflow
    auto handle = common::CublasHandle();
    auto Y = thrust::device_vector<T>(n * n);
    auto W = thrust::device_vector<T>(n * n);

    matrix_ops::sy2sb(handle, n, A.data(), n, Y.data(), n, W.data(), n, 32, 16);

    // 3. Validate the result
    if (util::Logger::is_verbose() && n <= 256) {
        matrix_ops::print(A, n, "sy2sb result");
    }
}

template void run_workflow_sy2sb<float>(size_t n, bool validate);
template void run_workflow_sy2sb<double>(size_t n, bool validate);