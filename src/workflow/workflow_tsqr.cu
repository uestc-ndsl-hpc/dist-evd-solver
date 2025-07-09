#include <cublas_v2.h>
#include <fmt/format.h>

#include <cmath>
#include <cstddef>

#include "log.h"
#include "matrix_ops.cuh"
#include "workflow.cuh"

template <typename T>
struct subtract_op {
    __host__ __device__ T operator()(const T& a, const T& b) const {
        return a - b;
    }
};

template <typename T>
void validate_tsqr(const thrust::device_vector<T>& A_orig,
                   const thrust::device_vector<T>& Q,
                   const thrust::device_vector<T>& R, size_t m, size_t n) {
    fmt::println("--- Running Validation ---");
    matrix_ops::CublasHandle handle;

    // 1. Allocate workspace for QR product
    thrust::device_vector<T> QR_prod(m * n);

    // 2. Compute QR = Q * R
    T alpha = 1.0;
    T beta = 0.0;
    cublasStatus_t status;
    if constexpr (std::is_same_v<T, float>) {
        status = cublasSgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, (const float*)&alpha,
            (const float*)thrust::raw_pointer_cast(Q.data()), m,
            (const float*)thrust::raw_pointer_cast(R.data()), n,
            (const float*)&beta,
            (float*)thrust::raw_pointer_cast(QR_prod.data()), m);
    } else if constexpr (std::is_same_v<T, double>) {
        status = cublasDgemm(
            handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, (const double*)&alpha,
            (const double*)thrust::raw_pointer_cast(Q.data()), m,
            (const double*)thrust::raw_pointer_cast(R.data()), n,
            (const double*)&beta,
            (double*)thrust::raw_pointer_cast(QR_prod.data()), m);
    }

    if (status != CUBLAS_STATUS_SUCCESS) {
        fmt::println("cublasGemm for QR failed.");
        return;
    }

    // 3. Compute Residual = A_orig - QR
    thrust::device_vector<T> residual = A_orig;  // Copy A_orig to residual
    thrust::transform(residual.begin(), residual.end(), QR_prod.begin(),
                      residual.begin(), subtract_op<T>());

    // 4. Compute norms using device pointers for robustness
    T norm_A = 0.0;
    T norm_residual = 0.0;

    // Set pointer mode to device, so cublasNrm2 writes result to a device ptr
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

    thrust::device_vector<T> d_norm_A(1);
    thrust::device_vector<T> d_norm_residual(1);

    if constexpr (std::is_same_v<T, float>) {
        cublasSnrm2(handle, A_orig.size(),
                    (const float*)thrust::raw_pointer_cast(A_orig.data()), 1,
                    (float*)thrust::raw_pointer_cast(d_norm_A.data()));
        cublasSnrm2(handle, residual.size(),
                    (const float*)thrust::raw_pointer_cast(residual.data()), 1,
                    (float*)thrust::raw_pointer_cast(d_norm_residual.data()));
    } else if constexpr (std::is_same_v<T, double>) {
        cublasDnrm2(handle, A_orig.size(),
                    (const double*)thrust::raw_pointer_cast(A_orig.data()), 1,
                    (double*)thrust::raw_pointer_cast(d_norm_A.data()));
        cublasDnrm2(handle, residual.size(),
                    (const double*)thrust::raw_pointer_cast(residual.data()), 1,
                    (double*)thrust::raw_pointer_cast(d_norm_residual.data()));
    }

    // Restore pointer mode to default
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

    // Copy results from device to host
    norm_A = d_norm_A[0];
    norm_residual = d_norm_residual[0];

    // 5. Compute and print backward error
    double backward_error = (norm_A > 0) ? (norm_residual / norm_A) : 0.0;
    fmt::println(fmt::format("Frobenius norm of A: {}", norm_A));
    fmt::println(fmt::format("Frobenius norm of A - QR: {}", norm_residual));
    fmt::println(fmt::format("Relative backward error ||A-QR||/||A||: {}",
                             backward_error));
    fmt::println("--- Validation Finished ---");
}

template <typename T>
void run_workflow_tsqr(size_t m, size_t n, bool validate) {
    util::Logger::println(
        fmt::format("Running TSQR for a {}x{} matrix...", m, n));

    // 1. Create cuBLAS handle using RAII wrapper
    matrix_ops::CublasHandle handle;

    // Warm-up cuBLAS to avoid initialization overhead in timing.
    {
        if (util::Logger::is_verbose()) {
            util::Logger::println("--- Performing cuBLAS warm-up ---");
        }
        const int n_warmup = 16384;
        thrust::device_vector<T> d_A(n_warmup * n_warmup, 1.0);
        thrust::device_vector<T> d_B(n_warmup * n_warmup, 1.0);
        thrust::device_vector<T> d_C(n_warmup * n_warmup, 0.0);
        T alpha = 1.0;
        T beta = 0.0;
        matrix_ops::CublasHandle handle;
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

    // 2. Create input matrix A
    auto A = matrix_ops::create_uniform_random<T>(m, n);

    // Keep a copy of A for validation if needed
    thrust::device_vector<T> A_copy;

    if (validate) {
        A_copy = A;
    }

    if (util::Logger::is_verbose()) {
        matrix_ops::print(A, m, n, "Input Matrix A");
    }

    // 3. Create output vector for R
    thrust::device_vector<T> R(n * n);

    // 4. Call the TSQR function. A is passed as in-out parameter and is
    // overwritten.
    util::Logger::tic("tsqr");
    matrix_ops::tsqr(handle, m, n, A.data(), R.data());
    cudaDeviceSynchronize();  // Wait for TSQR to finish before stopping timer
    util::Logger::toc("tsqr", 2 * m * n * n - 2.f / 3.f * std::pow(n, 3));

    util::Logger::println("TSQR decomposition completed successfully.");

    // 5. Print the results if verbose
    if (util::Logger::is_verbose()) {
        matrix_ops::print(A, m, n, "Output Orthogonal Matrix Q");
        matrix_ops::print(R, n, n, "Output Upper-Triangular Matrix R");
    }

    // 6. Validate the results if requested
    if (validate) {
        // A now holds Q
        validate_tsqr(A_copy, A, R, m, n);
    }
}

template void run_workflow_tsqr<float>(size_t m, size_t n, bool validate);
template void run_workflow_tsqr<double>(size_t m, size_t n, bool validate);