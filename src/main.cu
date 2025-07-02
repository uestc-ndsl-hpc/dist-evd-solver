#include <fmt/format.h>

#include <cstddef>

#include "argh.h"
#include "log.h"
#include "matrix_ops.cuh"

template <typename T>
void run_workflow(int n) {
    if constexpr (std::is_same_v<T, float>) {
        util::Logger::println("Using float precision");
    } else {
        util::Logger::println("Using double precision");
    }

    auto C = matrix_ops::create_symmetric_random<T>(n);

    if (util::Logger::is_verbose()) {
        matrix_ops::print(C, n, "Final Symmetric Matrix C");
    }
}

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
    util::Logger::println("--- Running Validation ---");
    matrix_ops::CublasHandle handle;

    // 1. Allocate workspace for QR product
    thrust::device_vector<T> QR_prod(m * n);

    // 2. Compute QR = Q * R
    T alpha = 1.0;
    T beta = 0.0;
    cublasStatus_t status;
    if constexpr (std::is_same_v<T, float>) {
        status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n,
                             (const float*)&alpha,
                             (const float*)thrust::raw_pointer_cast(Q.data()), m,
                             (const float*)thrust::raw_pointer_cast(R.data()), n,
                             (const float*)&beta,
                             (float*)thrust::raw_pointer_cast(QR_prod.data()), m);
    } else if constexpr (std::is_same_v<T, double>) {
        status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n,
                             (const double*)&alpha,
                             (const double*)thrust::raw_pointer_cast(Q.data()), m,
                             (const double*)thrust::raw_pointer_cast(R.data()), n,
                             (const double*)&beta,
                             (double*)thrust::raw_pointer_cast(QR_prod.data()), m);
    }

    if (status != CUBLAS_STATUS_SUCCESS) {
        util::Logger::println("cublasGemm for QR failed.");
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
    util::Logger::println(fmt::format("Frobenius norm of A: {}", norm_A));
    util::Logger::println(fmt::format("Frobenius norm of A - QR: {}", norm_residual));
    util::Logger::println(fmt::format("Relative backward error ||A-QR||/||A||: {}", backward_error));
    util::Logger::println("--- Validation Finished ---");
}

template <typename T>
void run_workflow_tsqr(size_t m, size_t n, bool validate) {
    util::Logger::println(fmt::format("Running TSQR for a {}x{} matrix...", m, n));

    // 1. Create cuBLAS handle using RAII wrapper
    matrix_ops::CublasHandle handle;

    // 2. Create input matrix A 
    auto A = matrix_ops::create_normal_random<T>(m, n);
    
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

    // 4. Call the TSQR function. A is passed as in-out parameter and is overwritten.
    matrix_ops::tsqr(handle, m, n, A, R);
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

int main(int argc, char** argv) {
    argh::parser cmdl(argv);

    const bool verbose = cmdl[{"-v", "--verbose"}];
    util::Logger::init(verbose);
    const bool print_time = cmdl[{"-t", "--time"}];
    util::Logger::init_timer(print_time);
    util::Logger::println("Starting dist-evd-solver");
    
    const bool validate = cmdl[{"--validate"}];

    auto n = (size_t)4;
    cmdl({"-n", "--size"}, 4) >> n;
    auto m = n;
    cmdl({"-m", "--m"}, n) >> m;

    if (cmdl[{"--double"}]) {
        util::Logger::println("Using double precision");
        run_workflow_tsqr<double>(m, n, validate);
    } else if (cmdl[{"--float"}]) {
        util::Logger::println("Using single precision");
        run_workflow_tsqr<float>(m, n, validate);
    } else {
        util::Logger::println("Using default precision");
        run_workflow_tsqr<float>(m, n, validate);
    }

    return 0;
}