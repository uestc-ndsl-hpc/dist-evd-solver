#include "gpu_handle_wrappers.h"
#include "log.h"
#include "matrix_ops.cuh"
#include "workflow.cuh"

template <typename T>
void run_workflow_sy2sb(size_t n, bool validate) {
    util::Logger::println("--- Running Workflow ---");
    // 1. Generate a random symmetric matrix
    auto A = matrix_ops::create_symmetric_random<T>(n);
    if (util::Logger::is_verbose()) {
        matrix_ops::print(A, n, "original matrix");
    }

    // 2. Run the workflow
    auto handle = common::CublasHandle();
    auto R = thrust::device_vector<T>(n * n);
    auto W = thrust::device_vector<T>(n * n);
    matrix_ops::sy2sb(handle, n, A.data(), R.data(), W.data(), n, n, n);

    // 3. Validate the result
    if (util::Logger::is_verbose()) {
        matrix_ops::print(A, n, "sy2sb result");
    }
}

template void run_workflow_sy2sb<float>(size_t n, bool validate);
template void run_workflow_sy2sb<double>(size_t n, bool validate);