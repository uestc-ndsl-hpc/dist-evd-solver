#include "log.h"
#include "matrix_ops_dist.cuh"
#include <thrust/host_vector.h>

namespace matrix_ops {
namespace dist {
template <typename T>
void sy2sb(const common::CublasHandle& handle, size_t n, T* A, size_t lda,
           T* Y, size_t ldy, T* W, size_t ldw,
           size_t nb, size_t b) {
    util::Logger::println("sy2sb dist");

    auto cusolverHandle = common::CusolverDnHandle();

    auto oriA = thrust::host_vector<T>(n * n);
    thrust::copy(A, A + n * n, oriA.begin());

    return;
}
}  // namespace dist
}  // namespace matrix_ops

template void matrix_ops::dist::sy2sb<float>(
    const common::CublasHandle& handle, size_t n, float* A, size_t lda,
    float* Y, size_t ldy, float* W, size_t ldw, size_t nb, size_t b);

template void matrix_ops::dist::sy2sb<double>(
    const common::CublasHandle& handle, size_t n, double* A, size_t lda,
    double* Y, size_t ldy, double* W, size_t ldw, size_t nb, size_t b);