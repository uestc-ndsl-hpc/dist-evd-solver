#include <cstddef>

#include <thrust/device_vector.h>
#include "gpu_handle_wrappers.h"
#include "internal/sy2sb/sy2sb_panelqr.cuh"
#include "matrix_ops.cuh"

namespace matrix_ops {

/**
 * @brief the function to execute symmetric matrix to symmetric band matrix
 *
 * @tparam T
 * @param handle cublas handler
 * @param n size of the matrix A
 * @param A_inout the matrix A
 */
template <typename T>
void sy2sb(const common::CublasHandle& handle, size_t n,
           thrust::device_ptr<T> A_inout) {
    // the size of the matrix A
    const auto m = (size_t)n;
    // the panel size
    const auto b = (size_t)32;
    // the block size
    const auto nb = (size_t)b * 4;

    auto lda = m;
    auto ldr = n;
    auto ldw = m;

    auto R = thrust::device_vector<T>(n * ldr);
    auto W = thrust::device_vector<T>(m * n);

    common::CusolverDnHandle cusolverHandle;
    // panel QR
    internal::sy2sb::panelQR(handle, cusolverHandle, m, n, A_inout, lda,
                             R.data(), ldr, W.data(), ldw);

    return;
}

}  // namespace matrix_ops