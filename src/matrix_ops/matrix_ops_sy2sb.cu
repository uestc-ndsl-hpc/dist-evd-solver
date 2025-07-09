#include <thrust/device_vector.h>

#include <cstddef>

#include "fmt/format.h"
#include "gpu_handle_wrappers.h"
#include "internal/sy2sb/sy2sb_panelqr.cuh"
#include "log.h"
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
           thrust::device_ptr<T> A_inout, thrust::device_ptr<T> Y_inout,
           thrust::device_ptr<T> W_inout, size_t lda, size_t ldy, size_t ldw) {
    // the size of the matrix A
    const auto m = (size_t)n;
    // the panel size
    const auto b = (size_t)32;
    // the block size
    const auto nb = (size_t)b * 4;
    // tmp R for compute W && Y
    auto R = thrust::device_vector<T>(n * n);
    auto R_ptr = R.data();
    auto ldr = n;

    common::CusolverDnHandle cusolverHandle;

    // main loop to process the matrix A's first nb panel
    for (auto i = b; i <= nb && i < n; i += b) {
        // the b panel size
        const auto b_panel_m = m - i;
        const auto b_panel_n = b;
        // the b panel ptrs
        auto b_panel_ptr = A_inout + i + (i - b) * lda;
        auto b_panel_W_ptr = W_inout + i + (i - b) * ldw;
        auto b_panel_R_ptr = R_ptr + i + (i - b) * ldr;
        // the b panel QR
        if (util::Logger::is_verbose()) {
            matrix_ops::print(b_panel_ptr, b_panel_m, b_panel_n,
                              fmt::format("b panel {}", i));
        }
        internal::sy2sb::panelQR(handle, cusolverHandle, b_panel_m, b_panel_n,
                                 b_panel_ptr, lda, b_panel_R_ptr, ldr,
                                 b_panel_W_ptr, ldw);
    }

    return;
}

}  // namespace matrix_ops

template void matrix_ops::sy2sb<float>(const common::CublasHandle& handle,
                                       size_t n,
                                       thrust::device_ptr<float> A_inout,
                                       thrust::device_ptr<float> Y_inout,
                                       thrust::device_ptr<float> W_inout,
                                       size_t lda, size_t ldy, size_t ldw);
template void matrix_ops::sy2sb<double>(const common::CublasHandle& handle,
                                        size_t n,
                                        thrust::device_ptr<double> A_inout,
                                        thrust::device_ptr<double> Y_inout,
                                        thrust::device_ptr<double> W_inout,
                                        size_t lda, size_t ldy, size_t ldw);