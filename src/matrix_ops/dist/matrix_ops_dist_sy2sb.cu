#include <thrust/host_vector.h>

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "log.h"
#include "matrix_ops.cuh"
#include "matrix_ops_dist.cuh"
#include "sy2sb_panelqr.cuh"

namespace matrix_ops {
namespace dist {
namespace internal {
/**
 * @brief compute the gpu index for a panel
 *
 * @param offset the offset of the panel
 * @param gpu_start the starting points of the gpu
 * @return size_t the gpu index
 */
size_t computeGPUIndex4Panel(size_t offset, std::vector<size_t>& gpu_start) {
    auto it = std::upper_bound(gpu_start.begin(), gpu_start.end(), offset);
    if (it == gpu_start.begin()) {
        throw std::out_of_range("offset is smaller than all starting points.");
    }
    return std::distance(gpu_start.begin(), it) - 1;
}

template <typename T>
void sy2sb_recrusive(const common::CublasHandle& handle,
                     const common::CusolverDnHandle& cusolverHandle, size_t n,
                     T* A, size_t lda, T* Y, size_t ldy, T* W, size_t ldw, T* R,
                     size_t ldr, size_t nb, size_t b, size_t gpu_num,
                     size_t occupy_each_gpu, std::vector<size_t>& gpu_start,
                     std::vector<thrust::device_vector<T>>& gpu_A,
                     std::vector<thrust::device_vector<T>>& gpu_oriA,
                     std::vector<thrust::device_vector<T>>& gpu_W,
                     std::vector<thrust::device_vector<T>>& gpu_Y,
                     std::vector<thrust::device_vector<T>>& gpu_R) {
    if (n % nb != 0) {
        throw std::runtime_error(
            "n % nb != 0 we don't support non-divisible size");
    }

    for (auto i = b; i <= nb && i < n; i += b) {
        auto panel_m = n - i;
        auto panel_n = b;
        auto offset = i + (i - b) * lda;
        auto gpu_index = computeGPUIndex4Panel(offset, gpu_start);
        cudaSetDevice(gpu_index);
        auto panel_ptr =
            gpu_A[gpu_index].data() + offset - gpu_start[gpu_index];
        auto panel_W_ptr =
            gpu_W[gpu_index].data() + offset - gpu_start[gpu_index];
        auto panel_Y_ptr =
            gpu_Y[gpu_index].data() + offset - gpu_start[gpu_index];
        auto panel_oriA_ptr =
            gpu_oriA[gpu_index].data() + offset - gpu_start[gpu_index];
        auto panel_R_ptr =
            gpu_R[gpu_index].data() + offset - gpu_start[gpu_index];

        matrix_ops::internal::sy2sb::panelQR(
            handle, cusolverHandle, panel_m, panel_n, panel_ptr, lda,
            panel_R_ptr, ldr, panel_W_ptr, ldw);

        // copy panel data to panelY (using lda)
        matrix_ops::matrix_copy<thrust::device_ptr<T>, thrust::device_ptr<T>,
                                T>(panel_ptr, lda, panel_Y_ptr, ldy, panel_m,
                                   panel_n);

        // copy panelR data to panel (using lda)
        matrix_ops::matrix_copy<thrust::device_ptr<T>, thrust::device_ptr<T>,
                                T>(panel_R_ptr, ldr, panel_ptr, lda, panel_m,
                                   panel_n);

        
    }
}

}  // namespace internal

template <typename T>
void sy2sb(const common::CublasHandle& handle, size_t n, T* A, size_t lda, T* Y,
           size_t ldy, T* W, size_t ldw, size_t nb, size_t b, size_t gpu_num) {
    util::Logger::println("sy2sb dist");

    auto cusolverHandle = common::CusolverDnHandle();

    auto oriA = thrust::host_vector<T>(n * n);

    auto R_h = thrust::host_vector<T>(n * n);
    auto ldr = n;

    thrust::copy(A, A + n * n, oriA.begin());

    // compute the element amount by gpu number
    if (n % b % gpu_num != 0) {
        throw std::runtime_error(
            "now the matrix is not well to devide into gpu_num's panel");
    }
    auto occupy_each_gpu = n / gpu_num;
    std::vector<size_t> gpu_start(gpu_num, 0);
    for (int i = 0; i < gpu_num; i++) {
        gpu_start[i] = i * occupy_each_gpu * n;
    }
    std::vector<cudaStream_t> gpu_stream(gpu_num);
    for (auto i = (size_t)0; i < gpu_num; i++) {
        cudaSetDevice(i);
        cudaStreamCreate(&gpu_stream[i]);
    }
    std::vector<thrust::device_vector<T>> gpu_A(gpu_num);
    std::vector<thrust::device_vector<T>> gpu_oriA(gpu_num);
    std::vector<thrust::device_vector<T>> gpu_W(gpu_num);
    std::vector<thrust::device_vector<T>> gpu_Y(gpu_num);
    std::vector<thrust::device_vector<T>> gpu_R(gpu_num);
    for (auto i = (size_t)0; i < gpu_num; i++) {
        fmt::println("== set gpu {} ==", i);
        cudaSetDevice(i);
        gpu_A[i] = thrust::device_vector<T>(occupy_each_gpu * n);
        gpu_oriA[i] = thrust::device_vector<T>(occupy_each_gpu * n);
        gpu_W[i] = thrust::device_vector<T>(occupy_each_gpu * n);
        gpu_Y[i] = thrust::device_vector<T>(occupy_each_gpu * n);
        gpu_R[i] = thrust::device_vector<T>(occupy_each_gpu * n);
        thrust::copy(A + gpu_start[i], A + gpu_start[i] + occupy_each_gpu * n,
                     gpu_A[i].begin());
        thrust::copy(oriA.data() + gpu_start[i],
                     oriA.data() + gpu_start[i] + occupy_each_gpu * n,
                     gpu_oriA[i].begin());
    }
    for (auto i = (size_t)0; i < gpu_num; i++) {
        cudaSetDevice(i);
        std::string title_A = fmt::format("gpu_A_{}", i);
        matrix_ops::print(gpu_A[i], n, occupy_each_gpu, title_A);
        std::string title_oriA = fmt::format("gpu_oriA_{}", i);
        matrix_ops::print(gpu_oriA[i].data(), n, occupy_each_gpu, title_oriA);
    }

    internal::sy2sb_recrusive(handle, cusolverHandle, n, A, lda, Y, ldy, W, ldw,
                              R_h.data(), ldr, nb, b, gpu_num, occupy_each_gpu,
                              gpu_start, gpu_A, gpu_oriA, gpu_W, gpu_Y, gpu_R);

    return;
}
}  // namespace dist
}  // namespace matrix_ops

template void matrix_ops::dist::sy2sb<float>(const common::CublasHandle& handle,
                                             size_t n, float* A, size_t lda,
                                             float* Y, size_t ldy, float* W,
                                             size_t ldw, size_t nb, size_t b,
                                             size_t gpu_num);

template void matrix_ops::dist::sy2sb<double>(
    const common::CublasHandle& handle, size_t n, double* A, size_t lda,
    double* Y, size_t ldy, double* W, size_t ldw, size_t nb, size_t b,
    size_t gpu_num);