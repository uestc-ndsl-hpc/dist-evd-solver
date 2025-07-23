#include <thrust/host_vector.h>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "gpu_handle_wrappers.h"
#include "log.h"
#include "matrix_ops.cuh"
#include "matrix_ops_dist.cuh"
#include "sy2sb_panelqr.cuh"

namespace matrix_ops {
namespace dist {
namespace internal {

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
void sy2sb_recrusive(size_t recrusive_depth, const common::CublasHandle& handle,
                     const common::CusolverDnHandle& cusolverHandle,
                     const common::CublasXtHandle& cublasXtHandle, size_t n,
                     T* A_host, size_t lda, T* Y_host, size_t ldy, T* W_host,
                     size_t ldw, size_t nb, size_t b, size_t gpu_num,
                     size_t occupy_each_gpu, std::vector<size_t>& gpu_start,
                     std::vector<thrust::device_vector<T>>& gpu_A,
                     std::vector<thrust::device_vector<T>>& gpu_oriA,
                     std::vector<thrust::device_vector<T>>& gpu_W,
                     std::vector<thrust::device_vector<T>>& gpu_Y,
                     std::vector<thrust::device_vector<T>>& gpu_R,
                     std::vector<thrust::device_vector<T>>& gpu_Z,
                     std::vector<thrust::device_vector<T>>& gpu_work) {
    auto recrusive_offset = recrusive_depth * (nb + nb * lda);
    auto gpu_index = computeGPUIndex4Panel(recrusive_offset, gpu_start);
    cudaSetDevice(gpu_index);
    auto recrusive_offset_finished = nb * recrusive_depth;
    auto A = gpu_A[gpu_index].data() + recrusive_offset - gpu_start[gpu_index];

    auto oriA =
        gpu_oriA[gpu_index].data() + recrusive_offset - gpu_start[gpu_index];
    auto W = gpu_W[gpu_index].data() + recrusive_offset - gpu_start[gpu_index];
    auto Y = gpu_Y[gpu_index].data() + recrusive_offset - gpu_start[gpu_index];
    auto R = gpu_R[gpu_index].data() + recrusive_offset_finished;
    auto Z = gpu_Z[gpu_index].data();
    auto ldz = lda;
    auto work_ptr = gpu_work[gpu_index].data();
    auto ldwork = nb;

    util::Logger::println("recrusive_depth {} : before dbbr update",
                          recrusive_depth);

    for (auto i = b; i <= nb && i < (n - recrusive_offset_finished); i += b) {
        auto panel_m = n - recrusive_offset_finished - i;
        auto panel_n = b;
        auto panel_ptr = A + i + (i - b) * lda;

        auto panel_W_ptr = W + i + (i - b) * ldw;
        auto panel_Y_ptr = Y + i + (i - b) * ldy;
        auto panel_Z_ptr = Z + i + (i - b) * nb;

        matrix_ops::internal::sy2sb::panelQR(handle, cusolverHandle, panel_m,
                                             panel_n, panel_ptr, lda, R, lda,
                                             panel_W_ptr, ldw);

        util::Logger::println(
            "recrusive_depth {} : finished panelQR in panel {}",
            recrusive_depth, i / b);

        // copy panel data to panelY (using lda)
        matrix_ops::matrix_copy<thrust::device_ptr<T>, thrust::device_ptr<T>,
                                T>(panel_ptr, lda, panel_Y_ptr, ldy, panel_m,
                                   panel_n);

        // copy panelR data to panel (using lda)
        matrix_ops::matrix_copy<thrust::device_ptr<T>, thrust::device_ptr<T>,
                                T>(R, lda, panel_ptr, lda, panel_m, panel_n);

        // update A by ZY mode
        // first panel process
        auto panel_OriA_ptr_xt = A_host + recrusive_offset + i * lda + i;
        if (i == b) {
            if constexpr (std::is_same_v<T, float>) {
                float alpha = 1.0f;
                float beta = 0.0f;
                cublasXtSgemm(cublasXtHandle, CUBLAS_OP_N, CUBLAS_OP_N, panel_m,
                              b, panel_m, &alpha, panel_OriA_ptr_xt, lda,
                              panel_W_ptr.get(), ldw, &beta, panel_Z_ptr.get(),
                              ldz);
            } else {
                double alpha = 1.0;
                double beta = 0.0;
                cublasXtDgemm(cublasXtHandle, CUBLAS_OP_N, CUBLAS_OP_N, panel_m,
                              b, panel_m, &alpha, panel_OriA_ptr_xt, lda,
                              panel_W_ptr.get(), ldw, &beta, panel_Z_ptr.get(),
                              ldz);
            }
            // panel_tmp = panel_z^T * panel_z
            matrix_ops::gemm(handle, b, b, panel_m, (T)1, panel_W_ptr, ldw,
                             true, panel_Z_ptr, ldz, false, (T)0, work_ptr,
                             ldwork);
            // panel_z = panel_z - panel_y * panel_z^T * panel_z
            matrix_ops::gemm(handle, panel_m, b, b, (T)(-0.5), panel_Y_ptr, ldy,
                             false, work_ptr, ldwork, false, (T)1, panel_Z_ptr,
                             ldz);
        } else {
            if constexpr (std::is_same_v<T, float>) {
                float alpha = 1.0f;
                float beta = 0.0f;
                cublasXtSgemm(cublasXtHandle, CUBLAS_OP_N, CUBLAS_OP_N, panel_m,
                              b, panel_m, &alpha, panel_OriA_ptr_xt, lda,
                              panel_W_ptr.get(), ldw, &beta, panel_Z_ptr.get(),
                              ldz);
            } else {
                double alpha = 1.0;
                double beta = 0.0;
                cublasXtDgemm(cublasXtHandle, CUBLAS_OP_N, CUBLAS_OP_N, panel_m,
                              b, panel_m, &alpha, panel_OriA_ptr_xt, lda,
                              panel_W_ptr.get(), ldw, &beta, panel_Z_ptr.get(),
                              ldz);
            }
            // panel_tmp = panel_z^T * panel_w
            matrix_ops::gemm(handle, i - b, b, panel_m, (T)1, Z + i, ldz, true,
                             panel_W_ptr, ldw, false, (T)0, work_ptr, ldwork);
            // panel_z = panel_z - Y+i * panel_z^T * panel_w
            matrix_ops::gemm(handle, panel_m, b, i - b, (T)(-1), Y + i, ldy,
                             false, work_ptr, ldwork, false, (T)1, panel_Z_ptr,
                             ldz);
            // panel_tmp = Y+i^T * panel_w
            matrix_ops::gemm(handle, i - b, b, panel_m, (T)(1), Y + i, ldy,
                             true, panel_W_ptr, ldw, false, (T)0, work_ptr,
                             ldwork);
            // panel_z = panel_z - (Z + i) * Y+i^T * panel_w
            matrix_ops::gemm(handle, panel_m, b, i - b, (T)(-1), Z + i, ldz,
                             false, work_ptr, ldwork, false, (T)1, panel_Z_ptr,
                             ldz);
            // panel_tmp = panel_w^T * panel_z
            matrix_ops::gemm(handle, b, b, panel_m, (T)1, panel_W_ptr, ldw,
                             true, panel_Z_ptr, ldz, false, (T)0, work_ptr,
                             ldwork);
            // panel_z = panel_z - 0.5 * panel_y * panel_w^T * panel_z
            matrix_ops::gemm(handle, panel_m, b, b, (T)(-0.5), panel_Y_ptr, ldy,
                             false, work_ptr, ldwork, false, (T)1, panel_Z_ptr,
                             ldz);
        }
        if (i < nb) {
            matrix_ops::gemm(handle, panel_m, b, i, (T)(-1), Y + i, ldy, false,
                             Z + i, ldz, true, (T)1, A + i + i * lda, lda);
            matrix_ops::gemm(handle, panel_m, b, i, (T)(-1), Z + i, ldz, false,
                             Y + i, ldy, true, (T)1, A + i + i * lda, lda);
        }

        util::Logger::println(
            "recrusive_depth {} : finished zy update in panel {}",
            recrusive_depth, i / b);
    }

    util::Logger::println("recrusive_depth {} : finished dbbr update",
                          recrusive_depth);

    // recursive exit
    if (n <= nb + recrusive_offset_finished) return;

    auto tail_matrix_host_ptr =
        A_host + (recrusive_depth + 1) * (nb + nb * lda);

    if constexpr (std::is_same_v<T, float>) {
        float alpha = -1.0f;
        float beta = 1.0f;
        cublasXtSsyr2k(cublasXtHandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                       n - recrusive_offset_finished - nb, nb, &alpha,
                       Y.get() + nb, ldy, Z.get() + nb, ldz, &beta,
                       tail_matrix_host_ptr, lda);
    } else {
        double alpha = -1.0;
        double beta = 1.0;
        cublasXtDsyr2k(cublasXtHandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                       n - recrusive_offset_finished - nb, nb, &alpha,
                       Y.get() + nb, ldy, Z.get() + nb, ldz, &beta,
                       tail_matrix_host_ptr, lda);
    }

    util::Logger::println("recrusive_depth {} : finished syr2k update",
                          recrusive_depth);

    // TODO: 待实际实现 copy Lower to Upper to build a full symmetric matrix for
    // Z = AW the updated part
    auto sub_matrix_n = n - nb - recrusive_offset_finished;
    {
        auto tmp = thrust::device_vector<T>(n * n);
        auto sub_matrix_ptr_oA = tmp.data() + recrusive_offset + nb * lda + nb;
        thrust::copy(A_host, A_host + n * n, tmp.begin());
        thrust::for_each(
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(sub_matrix_n * sub_matrix_n),
            make_symmetric_functor<T>(sub_matrix_ptr_oA, sub_matrix_n, lda));
        thrust::copy(tmp.begin(), tmp.end(), A_host);
        auto A_now = thrust::host_vector<T>(n * n);
        for (auto i = 0; i < gpu_num; i++) {
            cudaSetDevice(i);
            thrust::copy(gpu_A[i].begin(), gpu_A[i].end(),
                         A_now.begin() + gpu_start[i]);
        }
        matrix_ops::matrix_copy<T*, T*, T>(
            tail_matrix_host_ptr, lda,
            A_now.data() + recrusive_offset + nb * lda + nb, lda, sub_matrix_n,
            sub_matrix_n);
        for (auto i = 0; i < gpu_num; i++) {
            cudaSetDevice(i);
            thrust::copy(A_now.begin() + gpu_start[i],
                         A_now.begin() + gpu_start[i] + occupy_each_gpu * n,
                         gpu_A[i].begin());
        }
    }

    util::Logger::println("recrusive_depth {} : make symmetric update",
                          recrusive_depth);

    internal::sy2sb_recrusive(
        recrusive_depth + 1, handle, cusolverHandle, cublasXtHandle, n, A_host,
        lda, Y_host, ldy, W_host, ldw, nb, b, gpu_num, occupy_each_gpu,
        gpu_start, gpu_A, gpu_oriA, gpu_W, gpu_Y, gpu_R, gpu_Z, gpu_work);
}
}  // namespace internal

template <typename T>
void sy2sb(const common::CublasHandle& handle, size_t n, T* A, size_t lda, T* Y,
           size_t ldy, T* W, size_t ldw, size_t nb, size_t b, size_t gpu_num) {
    auto cusolverHandle = common::CusolverDnHandle();

    auto oriA = thrust::host_vector<T>(n * n);

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
    std::vector<thrust::device_vector<T>> gpu_Z(gpu_num);
    std::vector<thrust::device_vector<T>> gpu_work(gpu_num);

    for (auto i = (size_t)0; i < gpu_num; i++) {
        cudaSetDevice(i);
        gpu_A[i] = thrust::device_vector<T>(occupy_each_gpu * n);
        gpu_oriA[i] = thrust::device_vector<T>(occupy_each_gpu * n);
        gpu_W[i] = thrust::device_vector<T>(occupy_each_gpu * n);
        gpu_Y[i] = thrust::device_vector<T>(occupy_each_gpu * n);
        gpu_R[i] = thrust::device_vector<T>(n * nb);
        gpu_Z[i] = thrust::device_vector<T>(n * nb);
        gpu_work[i] = thrust::device_vector<T>(nb * nb);
        thrust::copy(A + gpu_start[i], A + gpu_start[i] + occupy_each_gpu * n,
                     gpu_A[i].begin());
        thrust::copy(oriA.data() + gpu_start[i],
                     oriA.data() + gpu_start[i] + occupy_each_gpu * n,
                     gpu_oriA[i].begin());
    }

    if (n % nb % gpu_num != 0) {
        throw std::runtime_error(
            "n % nb != 0 we don't support non-divisible size");
    }

    common::CublasXtHandle cublasXtHandle;

    std::vector<int> gpu_used_xt(gpu_num);
    std::iota(gpu_used_xt.begin(), gpu_used_xt.end(), 0);
    cublasXtDeviceSelect(cublasXtHandle, gpu_num, gpu_used_xt.data());

    internal::sy2sb_recrusive(0, handle, cusolverHandle, cublasXtHandle, n, A,
                              lda, Y, ldy, W, ldw, nb, b, gpu_num,
                              occupy_each_gpu, gpu_start, gpu_A, gpu_oriA,
                              gpu_W, gpu_Y, gpu_R, gpu_Z, gpu_work);

    for (auto i = (size_t)0; i < gpu_num; i++) {
        cudaSetDevice(i);
        thrust::copy(gpu_A[i].begin(), gpu_A[i].end(), A + gpu_start[i]);
        thrust::copy(gpu_W[i].begin(), gpu_W[i].end(), W + gpu_start[i]);
        thrust::copy(gpu_Y[i].begin(), gpu_Y[i].end(), Y + gpu_start[i]);
    }

    util::Logger::println("sy2sb dist finished");

    {
        // TODO: 一个对称的伪实现
        thrust::device_vector<T> tmp(n * n);
        thrust::copy(A, A + n * n, tmp.begin());
        // make A_inout symmetric
        thrust::for_each(
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(n * n),
            internal::make_symmetric_functor<T>(tmp.data(), n, lda));
        thrust::copy(tmp.begin(), tmp.end(), A);
    }

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