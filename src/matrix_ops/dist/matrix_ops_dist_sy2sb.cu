#include <nccl.h>
#include <omp.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "fmt/format.h"
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
void computeAwMgpu(std::vector<common::CublasHandle>& cublas_handle_mg,
                   size_t omp_index, size_t gpu_index, size_t panel_m,
                   size_t nb, size_t b, size_t occupy_each_gpu, size_t i,
                   size_t recrusive_offset, size_t recrusive_offset_finished,
                   size_t rest_gpu_num, size_t n, ncclDataType_t nccl_type,
                   size_t lda, size_t ldz, thrust::device_ptr<T> panel_Z_ptr,
                   std::vector<size_t>& gpu_start,
                   std::vector<thrust::device_vector<T>>& gpu_work,
                   std::vector<thrust::device_vector<T>>& gpu_oriA,
                   std::vector<thrust::device_vector<T>>& z_recv,
                   std::vector<cudaStream_t>& streams,
                   std::vector<ncclComm_t>& comms,
                   std::vector<ncclComm_t>& comms_bcast) {
    auto omp_gpu_index = gpu_index + omp_index;
    cudaSetDevice(omp_gpu_index);
    ncclBcast(gpu_work[omp_gpu_index].data().get(), b * panel_m, nccl_type, 0,
              comms_bcast[omp_gpu_index], streams[omp_gpu_index]);
    cudaStreamSynchronize(streams[omp_gpu_index]);

    auto& panel_handle = cublas_handle_mg[omp_gpu_index];
    auto oriA_panel =
        gpu_oriA[omp_gpu_index].data() + i + recrusive_offset_finished;
    auto z_panel_rows = occupy_each_gpu;

    auto aw_panel = gpu_work[omp_gpu_index].data() + n * nb;

    if (omp_index == 0) {
        oriA_panel = gpu_oriA[omp_gpu_index].data() + recrusive_offset -
                     gpu_start[omp_gpu_index] + i + i * lda;
        z_panel_rows = panel_m - occupy_each_gpu * (rest_gpu_num - 1);
    }

    if (z_panel_rows > 0) {
        try {
            matrix_ops::gemm(panel_handle, z_panel_rows, b, panel_m, (T)1,
                             oriA_panel, lda, true,
                             gpu_work[omp_gpu_index].data(), panel_m, false,
                             (T)0, aw_panel, z_panel_rows);
        } catch (const std::exception& e) {
            throw std::runtime_error(
                fmt::format("here aw gemm error exception: {}", e.what()));
        } catch (...) {
            throw std::runtime_error(
                "here aw gemm error: an unknown exception "
                "occurred");
        }
    }
    ncclGroupStart();
    if (omp_index != 0) {
        auto gpu_id = gpu_index + omp_index;
        ncclSend(aw_panel.get(), occupy_each_gpu * b, nccl_type, gpu_index,
                 comms[gpu_id], streams[gpu_id]);

    } else {
        for (auto gpu_offset = 1; gpu_offset < rest_gpu_num; gpu_offset++) {
            auto gpu_id = gpu_index + gpu_offset;
            ncclRecv(z_recv[gpu_offset - 1].data().get(), occupy_each_gpu * b,
                     nccl_type, gpu_id, comms[gpu_index], streams[gpu_index]);
        }
    }
    ncclGroupEnd();

    cudaStreamSynchronize(streams[omp_gpu_index]);

    if (omp_index == 0) {
        if (z_panel_rows > 0) {
            matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                    thrust::device_ptr<T>, T>(
                aw_panel, z_panel_rows, panel_Z_ptr, ldz, z_panel_rows, b);
        }

        for (auto index = 1; index < rest_gpu_num; index++) {
            auto row_finished = (index - 1) * occupy_each_gpu + panel_m -
                                occupy_each_gpu * (rest_gpu_num - 1);
            matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                    thrust::device_ptr<T>, T>(
                z_recv[index - 1].data(), occupy_each_gpu,
                panel_Z_ptr + row_finished, ldz, occupy_each_gpu, b);
        }
    }
}

template <typename T>
void sy2sb_recrusive(size_t recrusive_depth,
                     std::vector<common::CublasHandle>& cublas_handle_mg,
                     std::vector<common::CusolverDnHandle>& cusolver_handle_mg,
                     const common::CublasXtHandle& cublasXtHandle, size_t n,
                     T* oriA_host, size_t lda, T* Y_host, size_t ldy, T* W_host,
                     size_t ldw, size_t nb, size_t b, size_t gpu_num,
                     size_t occupy_each_gpu, std::vector<size_t>& gpu_start,
                     std::vector<thrust::device_vector<T>>& gpu_A,
                     std::vector<thrust::device_vector<T>>& gpu_W,
                     std::vector<thrust::device_vector<T>>& gpu_Y,
                     std::vector<thrust::device_vector<T>>& gpu_R,
                     std::vector<thrust::device_vector<T>>& gpu_Z,
                     std::vector<thrust::device_vector<T>>& gpu_work,
                     std::vector<thrust::device_vector<T>>& gpu_oriA,
                     std::vector<cudaStream_t>& streams,
                     std::vector<ncclComm_t>& comms) {
    auto nccl_type = ncclFloat32;
    if constexpr (std::is_same_v<T, float>) {
        nccl_type = ncclFloat32;
    } else if constexpr (std::is_same_v<T, double>) {
        nccl_type = ncclFloat64;
    }

    auto recrusive_offset = recrusive_depth * (nb + nb * lda);
    auto gpu_index = computeGPUIndex4Panel(recrusive_offset, gpu_start);
    auto recrusive_offset_finished = nb * recrusive_depth;
    auto A = gpu_A[gpu_index].data() + recrusive_offset - gpu_start[gpu_index];
    auto W = gpu_W[gpu_index].data() + recrusive_offset - gpu_start[gpu_index];
    auto Y = gpu_Y[gpu_index].data() + recrusive_offset - gpu_start[gpu_index];
    auto R = gpu_R[gpu_index].data() + recrusive_offset_finished;
    auto Z = gpu_Z[gpu_index].data();
    auto ldz = lda;
    auto work_ptr = gpu_work[gpu_index].data();
    auto ldwork = nb;
    auto& handle = cublas_handle_mg[gpu_index];
    auto& cusolverHandle = cusolver_handle_mg[gpu_index];

    // create sub communicator
    std::vector<ncclComm_t> bcast_comms(gpu_num, nullptr);
#pragma omp parallel num_threads(gpu_num)
    {
        int my_gpu_id = omp_get_thread_num();
        cudaSetDevice(my_gpu_id);

        int color = (my_gpu_id >= gpu_index) ? 1 : NCCL_SPLIT_NOCOLOR;
        int key = my_gpu_id;

        ncclCommSplit(comms[my_gpu_id], color, key, &bcast_comms[my_gpu_id],
                      NULL);
    }

    cudaSetDevice(gpu_index);

    for (auto i = b; i <= nb && i < (n - recrusive_offset_finished); i += b) {
        auto panel_m = n - recrusive_offset_finished - i;
        auto panel_n = b;
        auto panel_ptr = A + i + (i - b) * lda;

        auto panel_W_ptr = W + i + (i - b) * ldw;
        auto panel_Y_ptr = Y + i + (i - b) * ldy;
        auto panel_Z_ptr = Z + i + (i - b) * ldz;

        matrix_ops::internal::sy2sb::panelQR(handle, cusolverHandle, panel_m,
                                             panel_n, panel_ptr, lda, R, lda,
                                             panel_W_ptr, ldw);

        // copy panel data to panelY (using lda)
        matrix_ops::matrix_copy<thrust::device_ptr<T>, thrust::device_ptr<T>,
                                T>(panel_ptr, lda, panel_Y_ptr, ldy, panel_m,
                                   panel_n);

        // copy panelR data to panel (using lda)
        matrix_ops::matrix_copy<thrust::device_ptr<T>, thrust::device_ptr<T>,
                                T>(R, lda, panel_ptr, lda, panel_m, panel_n);

        // update A by ZY mode
        // first panel process
        auto rest_gpu_num = gpu_num - gpu_index;

        if (rest_gpu_num > 1) {
            // copy W to workspace
            matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                    thrust::device_ptr<T>, T>(
                panel_W_ptr, ldw, gpu_work[gpu_index].data(), panel_m, panel_m,
                b);
            std::vector<thrust::device_vector<T>> z_recv(rest_gpu_num - 1);
            for (int index = 0; index < rest_gpu_num - 1; index++) {
                z_recv[index].resize(occupy_each_gpu * b);
            }
#pragma omp parallel num_threads(rest_gpu_num)
            {
                auto omp_index = omp_get_thread_num();

                computeAwMgpu(cublas_handle_mg, omp_index, gpu_index, panel_m,
                              nb, b, occupy_each_gpu, i, recrusive_offset,
                              recrusive_offset_finished, rest_gpu_num, n,
                              nccl_type, lda, ldz, panel_Z_ptr, gpu_start,
                              gpu_work, gpu_oriA, z_recv, streams, comms,
                              bcast_comms);
            }
        } else {
            // for single card, we just execute gemm
            auto oriA_panel = gpu_oriA[gpu_index].data() + recrusive_offset -
                              gpu_start[gpu_index] + i + i * lda;
            matrix_ops::gemm(handle, panel_m, b, panel_m, (T)1, oriA_panel, lda,
                             false, panel_W_ptr, ldw, false, (T)0, panel_Z_ptr,
                             ldz);
        }

        if (i == b) {
            // panel_tmp = panel_z^T * panel_z
            matrix_ops::gemm(handle, b, b, panel_m, (T)1, panel_W_ptr, ldw,
                             true, panel_Z_ptr, ldz, false, (T)0, work_ptr,
                             ldwork);
            // panel_z = panel_z - panel_y * panel_z^T * panel_z
            matrix_ops::gemm(handle, panel_m, b, b, (T)(-0.5), panel_Y_ptr, ldy,
                             false, work_ptr, ldwork, false, (T)1, panel_Z_ptr,
                             ldz);
        } else {
            // panel_tmp = (Z + i)^T * panel_w
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
            common::CublasHandle tmp_handle;

            matrix_ops::gemm(tmp_handle, panel_m, b, i, (T)(-1), Y + i, ldy,
                             false, Z + i, ldz, true, (T)1, A + i + i * lda,
                             lda);

            matrix_ops::gemm(tmp_handle, panel_m, b, i, (T)(-1), Z + i, ldz,
                             false, Y + i, ldy, true, (T)1, A + i + i * lda,
                             lda);
        }
    }

    // recursive exit
    if (n <= nb + recrusive_offset_finished) return;

    auto offset = (recrusive_depth + 1) * (nb + nb * lda);
    auto tail_gpu_start_index = computeGPUIndex4Panel(offset, gpu_start);
    auto rest_gpu_num = gpu_num - gpu_index;
    auto tail_matrix_ptr = gpu_oriA[tail_gpu_start_index].data() + offset -
                           gpu_start[tail_gpu_start_index];
    auto sub_matrix_n = n - recrusive_offset_finished - nb;

    cudaSetDevice(gpu_index);

    thrust::device_vector<T> z_send(n * nb);

    if (tail_gpu_start_index == gpu_index) {
        if (rest_gpu_num == 1) {
            matrix_ops::syr2k(handle, n - recrusive_offset_finished - nb, nb,
                              (T)(-1), Y + nb, ldy, Z + nb, ldz, (T)1,
                              tail_matrix_ptr, lda);

            thrust::for_each(
                thrust::make_counting_iterator<size_t>(0),
                thrust::make_counting_iterator<size_t>(sub_matrix_n *
                                                       sub_matrix_n),
                make_symmetric_functor<T>(tail_matrix_ptr, sub_matrix_n, lda));

            matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                    thrust::device_ptr<T>, T>(
                tail_matrix_ptr, lda, A + nb + nb * lda, lda,
                n - recrusive_offset_finished - nb,
                n - recrusive_offset_finished - nb);
        } else {
            matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                    thrust::device_ptr<T>, T>(
                Y + nb, lda, gpu_work[gpu_index].data(), sub_matrix_n,
                sub_matrix_n, nb);

            matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                    thrust::device_ptr<T>, T>(
                Z + nb, lda, z_send.data(), sub_matrix_n, sub_matrix_n, nb);
#pragma omp parallel num_threads(rest_gpu_num)
            {
                auto gpu_id = omp_get_thread_num() + gpu_index;
                cudaSetDevice(gpu_id);
                auto z_bcast = gpu_Z[gpu_id].data();
                if (gpu_id == gpu_index) {
                    z_bcast = z_send.data();
                }
                ncclBcast(z_bcast.get(), sub_matrix_n * nb, nccl_type, 0,
                          bcast_comms[gpu_id], streams[gpu_id]);
                ncclBcast(gpu_work[gpu_id].data().get(), sub_matrix_n * nb,
                          nccl_type, 0, bcast_comms[gpu_id], streams[gpu_id]);
                auto syr2k_panel_col = occupy_each_gpu;
                auto& syr2k_panel_handle = cublas_handle_mg[gpu_id];
                auto syr2k_panel_oriA_ptr =
                    gpu_oriA[gpu_id].data() + (n - sub_matrix_n);
                auto dst_A_ptr = gpu_A[gpu_id].data() + (n - sub_matrix_n);

                auto zy_panel_offset =
                    sub_matrix_n - (rest_gpu_num - 1) * occupy_each_gpu +
                    (gpu_id - gpu_index - 1) * occupy_each_gpu;

                if (gpu_id == gpu_index) {
                    syr2k_panel_col =
                        sub_matrix_n - (rest_gpu_num - 1) * occupy_each_gpu;
                    syr2k_panel_oriA_ptr = tail_matrix_ptr;
                    dst_A_ptr =
                        gpu_A[gpu_index].data() + offset - gpu_start[gpu_index];
                    zy_panel_offset = 0;
                }

                matrix_ops::gemm(
                    syr2k_panel_handle, sub_matrix_n, syr2k_panel_col, nb,
                    T(-1), z_bcast, sub_matrix_n, false,
                    gpu_work[gpu_id].data() + zy_panel_offset, sub_matrix_n,
                    true, T(1), syr2k_panel_oriA_ptr, lda);
                matrix_ops::gemm(syr2k_panel_handle, sub_matrix_n,
                                 syr2k_panel_col, nb, T(-1),
                                 gpu_work[gpu_id].data(), sub_matrix_n, false,
                                 z_bcast + zy_panel_offset, sub_matrix_n, true,
                                 T(1), syr2k_panel_oriA_ptr, lda);

                matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                        thrust::device_ptr<T>, T>(
                    syr2k_panel_oriA_ptr, lda, dst_A_ptr, lda, sub_matrix_n,
                    syr2k_panel_col);
            }
        }
    } else {
        if (rest_gpu_num == 1) {
            matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                    thrust::device_ptr<T>, T>(
                Y + nb, lda, gpu_work[gpu_index].data(), sub_matrix_n,
                sub_matrix_n, nb);

            matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                    thrust::device_ptr<T>, T>(
                Z + nb, lda, z_send.data(), sub_matrix_n, sub_matrix_n, nb);

            try {
#pragma omp parallel num_threads(2)
                {
                    ncclGroupStart();

                    if (omp_get_thread_num() == 0) {
                        ncclSend(gpu_work[gpu_index].data().get(),
                                 sub_matrix_n * nb, nccl_type,
                                 tail_gpu_start_index, comms[gpu_index],
                                 streams[gpu_index]);
                    } else {
                        cudaSetDevice(tail_gpu_start_index);
                        ncclRecv(gpu_work[tail_gpu_start_index].data().get(),
                                 sub_matrix_n * nb, nccl_type, gpu_index,
                                 comms[tail_gpu_start_index],
                                 streams[tail_gpu_start_index]);
                        cudaSetDevice(gpu_index);
                    }

                    ncclGroupEnd();

                    if (omp_get_thread_num() == 0) {
                        cudaStreamSynchronize(streams[gpu_index]);
                    } else {
                        cudaStreamSynchronize(streams[tail_gpu_start_index]);
                    }
                }
            } catch (...) {
                throw std::runtime_error(fmt::format(
                    "NCCL error because of the Y send/recv sub_matrix_n {} "
                    "gpu_index {} tail_gpu_start_index {}",
                    sub_matrix_n, gpu_index, tail_gpu_start_index));
            }

            try {
#pragma omp parallel num_threads(2)
                {
                    ncclGroupStart();
                    if (omp_get_thread_num() == 0) {
                        ncclSend(z_send.data().get(), sub_matrix_n * nb,
                                 nccl_type, tail_gpu_start_index,
                                 comms[gpu_index], streams[gpu_index]);
                    } else {
                        cudaSetDevice(tail_gpu_start_index);
                        ncclRecv(gpu_Z[tail_gpu_start_index].data().get(),
                                 sub_matrix_n * nb, nccl_type, gpu_index,
                                 comms[tail_gpu_start_index],
                                 streams[tail_gpu_start_index]);
                        cudaSetDevice(gpu_index);
                    }

                    ncclGroupEnd();

                    if (omp_get_thread_num() == 0) {
                        cudaStreamSynchronize(streams[gpu_index]);
                    } else {
                        cudaStreamSynchronize(streams[tail_gpu_start_index]);
                    }
                }
            } catch (...) {
                throw std::runtime_error(fmt::format(
                    "NCCL error because of the Z send/recv sub_matrix_n {} "
                    "gpu_index {} tail_gpu_start_index {}",
                    sub_matrix_n, gpu_index, tail_gpu_start_index));
            }

            cudaSetDevice(tail_gpu_start_index);
            auto& syr2k_handle = cublas_handle_mg[tail_gpu_start_index];
            matrix_ops::syr2k(syr2k_handle, sub_matrix_n, nb, (T)(-1),
                              gpu_work[tail_gpu_start_index].data(),
                              sub_matrix_n, gpu_Z[tail_gpu_start_index].data(),
                              sub_matrix_n, (T)1, tail_matrix_ptr, lda);
            thrust::for_each(
                thrust::make_counting_iterator<size_t>(0),
                thrust::make_counting_iterator<size_t>(sub_matrix_n *
                                                       sub_matrix_n),
                make_symmetric_functor<T>(tail_matrix_ptr, sub_matrix_n, lda));
            matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                    thrust::device_ptr<T>, T>(
                tail_matrix_ptr, lda,
                gpu_A[tail_gpu_start_index].data() + offset -
                    gpu_start[tail_gpu_start_index],
                lda, sub_matrix_n, sub_matrix_n);
        } else {
            matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                    thrust::device_ptr<T>, T>(
                Y + nb, lda, gpu_work[gpu_index].data(), sub_matrix_n,
                sub_matrix_n, nb);

            matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                    thrust::device_ptr<T>, T>(
                Z + nb, lda, z_send.data(), sub_matrix_n, sub_matrix_n, nb);
#pragma omp parallel num_threads(rest_gpu_num)
            {
                auto gpu_id = omp_get_thread_num() + gpu_index;
                cudaSetDevice(gpu_id);
                auto z_bcast = gpu_Z[gpu_id].data();
                if (gpu_id == gpu_index) {
                    z_bcast = z_send.data();
                }
                ncclBcast(z_bcast.get(), sub_matrix_n * nb, nccl_type, 0,
                          bcast_comms[gpu_id], streams[gpu_id]);
                cudaStreamSynchronize(streams[gpu_id]);
                ncclBcast(gpu_work[gpu_id].data().get(), sub_matrix_n * nb,
                          nccl_type, 0, bcast_comms[gpu_id], streams[gpu_id]);
                cudaStreamSynchronize(streams[gpu_id]);
                if (gpu_id != gpu_index) {
                    auto syr2k_panel_col = occupy_each_gpu;
                    auto& syr2k_panel_handle = cublas_handle_mg[gpu_id];
                    auto syr2k_panel_oriA_ptr =
                        gpu_oriA[gpu_id].data() + (n - sub_matrix_n);
                    auto dst_A_ptr = gpu_A[gpu_id].data() + (n - sub_matrix_n);

                    auto zy_panel_offset = (gpu_id - gpu_index - 1) * nb;

                    matrix_ops::gemm(
                        syr2k_panel_handle, sub_matrix_n, syr2k_panel_col, nb,
                        T(-1), z_bcast, sub_matrix_n, false,
                        gpu_work[gpu_id].data() + zy_panel_offset, sub_matrix_n,
                        true, T(1), syr2k_panel_oriA_ptr, lda);
                    matrix_ops::gemm(
                        syr2k_panel_handle, sub_matrix_n, syr2k_panel_col, nb,
                        T(-1), gpu_work[gpu_id].data(), sub_matrix_n, false,
                        z_bcast + zy_panel_offset, sub_matrix_n, true, T(1),
                        syr2k_panel_oriA_ptr, lda);

                    matrix_ops::matrix_copy<thrust::device_ptr<T>,
                                            thrust::device_ptr<T>, T>(
                        syr2k_panel_oriA_ptr, lda, dst_A_ptr, lda, sub_matrix_n,
                        syr2k_panel_col);
                }
            }
        }
    }

#pragma omp parallel num_threads(gpu_num)
    {
        int my_gpu_id = omp_get_thread_num();
        if (bcast_comms[my_gpu_id] != nullptr) {
            cudaSetDevice(my_gpu_id);
            ncclCommDestroy(bcast_comms[my_gpu_id]);
        }
    }

    internal::sy2sb_recrusive(recrusive_depth + 1, cublas_handle_mg,
                              cusolver_handle_mg, cublasXtHandle, n, oriA_host,
                              lda, Y_host, ldy, W_host, ldw, nb, b, gpu_num,
                              occupy_each_gpu, gpu_start, gpu_A, gpu_W, gpu_Y,
                              gpu_R, gpu_Z, gpu_work, gpu_oriA, streams, comms);
}  // namespace internal
}  // namespace internal

template <typename T>
void sy2sb(const common::CublasHandle& handle, size_t n, T* A, size_t lda, T* W,
           size_t ldw, T* Y, size_t ldy, size_t nb, size_t b, size_t gpu_num) {
    // print args
    util::Logger::println(
        "sy2sb: n={}, lda={}, ldw={}, ldy={}, nb={}, b={}, gpu_num={}", n, lda,
        ldw, ldy, nb, b, gpu_num);

    // get current device
    int init_device;
    cudaGetDevice(&init_device);

    auto cusolverHandle = common::CusolverDnHandle();

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

    std::vector<common::CublasHandle> gpu_cublas_handle;
    std::vector<common::CusolverDnHandle> gpu_cusolverdn_handle;

    std::vector<cudaStream_t> gpu_stream(gpu_num);

    std::vector<thrust::device_vector<T>> gpu_A(gpu_num);
    std::vector<thrust::device_vector<T>> gpu_W(gpu_num);
    std::vector<thrust::device_vector<T>> gpu_Y(gpu_num);
    std::vector<thrust::device_vector<T>> gpu_R(gpu_num);
    std::vector<thrust::device_vector<T>> gpu_Z(gpu_num);
    std::vector<thrust::device_vector<T>> gpu_work(gpu_num);
    std::vector<thrust::device_vector<T>> gpu_oriA(gpu_num);

    std::vector<ncclComm_t> comms(gpu_num);

    for (auto i = (size_t)0; i < gpu_num; i++) {
        cudaSetDevice(i);

        // stream
        cudaStreamCreate(&gpu_stream[i]);

        // handle
        gpu_cublas_handle.push_back(common::CublasHandle());
        cublasSetStream(gpu_cublas_handle[i], gpu_stream[i]);
        gpu_cusolverdn_handle.push_back(common::CusolverDnHandle());
        cusolverDnSetStream(gpu_cusolverdn_handle[i], gpu_stream[i]);

        // memory
        gpu_A[i] = thrust::device_vector<T>(occupy_each_gpu * n);
        gpu_W[i] = thrust::device_vector<T>(occupy_each_gpu * n, (T)0);
        gpu_Y[i] = thrust::device_vector<T>(occupy_each_gpu * n, (T)0);
        gpu_R[i] = thrust::device_vector<T>(n * nb);
        gpu_Z[i] = thrust::device_vector<T>(n * nb, 0);
        gpu_oriA[i] = thrust::device_vector<T>(occupy_each_gpu * n);
        gpu_work[i] = thrust::device_vector<T>(2 * n * nb);
        try {
            thrust::copy(A + gpu_start[i],
                         A + gpu_start[i] + occupy_each_gpu * n,
                         gpu_A[i].begin());
            thrust::copy(A + gpu_start[i],
                         A + gpu_start[i] + occupy_each_gpu * n,
                         gpu_oriA[i].begin());
        } catch (...) {
            throw std::runtime_error("gpu_A copy failed");
        }
    }

    if (n % nb % gpu_num != 0) {
        throw std::runtime_error(
            "n % nb != 0 we don't support non-divisible size");
    }

    common::CublasXtHandle cublasXtHandle;

    std::vector<int> gpu_used(gpu_num);
    std::iota(gpu_used.begin(), gpu_used.end(), 0);
    auto status = ncclCommInitAll(comms.data(), gpu_num, gpu_used.data());
    if (status != ncclSuccess) {
        throw std::runtime_error("ncclCommInitAll failed");
    }

    cublasXtDeviceSelect(cublasXtHandle, gpu_num, gpu_used.data());

    internal::sy2sb_recrusive(
        0, gpu_cublas_handle, gpu_cusolverdn_handle, cublasXtHandle, n, A, lda,
        Y, ldy, W, ldw, nb, b, gpu_num, occupy_each_gpu, gpu_start, gpu_A,
        gpu_W, gpu_Y, gpu_R, gpu_Z, gpu_work, gpu_oriA, gpu_stream, comms);

    for (auto i = (size_t)0; i < gpu_num; i++) {
        cudaSetDevice(i);
        thrust::copy(gpu_A[i].begin(), gpu_A[i].end(), A + gpu_start[i]);
        thrust::copy(gpu_W[i].begin(), gpu_W[i].end(), W + gpu_start[i]);
        thrust::copy(gpu_Y[i].begin(), gpu_Y[i].end(), Y + gpu_start[i]);
    }

    try {
        // TODO: 一个对称的伪实现
        thrust::device_vector<T> tmp(n * n);
        thrust::copy(A, A + n * n, tmp.begin());
        // make A_inout symmetric
        thrust::for_each(
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(n * n),
            internal::make_symmetric_functor<T>(tmp.data(), n, lda));
        thrust::copy(tmp.begin(), tmp.end(), A);
    } catch (...) {
        throw std::runtime_error("make symmetric failed");
    }

    cudaSetDevice(init_device);

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