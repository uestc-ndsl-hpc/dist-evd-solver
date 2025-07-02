#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <cassert>
#include <type_traits>

#include "matrix_ops.cuh"

namespace matrix_ops {
    
namespace internal {

// 使用constexpr替代宏定义，更符合C++风格
constexpr int TSQR_BLOCK_DIM_X = 32;
constexpr int TSQR_BLOCK_DIM_Y = 32;
constexpr int TSQR_NUM_DATA_ROW = 8;
constexpr int TSQR_BLOCK_SIZE = TSQR_BLOCK_DIM_X * TSQR_NUM_DATA_ROW;  // 256

template <typename T>
struct shared_memory;

template <>
struct shared_memory<float> {
    __device__ static float* get_pointer() {
        extern __shared__ float shared_mem_float[];
        return shared_mem_float;
    }
};

template <>
struct shared_memory<double> {
    __device__ static double* get_pointer() {
        extern __shared__ double shared_mem_double[];
        return shared_mem_double;
    }
};

template <typename T>
static __inline__ __device__ T warp_all_reduce_sum(T val) {
    for (int mask = warpSize / 2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <typename T>
__global__ void tsqr_kernel(int m, int n, T* A, int lda, T* R, int ldr) {
    shared_memory<T> shared;
    T* shared_A = shared.get_pointer();
    int ldsa = TSQR_BLOCK_SIZE;

    const int thread_idx_x = threadIdx.x;
    const int thread_idx_y = threadIdx.y;
    const int block_idx_x = blockIdx.x;

    int block_size = min(TSQR_BLOCK_SIZE, m - block_idx_x * TSQR_BLOCK_SIZE);

    A = A + block_idx_x * TSQR_BLOCK_SIZE;
    R = R + block_idx_x * n;

    int num_data_col = (n + TSQR_BLOCK_DIM_Y - 1) / TSQR_BLOCK_DIM_Y;

    T acc[TSQR_NUM_DATA_ROW];

#pragma unroll
    for (int k = 0; k < TSQR_NUM_DATA_ROW; ++k) {
        int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
        if (row_idx < block_size) {
            for (int h = 0; h < num_data_col; ++h) {
                int col_idx = thread_idx_y + h * TSQR_BLOCK_DIM_Y;
                if (col_idx < n) {
                    shared_A[row_idx + col_idx * ldsa] =
                        A[row_idx + col_idx * lda];
                }
            }
        }
    }

    __syncthreads();

    T q[TSQR_NUM_DATA_ROW];

    for (int cols = 0; cols < n; cols++) {
        T nu = 0.0;
        if (thread_idx_y == cols % TSQR_BLOCK_DIM_Y) {
#pragma unroll
            for (int k = 0; k < TSQR_NUM_DATA_ROW; k++) {
                acc[k] = 0.0;
                int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
                if (row_idx >= cols && row_idx < block_size) {
                    q[k] = shared_A[row_idx + cols * ldsa];
                    acc[k] = q[k] * q[k];
                }
                nu += acc[k];
            }

            T norm_x_square = warp_all_reduce_sum(nu);
            T norm_x = sqrt(norm_x_square);

            T scale = 1.0 / norm_x;
#pragma unroll
            for (int k = 0; k < TSQR_NUM_DATA_ROW; k++) {
                int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
                if (row_idx >= cols && row_idx < block_size) {
                    q[k] *= scale;
                }
            }

            int thread_idx = cols % TSQR_BLOCK_DIM_X;
            int thread_off = cols / TSQR_BLOCK_DIM_X;
            T u1 = 0;
            if (thread_idx_x == thread_idx) {
                q[thread_off] += (q[thread_off] >= 0) ? 1 : -1;
                u1 = q[thread_off];
                R[cols + cols * ldr] = (u1 >= 0) ? -norm_x : norm_x;
            }
            u1 = __shfl_sync(0xFFFFFFFF, u1, thread_idx);

            scale = 1.0 / (sqrt(abs(u1)));
#pragma unroll
            for (int k = 0; k < TSQR_NUM_DATA_ROW; k++) {
                int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
                if (row_idx >= cols && row_idx < block_size) {
                    shared_A[row_idx + cols * ldsa] = q[k] * scale;
                }
            }
        }

        __syncthreads();

        for (int h = 0; h < num_data_col; h++) {
            int opCols = thread_idx_y + h * TSQR_BLOCK_DIM_Y;
            if (cols < opCols && opCols < n) {
                nu = 0.0;
#pragma unroll
                for (int k = 0; k < TSQR_NUM_DATA_ROW; k++) {
                    acc[k] = 0.0;
                    int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
                    if (row_idx >= cols && row_idx < block_size) {
                        q[k] = shared_A[row_idx + cols * ldsa];
                        acc[k] = q[k] * shared_A[row_idx + opCols * ldsa];
                    }
                    nu += acc[k];
                }
                T utx = warp_all_reduce_sum(nu);

#pragma unroll
                for (int k = 0; k < TSQR_NUM_DATA_ROW; k++) {
                    int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
                    if (row_idx >= cols && row_idx < block_size) {
                        shared_A[row_idx + opCols * ldsa] -= utx * q[k];
                    }
                }
            }
        }
    }

    __syncthreads();

    int rRowDataNum = (n + (TSQR_BLOCK_DIM_X - 1)) / TSQR_BLOCK_DIM_X;
    for (int h = 0; h < num_data_col; h++) {
        int opCols = thread_idx_y + h * TSQR_BLOCK_DIM_Y;
        if (opCols >= n) continue;

#pragma unroll
        for (int k = 0; k < rRowDataNum; k++) {
            int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
            if (row_idx < opCols && row_idx < n) {
                R[row_idx + opCols * ldr] = shared_A[row_idx + opCols * ldsa];
                shared_A[row_idx + opCols * ldsa] = 0.0;
            }
            if (row_idx > opCols && row_idx < n) {
                R[row_idx + opCols * ldr] = 0.0;
            }
        }
    }

    for (int h = 0; h < num_data_col; h++) {
        int opCols = thread_idx_y + h * TSQR_BLOCK_DIM_Y;
        if (opCols >= n) continue;

#pragma unroll
        for (int k = 0; k < TSQR_NUM_DATA_ROW; k++) {
            int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
            q[k] = (row_idx == opCols) ? 1.0 : 0.0;
        }
        __syncwarp();

        for (int cols = n - 1; cols >= 0; cols--) {
            if (opCols >= cols) {
                T nu = 0.0;
#pragma unroll
                for (int k = 0; k < TSQR_NUM_DATA_ROW; k++) {
                    acc[k] = 0.0;
                    int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
                    if (row_idx < block_size) {
                        acc[k] = shared_A[row_idx + cols * ldsa] * q[k];
                        nu += acc[k];
                    }
                }
                T utq = warp_all_reduce_sum(nu);

#pragma unroll
                for (int k = 0; k < TSQR_NUM_DATA_ROW; k++) {
                    int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
                    if (row_idx < block_size) {
                        q[k] -= utq * shared_A[row_idx + cols * ldsa];
                    }
                }
                __syncwarp();
            }
        }

#pragma unroll
        for (int k = 0; k < TSQR_NUM_DATA_ROW; k++) {
            int row_idx = thread_idx_x + k * TSQR_BLOCK_DIM_X;
            if (row_idx < block_size) {
                A[row_idx + opCols * lda] = q[k];
            }
        }
    }
}

template <typename T>
void tsqr_recursive(cublasHandle_t cublas_handle, cudaDataType_t cuda_data_type,
                    cublasComputeType_t cublas_compute_type, int m, int n, T* A,
                    int lda, T* R, int ldr, T* work_pool) {
    dim3 blockDim(TSQR_BLOCK_DIM_X, TSQR_BLOCK_DIM_Y);
    int share_memory_size = TSQR_BLOCK_SIZE * n * sizeof(T);

    if (m <= TSQR_BLOCK_SIZE) {
        tsqr_kernel<T>
            <<<1, blockDim, share_memory_size>>>(m, n, A, lda, R, ldr);
        cudaDeviceSynchronize();
        return;
    }

    int blockNum = (m + TSQR_BLOCK_SIZE - 1) / TSQR_BLOCK_SIZE;
    int ldwork = blockNum * n;

    tsqr_kernel<T><<<blockNum, blockDim, share_memory_size>>>(
        m, n, A, lda, work_pool, ldwork);

    tsqr_recursive<T>(cublas_handle, cuda_data_type, cublas_compute_type,
                      ldwork, n, work_pool, ldwork, R, ldr,
                      work_pool + ldwork * n);

    T tone = 1.0, tzero = 0.0;
    cublasGemmStridedBatchedEx(
        cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, TSQR_BLOCK_SIZE, n, n, &tone,
        A, cuda_data_type, lda, (long long)TSQR_BLOCK_SIZE, work_pool,
        cuda_data_type, ldwork, (long long)n, &tzero, A, cuda_data_type, lda,
        (long long)TSQR_BLOCK_SIZE, m / TSQR_BLOCK_SIZE, cublas_compute_type,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    int mm = m % TSQR_BLOCK_SIZE;
    if (0 < mm) {
        cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, mm, n, n, &tone,
                     A + (m - mm), cuda_data_type, lda,
                     work_pool + (m / TSQR_BLOCK_SIZE * n), cuda_data_type,
                     ldwork, &tzero, A + (m - mm), cuda_data_type, lda,
                     cublas_compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
}

}  // namespace internal

template <typename T>
void tsqr(const CublasHandle& handle, size_t m, size_t n,
          thrust::device_vector<T>& A_inout, thrust::device_vector<T>& R) {
    assert(m >= n);
    assert(A_inout.size() >= m * n);
    assert(R.size() >= n * n);

    static_assert(internal::TSQR_BLOCK_SIZE % internal::TSQR_BLOCK_DIM_X == 0,
                  "TSQR_BLOCK_SIZE must be a multiple of TSQR_BLOCK_DIM_X");
    static_assert(internal::TSQR_BLOCK_DIM_X * internal::TSQR_NUM_DATA_ROW ==
                      internal::TSQR_BLOCK_SIZE,
                  "TSQR_NUM_DATA_ROW definition is incorrect");

    cudaDataType_t cuda_data_type;
    cublasComputeType_t cublas_compute_type;

    if (std::is_same<T, double>::value) {
        cuda_data_type = CUDA_R_64F;
        cublas_compute_type = CUBLAS_COMPUTE_64F;
    } else if (std::is_same<T, float>::value) {
        cuda_data_type = CUDA_R_32F;
        cublas_compute_type = CUBLAS_COMPUTE_32F;
    } else if (std::is_same<T, half>::value) {
        cuda_data_type = CUDA_R_16F;
        cublas_compute_type = CUBLAS_COMPUTE_16F;
    } else {
        // Unsupported type
        return;
    }

    size_t work_size = 0;
    size_t current_m = m;
    while (current_m > internal::TSQR_BLOCK_SIZE) {
        size_t block_num = (current_m + internal::TSQR_BLOCK_SIZE - 1) /
                           internal::TSQR_BLOCK_SIZE;
        size_t next_m = block_num * n;
        if (next_m == 0) break;
        work_size += next_m * n;
        current_m = next_m;
    }

    thrust::device_vector<T> work;
    if (work_size > 0) {
        work.resize(work_size);
    }

    T* A_ptr = thrust::raw_pointer_cast(A_inout.data());
    T* R_ptr = thrust::raw_pointer_cast(R.data());
    T* work_ptr = thrust::raw_pointer_cast(work.data());

    int lda = m;
    int ldr = n;

    int share_memory_size = internal::TSQR_BLOCK_SIZE * n * sizeof(T);
    cudaFuncSetAttribute(internal::tsqr_kernel<T>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         share_memory_size);

    internal::tsqr_recursive<T>(handle, cuda_data_type, cublas_compute_type, m,
                                n, A_ptr, lda, R_ptr, ldr, work_ptr);
}

// Template explicit instantiations
template void tsqr<float>(const CublasHandle& handle, size_t m, size_t n,
                          thrust::device_vector<float>& A_inout,
                          thrust::device_vector<float>& R);

template void tsqr<double>(const CublasHandle& handle, size_t m, size_t n,
                           thrust::device_vector<double>& A_inout,
                           thrust::device_vector<double>& R);

}  // namespace matrix_ops
