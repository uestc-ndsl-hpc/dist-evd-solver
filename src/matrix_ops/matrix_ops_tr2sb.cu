#include <thrust/device_vector.h>

#include <cstring>

#include "matrix_ops.cuh"

namespace matrix_ops {
namespace tr2sb {

template <typename T>
static __inline__ __device__ T warpAllReduceSumV2(T val, int ThreadCount = 32) {
    for (int mask = ThreadCount / 2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

template <typename T>
__global__ void BC_kernel_computerQ_1Col_V8_10(int n, int perBlockN,
                                               int largeBlockNum,
                                               int sweepCount,
                                               int lastSweepUCount,
                                               int sweepIndex, T* dCU,
                                               // long countU,
                                               T* dQ, long ldQ) {
    // extern __shared__ T externSM[];
    // // 也用于存放u
    // T* sU2 = externSM;
    // extern __shared__ char shared_mem[];

    extern __shared__ char shared_mem[];
    T* sU2 = reinterpret_cast<T*>(shared_mem);

    // extern __shared__ T sU2[];

    __shared__ T stailQ[MAX_WARP_COUNT * U_COL_EXRTERN_COUNT];

    __shared__ T stailQW[MAX_WARP_COUNT * U_COL_EXRTERN_COUNT];

    __shared__ T sTData[MAX_WARP_COUNT * 32];

    T rQ[U_COUNT];  // 使用寄存器替换shared memory来存储数据

    // double rU[U_COUNT];
    using RQ4Type = typename std::conditional_t<std::is_same_v<T, double>,
                                                double4*, float4*>;
    RQ4Type rQ4 = reinterpret_cast<RQ4Type>(rQ);

    int bInx = blockIdx.x;
    if (bInx < largeBlockNum) {
        perBlockN += 1;
        dQ = dQ + bInx * perBlockN * ldQ;
    } else {
        dQ = dQ + (bInx * perBlockN + largeBlockNum) * ldQ;
    }

    int i = threadIdx.x;
    int j = threadIdx.y;

    // if (0 == bInx && 0 == i && 0 == j) {
    //     printf(
    //         "begin BInx = %d,sweepIndex =%d, sU2=%p, shared_mem = %p, "
    //         "shared_mem[131071] = %c. \n",
    //         bInx, sweepIndex, sU2, shared_mem, shared_mem[131071]);
    // }

    // dQ = dQ + bInx*perBlockN*ldQ;

    // if(bInx == gridDim.x-1)
    // {
    //   perBlockN =  n - bInx*perBlockN;
    // }

    int totalU =
        lastSweepUCount + sweepIndex * U_LEN_PROC_1TIME;  // 每次处理1列中的8个u

    long sweepBaseRow = (sweepCount - sweepIndex - 1) * U_LEN_PROC_1TIME;
    long indexU = 0;

    // totalU = (U_LEN_PROC_1TIME - 2) + sweepIndex * U_LEN_PROC_1TIME;

#pragma unroll
    for (; totalU > 0;) {
        __syncthreads();

// int procUCOl = min(U_COL_EXRTERN_COUNT, totalU);

// 读取动态内存中可以存放的u
#pragma unroll
        for (int k = j; k < U_COL_EXRTERN_COUNT; k += MAX_WARP_COUNT) {
#pragma unroll
            for (int t = 0; t < U_COUNT; t++) {
                // printf(
                //     "W offset = %d, k =%d, i = %d,t=%d, U_COL_EXRTERN_COUNT "
                //     "=%d, MAX_WARP_COUNT=%d.\n",
                //     k * U_LEN_PROC_1TIME + i + t * 32, k, i, t,
                //     U_COL_EXRTERN_COUNT, MAX_WARP_COUNT);
                sU2[k * U_LEN_PROC_1TIME + i + t * 32] =
                    dCU[(indexU + k) * U_LEN_PROC_1TIME + i * U_COUNT + t];

                // sU2[k*U_LEN_PROC_1TIME+i + t *32] =
                // dCU[(indexU+k)*U_LEN_PROC_1TIME + i +t*32];
            }
        }

        __syncthreads();

        // #pragma unroll
        for (int k = j; k < perBlockN; k += MAX_WARP_COUNT) {
            // 计算Q
            // 3.2 计算u'*q
            // #pragma unroll
            // for (int t = 0; t < U_COUNT; t++)
            // {
            //   rQ[t] = dQ[k * ldQ + sweepBaseRow + i*U_COUNT + t];
            // }

            RQ4Type tmpDQ4 =
                reinterpret_cast<RQ4Type>(dQ + k * ldQ + sweepBaseRow);
#pragma unroll
            for (int t = 0; t < U_COUNT / 4; t++) {
                rQ4[t] = tmpDQ4[i * U_COUNT / 4 + t];
            }

            __syncwarp();

// 读取每行尾部多余的需要参与运算的q
#pragma unroll
            for (int t = i; t < U_COL_EXRTERN_COUNT; t += 32) {
                stailQ[j * U_COL_EXRTERN_COUNT + t] =
                    dQ[k * ldQ + sweepBaseRow + U_LEN_PROC_1TIME + t];
            }

            __syncwarp();

            int h = 0;
// 处理动态内存中的u
#pragma unroll
            for (; h < U_COL_EXRTERN_COUNT; h++) {
                // 写入最上面的Q
                if (0 != i) {
                    sTData[j * 32 + i] = rQ[0];
                } else {
                    // 将需要写入的元素先写入到共享内存中
                    stailQW[j * U_COL_EXRTERN_COUNT + h] = rQ[0];
                }

                __syncwarp();

// 进行数据的搬移
#pragma unroll
                for (int t = 0; t < U_COUNT - 1; t++) {
                    rQ[t] = rQ[t + 1];
                }

                if (31 != i) {
                    rQ[U_COUNT - 1] = sTData[j * 32 + i + 1];
                } else {
                    rQ[U_COUNT - 1] = stailQ[j * U_COL_EXRTERN_COUNT + h];
                }
                __syncwarp();

                T nux = 0.0;

#pragma unroll
                for (int t = 0; t < U_COUNT; t++) {
                    nux += sU2[h * U_LEN_PROC_1TIME + i + t * 32] * rQ[t];
                    // nux += rU[t] * rQ[t];
                }

                nux = warpAllReduceSumV2(nux, SYNC_THREAD_NUM);

#pragma unroll
                for (int t = 0; t < U_COUNT; t++) {
                    rQ[t] -= nux * sU2[h * U_LEN_PROC_1TIME + i + t * 32];
                    // rQ[t] -= nux * rU[t];
                }
            }

// 将缓存到共享内存中的Q写入到全局内存中
#pragma unroll
            for (int t = i; t < U_COL_EXRTERN_COUNT; t += 32) {
                // stailQ[j*U_COL_EXRTERN_COUNT + t] = dQ[k * ldQ +
                // sweepBaseRow + U_LEN_PROC_1TIME + t];
                dQ[k * ldQ + sweepBaseRow + t] =
                    stailQW[j * U_COL_EXRTERN_COUNT + t];
            }

            tmpDQ4 = reinterpret_cast<RQ4Type>(dQ + k * ldQ + sweepBaseRow + h);
#pragma unroll
            for (int t = 0; t < U_COUNT / 4; t++) {
                tmpDQ4[i * U_COUNT / 4 + t] = rQ4[t];
            }
        }

        indexU += U_COL_EXRTERN_COUNT;
        totalU -= U_COL_EXRTERN_COUNT;

        sweepBaseRow += U_COL_EXRTERN_COUNT;

        __syncthreads();
    }
}

}  // namespace tr2sb
}  // namespace matrix_ops

template __global__ void matrix_ops::tr2sb::BC_kernel_computerQ_1Col_V8_10<
    float>(int n, int perBlockN, int largeBlockNum, int sweepCount,
           int lastSweepUCount, int sweepIndex, float* dCU,
           // long countU,
           float* dQ, long ldQ);

template __global__ void matrix_ops::tr2sb::BC_kernel_computerQ_1Col_V8_10<
    double>(int n, int perBlockN, int largeBlockNum, int sweepCount,
            int lastSweepUCount, int sweepIndex, double* dCU,
            // long countU,
            double* dQ, long ldQ);
