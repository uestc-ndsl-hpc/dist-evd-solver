#include <thrust/device_vector.h>


// #include "gpu_handle_wrappers.h"
#include "log.h"
#include "matrix_ops.cuh"
// #include "sy2sb_panelqr.cuh"

#include <cooperative_groups.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <algorithm>

namespace matrix_ops {
namespace sb2tr {

namespace cg = cooperative_groups;

// #include "fileOpTool.h"
template <typename T>
static __inline__ __device__ T warpAllReduceSum(T val) {
    for (int mask = warpSize / 2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

__device__ bool g_cycleFlag = true;    // 全局循环标志,用于判断是否需要继续处理
__device__ int g_headSweepIndex = 0;   // 当前PW的最开始趟BC的索引
__device__ int g_tailSweepIndex = -1;  // 当前PW的最后1趟BC的索引

#define MY_DEBUG 0

#define MY_DEBUG_NO_COM 0

__device__ int DEBUG_BLOCK = 0;

template <int BandWidth, typename T>
// __device__ void chasing_kernel_one_timeV7(int n, int b, double *dSubA, int
// ldSubA, int startRow)
__global__ void kernel_BC_NVShmem_V3(int n, int b, int ns, T* dSubA, int ldSubA,
                                     T* dU, int ldU, int blockNum, int PEIndex,
                                     int PENum, int* com, int* prePEWriteCom,
                                     //  int *nextPEWriteTailSweepProcRow,
                                     //  int * g_cycleFlag
                                     int* nextPEWriteTailSweepProcRow) {
    auto grid = cg::this_grid();
    // auto block = cg::this_thread_block();

    int Nx = blockDim.x;
    int Ny = blockDim.y;

    // 内部使用,不进行判断
    if (BandWidth != b) {
        return;
    }

    int warpGroupThdCount = Ny / 2;

    int bInx = grid.block_rank();

    int i = threadIdx.x;
    int j = threadIdx.y;

    __shared__ T u[BandWidth];

    __shared__ T
        S1[BandWidth * BandWidth];  // 共享内存,用于存放[B1,S,B2]进行变换
    __shared__ T
        S2[BandWidth * BandWidth];  // 共享内存,用于存放[B1,S,B2]进行变换
    int ldSS = b;

    // 当前PE等待的最开始趟BC的索引
    int cuProcSweepIndex = bInx;  // 当前block处理的BC Sweep的索引

    // 计算当前PE的起始位置和结束位置
    int startCol = PEIndex * ns;
    int endCol = (PEIndex + 1) * ns;

    // 初始化处理状态: 0 -- 处理前面PE的BC Sweep; 1 -- 处理当前PE的BC Sweep
    int procState = 0;

    if (0 == PEIndex) {
        // 第1个PE只有procState=1的情况,因为没有前面的PE需要等待
        procState = 1;
    }

    bool waitFlag =
        true;  // 等待标志, 用于判断当前PE是否需要等待前面PE的BC Sweep完成;
               // 开始的时候需要等待前面的PE完成相应的BC Sweep.

    int k;             // 迭代次数,用于判断是否需要进行远程数据的处理
    int k1, k2;        // 用于记录当前PE处理的起始k和结束k
    int crossPECount;  // 用于记录当前PE处理的过程中,有多少次会涉及到nextPE的数据

    int opRow;

    T* B1;  // 用于存放B1的起始位置, B1的起始位置为(opRow, opRow-b)
    T* S;   // 用于存放S的起始位置, S的起始位置为(opRow, opRow)
    // T *B2; // 用于存放B2的起始位置, B2的起始位置为(opRow+b, opRow)
    T* uB;  // 用于存放u向量的起始位置, u向量的起始位置为(opRow, 0)

    int rowB1, colB1;  // 用于记录B1的行数和列数
    int rowS, colS;    // 用于记录S的行数和列数
    // int rowB2, colB2; // 用于记录B2的行数和列数

    int endBCSweepIndex = endCol - 1;
    if (PENum - 1 == PEIndex) {
        endBCSweepIndex = n - 3;
    }

    // endBCSweepIndex = 0;

    __syncthreads();

    while (true == g_cycleFlag)
    // while(0 == nvshmem_int_g(g_cycleFlag, 0))
    {
        if (true == waitFlag && procState < 2) {
            if (0 == procState) {
                if (1 == prePEWriteCom[cuProcSweepIndex]) {
                    waitFlag =
                        false;  // 重置waitFlag, 只有waitSweepIndex趟BC
                                // Sweep完成后,
                                // 才会等待新的waitSweepIndex+ns趟BC Sweep
                    colS = b;   // 当procState=0时, 初始时S的列数为b

                    // k1 = (startCol - (cuProcSweepIndex + 1-1) + (b-1)) / b;
                    // k2 = (endCol - (cuProcSweepIndex + 1-1) + (b-1)) / b;

                    // 直接按列进行计算;但是注意第1个B1的宽度只有1
                    k1 = (startCol - (cuProcSweepIndex + 1) + (b - 1)) / b + 1;
                    k2 = (endCol - (cuProcSweepIndex + 1) + (b - 1)) / b + 1;
                }
            } else {
                waitFlag =
                    false;  // 重置waitFlag, 只有waitSweepIndex趟BC Sweep完成后,
                            // 才会等待新的waitSweepIndex+ns趟BC Sweep
                colS = 1;   // 当procState=1时, 初始时S的列数为1

                k1 = 0;
                // k2 = (endCol - (cuProcSweepIndex + 1-1) + (b-1)) / b;

                k2 = (endCol - (cuProcSweepIndex + 1) + (b - 1)) / b + 1;
            }

            if (false == waitFlag) {
                // 最后1个PE的最后2个BC Sweep不需要等处理
                if (cuProcSweepIndex > endBCSweepIndex) {
                    k = k2 = 0;
                    crossPECount = 0;
                } else {
                    // 获取起始位置
                    // 这个使用waitSweepIndex来进行计算
                    // 因为我们是按列进行存储的,所以我们需要计算B1的起始列数,我们以S为定位标记;
                    // 第1个S的起始位置为(cuProcSweepIndex+1,
                    // cuProcSweepIndex+1);只要不是最后1个PE,S都是以(b,b)的尺寸增加;
                    // 同时在procState=0的情况下, B1的尺寸应该也是(b,b),
                    // 所以B1的起始位置为(cuProcSweepIndex+1+k1*b,
                    // cuProcSweepIndex+1+k1*b-b)
                    // 所以其处理waitSweepIndex趟的起始列为waitSweepIndex+1+k1*b-b刚好大于等于startCol;
                    // 其处理waitSweepIndex趟的结束类为waitSweepIndex+1+k2*b-b刚好大于等于endCol
                    // (处理的时候不包含k2).

                    // 计算出waitSweepIndex趟本PE处理的起始k1
                    // 这种计算方法也是正确的,只是理解起来比较复杂
                    // {
                    // colB1 = colS;
                    // k1 = (startCol - (cuProcSweepIndex + 1-1) + (b-1)) / b;
                    // k1 = max(0, k1); // 确保k1不小于0,因为k1是从0开始的

                    // // 计算出waitSweepIndex趟本PE处理的结束k2
                    // k2 = (endCol - (cuProcSweepIndex + 1-1) + (b-1)) / b;
                    // }

                    // 也就是cuProcSweepIndex小于startCol的时候
                    // if(0 == procState){
                    //   k1 = (startCol - (cuProcSweepIndex + 1-1) + (b-1)) / b;
                    //   k2 = (endCol - (cuProcSweepIndex + 1-1) + (b-1)) / b;

                    // }else{
                    //   // cuProcSweepIndex大于等于startCol的时候
                    //   k1 = 0;
                    //   k2 = (endCol - (cuProcSweepIndex + 1-1) + (b-1)) / b;
                    // }

                    opRow = cuProcSweepIndex + 1 + k1 * b;

                    // 计算出有最后的几次处理会涉及到nextPE的数据
                    // 在proState=0的情况下, 每次处理影响的列数为2b,
                    // 所以有1次或者2次涉及nextPE的数据的情况
                    crossPECount =
                        1;  // 默认情况下,每次处理都会涉及到nextPE的数据
                    if (((k2 - k1) > 1) &&
                        cuProcSweepIndex + 1 + (k2 - 2) * b + 2 * b > endCol) {
                        // 说明倒数第2次也会涉及到nextPE的数据
                        crossPECount = 2;
                    }

                    if (PEIndex == PENum - 1) {
                        // 最后1个PE不需要涉及nextPE的数据
                        // 但其最后1次的处理有所不同
                        crossPECount = 0;
                    }

                    // firstFlag = true; // 第1次进入
                    k = k1;
                }

                // if(0==i && 0==j)
                // {
                //   printf("line= %d,BInx: %d, PEIndex: %d, PENum: %d,
                //   cuProcSweepIndex: %d, startCol: %ld, endCol: %ld, k1=%d,
                //   k2=%d, crossPECount = %d\n",
                //         __LINE__, bInx, PEIndex, PENum, cuProcSweepIndex,
                //         startCol, endCol, k1, k2, crossPECount);
                // }
            }
        }

        __syncthreads();

        if (false == waitFlag) {
            // 处理前面不需要nextPE数据的情况
            // 还需要处理最后1个PE的所有情况
            if (k < k2 - crossPECount) {
                if ((0 == cuProcSweepIndex) ||
                    // (g_headSweepIndex == cuProcSweepIndex && opRow + 2*b <
                    // nvshmem_int_g(nextPEWriteTailSweepProcRow, PEIndex)) ||
                    ((PEIndex < PENum - 1) &&
                     g_headSweepIndex == cuProcSweepIndex &&
                     opRow + 2 * b < *nextPEWriteTailSweepProcRow) ||
                    (g_headSweepIndex != cuProcSweepIndex &&
                     opRow + 2 * b < com[cuProcSweepIndex - 1]))
                // (opRow + 2 * b < com[cuProcSweepIndex-1]))
                {
#if MY_DEBUG_NO_COM
                    if (0 == i && 0 == j) {
                        printf(
                            "[Begin] line= %d,BInx: %d, PEIndex: %d, PENum: "
                            "%d, cuProcSweepIndex: %d, startCol: %d, endCol: "
                            "%d, k1=%d, k2=%d, k=%d, opRow: %d\n",
                            __LINE__, bInx, PEIndex, PENum, cuProcSweepIndex,
                            startCol, endCol, k1, k2, k, opRow);
                    }
                    __syncthreads();
#endif

                    rowB1 = min(
                        b,
                        (int)(n -
                              opRow));  // 因为要处理B1矩阵的起始位置为(opRow,
                                        // opRow-b)
                    colB1 =
                        colS;  // 只要不是本PE发起的BC Sweep,那么B1的列数就是b

                    // 计算B1的位置, 我们以S的位置为准, B1的位置为(opRow,
                    // opRow-b) B1 = dSubA + b + (opRow - b - startCol) *
                    // ldSubA;
                    B1 = dSubA + colB1 + (opRow - colB1 - startCol) * ldSubA;

                    // 如果初次进来,需要加载B1到S1中
                    if (k1 == k) {
                        if (0 == i && 0 == j) {
                            g_tailSweepIndex++;
                        }

                        for (int opColB1 = j; opColB1 < colB1; opColB1 += Ny) {
                            for (int opRowB1 = i; opRowB1 < rowB1;
                                 opRowB1 += Nx) {
                                // 由于是阶梯状,所以B1每列会多减去列数
                                S1[opRowB1 + opColB1 * ldSS] =
                                    B1[(opRowB1 - opColB1) + opColB1 * ldSubA];
                            }
                        }

                        // 最开始的时候S没有数据
                        rowS = 0;
                        colS = 0;
                    }

                    // __syncthreads();

                    // if(0==i && 0==j)
                    // {
                    //   printf("line= %d,BInx: %d, PEIndex: %d, PENum: %d,
                    //   cuProcSweepIndex: %d, startCol: %ld, endCol: %ld,
                    //   k1=%d, k2=%d, k=%d\n",
                    //         __LINE__, bInx, PEIndex, PENum, cuProcSweepIndex,
                    //         startCol, endCol, k1, k2, k);
                    // }

                    __syncthreads();

#if MY_DEBUG_NO_COM
                    if ((31 == i) && (31 == j)) {
                        printf(
                            "line= %d,BInx: %d, PEIndex: %d, PENum: %d, "
                            "cuProcSweepIndex: %d, startCol: %d, endCol: %d, "
                            "k1=%d, k2=%d, k=%d,"
                            "rowB1: %d, colB1: %d, opRow: %d\n",
                            __LINE__, bInx, PEIndex, PENum, cuProcSweepIndex,
                            startCol, endCol, k1, k2, k, rowB1, colB1, opRow);
                        for (int h = 0; h < colB1; h++) {
                            for (int k = 0; k < rowB1; k++) {
                                printf("B1[%d][%d] = %f,", k, h,
                                       S1[k + h * ldSS]);
                            }
                            printf("\n");
                        }
                        // printf("\b");
                    }
                    __syncthreads();
#endif

                    if (0 != j) {
                        // 第1组warp: 对S2进行处理--从S2写回preS,取新的S到S2中
                        // 想把写回和读入编写到1个循环中,但是preS和S的形状大小可能不一样(特指第1和最后1个),所以分开实现

                        // 注意在写回S的时候我们使用对称性,只写回去下三角部分的元素

                        for (int opColS = j - 1; opColS < colS;
                             opColS += (Ny - 1)) {
                            for (int opRowS = i;
                                 (opColS <= opRowS) && (opRowS < rowS);
                                 opRowS += Nx) {
                                // pB2[opRowB2][opColB2] = B1[opColB2][opRowB2]
                                // = SS[opColB2][opRowB2]
                                // 由于是阶梯状,所以S每列会多减去列数
                                S[(opRowS - opColS) + opColS * ldSubA] =
                                    S2[opRowS + opColS * ldSS];
                            }
                        }

                        // 更新新的S, S的起始位置为(opRow, opRow)
                        colS = rowS = rowB1;
                        S = dSubA + (opRow - startCol) * ldSubA;

                        // 利用对称性,只拷贝下三角部分的元素
                        for (int opColS = j - 1; opColS < colS;
                             opColS += (Ny - 1)) {
                            for (int opRowS = i;
                                 (opColS <= opRowS) && (opRowS < rowS);
                                 opRowS += Nx) {
                                S2[opRowS + opColS * ldSS] =
                                    S[(opRowS - opColS) + opColS * ldSubA];

                                // 利用对称性
                                S2[opColS + opRowS * ldSS] =
                                    S2[opRowS + opColS * ldSS];
                            }
                        }
                    } else {
                        // 第2组warp:
                        // 对S1进行处理--取B1第1列到U中;求Householder向量;对B1进行Householder变换
                        // 2.1将B1中第一列的数据拷贝到u中
                        for (int opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx) {
                            //  u[opRowB1] = B1[opRowB1][0]
                            u[opRowB1] = S1[opRowB1];
                        }

                        __syncwarp();

                        // 2.2 求出norm_x
                        T nu = 0.0;
                        // nu = 0.0;

                        for (int opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx) {
                            //  u[opRowB1] = B1[opRowB1][0]
                            nu += u[opRowB1] * u[opRowB1];
                        }

                        // 需要将1个lane中所有线程求出的norm_squre加到一起,同时进行同步
                        T norm_x_squre = warpAllReduceSum(nu);
                        T norm_x = sqrt(norm_x_squre);

                        // 2.3、求u=x/norm(x);
                        T scale = 1.0 / norm_x;
                        for (int opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx) {
                            //  u[opRowB1] = B1[opRowB1][0]
                            u[opRowB1] *= scale;
                        }

                        __syncwarp();

                        // 2.4、求u(0)= u(0)+sign(u(0));
                        // 每列找一个线程来计算即可
                        if (0 == i) {
                            T u1 = u[0];

                            u[0] += (u1 >= 0) ? 1 : -1;

                            // 把normx存放到RR中，也就是对角线的元素
                            // 使用这个值可以少进行一步计算,暂时没考虑,后期考虑
                            // T RR = (u1 >= 0) ? -norm_x : norm_x;
                        }

                        __syncwarp();

                        // 2.5、u=u/sqrt(abs(u(0))),计算HouseHolder向量
                        scale = 1 / (sqrt(abs(u[0])));

                        for (int opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx) {
                            //  u[opRowB1] = B1[opRowB1][0]
                            u[opRowB1] *= scale;
                        }

                        // 将求出的u向量放置到uB中
                        uB = dU + cuProcSweepIndex *
                                      ldU;  // uB的起始位置为(opRow, 0)
                        for (int opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx) {
                            //  u[opRowB1] = B1[opRowB1][0]
                            uB[opRow - startCol + opRowB1] = u[opRowB1];
                        }

                        // 更新新的S -- 这些warp也需要更新
                        colS = rowS = rowB1;
                        S = dSubA + (opRow - startCol) * ldSubA;
                    }

                    __syncthreads();

#if MY_DEBUG_NO_COM
                    // #if 1
                    if ((31 == i) && (31 == j)) {
                        printf(
                            "line= %d,BInx: %d, PEIndex: %d, PENum: %d, "
                            "cuProcSweepIndex: %d, opRow: %d.\n",
                            __LINE__, bInx, PEIndex, PENum, cuProcSweepIndex,
                            opRow);
                        for (int k = 0; k < rowB1; k++) {
                            printf("u[%d] = %f,", k, u[k]);
                        }
                        printf("\n");
                    }

                    __syncthreads();
#endif

#if MY_DEBUG_NO_COM
                    // if ((DEBUG_BLOCK == bInx) && (0 == i) && (0 == j))
                    if ((31 == i) && (31 == j)) {
                        // printf("block[%d] [%d][%d] come line=%d.\n",
                        //       bInx, i, j, __LINE__);
                        printf(
                            "line= %d,BInx: %d, PEIndex: %d, PENum: %d, "
                            "cuProcSweepIndex: %d, startCol: %d, endCol: %d, "
                            "k1=%d, k2=%d, k=%d,"
                            "rowS: %d, colS: %d, opRow: %d\n",
                            __LINE__, bInx, PEIndex, PENum, cuProcSweepIndex,
                            startCol, endCol, k1, k2, k, rowS, colS, opRow);

                        for (int h = 0; h < colS; h++) {
                            for (int k = 0; k < rowS; k++) {
                                printf("S[%d][%d]=%f,", k, h, S2[k + h * ldSS]);
                            }
                            printf("\n");
                        }
                        // printf("\b");
                    }
                    __syncthreads();
#endif

                    // 3.1.2 一起对B1进行Householder变换
                    for (int opColB1 = j; opColB1 < colB1; opColB1 += Ny) {
                        T nu = 0.0;
                        // 先计算u'x
                        for (int opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx) {
                            nu += u[opRowB1] * S1[opRowB1 + opColB1 * ldSS];
                        }

                        T utx = warpAllReduceSum(nu);

                        // 计算x-uu'x
                        for (int opRowB1 = i; opRowB1 < rowB1; opRowB1 += Nx) {
                            S1[opRowB1 + opColB1 * ldSS] -= utx * u[opRowB1];
                        }

                        __syncwarp();
                    }

                    __syncthreads();

#if MY_DEBUG_NO_COM
                    if ((31 == i) && (31 == j)) {
                        printf(
                            "line= %d,BInx: %d, PEIndex: %d, PENum: %d, "
                            "cuProcSweepIndex: %d, startCol: %d, endCol: %d, "
                            "k1=%d, k2=%d, k=%d,"
                            "rowB1: %d, colB1: %d, opRow: %d\n",
                            __LINE__, bInx, PEIndex, PENum, cuProcSweepIndex,
                            startCol, endCol, k1, k2, k, rowB1, colB1, opRow);
                        for (int h = 0; h < colB1; h++) {
                            for (int k = 0; k < rowB1; k++) {
                                printf("B1[%d][%d] = %f,", k, h,
                                       S1[k + h * ldSS]);
                            }
                            printf("\n");
                        }
                        // printf("\b");
                    }
                    __syncthreads();
#endif

                    // 3.2 第1组warp:
                    // 对S1进行处理--写回B1(包括B1和其转置位置),取B2到S1中 注意:
                    // 取B2前都要判断数据同步条件--其实就是在B2变化的时候进行修改和判断同步条件
                    // 3.2 第2组warp: 对S2进行处理--对S进行Householder变换
                    if (j < warpGroupThdCount) {
                        // 第1组warp:
                        // 对S2进行处理--写回B1(包括B1和其转置位置),取B2到S1中

                        for (int opColB1 = j; opColB1 < colB1;
                             opColB1 += warpGroupThdCount) {
                            for (int opRowB1 = i; opRowB1 < rowB1;
                                 opRowB1 += Nx) {
                                B1[(opRowB1 - opColB1) + opColB1 * ldSubA] =
                                    S1[opRowB1 + opColB1 * ldSS];

                                // 写出B1转置
                                // B1_T[opColB1 + opRowB1 * ldS] = S1[opRowB1 +
                                // opColB1 * ldSS];
                            }
                        }

                        opRow += rowB1;  // 更新下一次A的长度
                        rowB1 = min(
                            b,
                            (int)(n - opRow));  // 因为opRow是起始位置,所以是n-1
                                                // - opRow +1 = n - opRow;
                        colB1 = colS;
                        // B1    = dSubA + colB1 + (opRow - colB1) * ldSubA;
                        B1 =
                            dSubA + colB1 + (opRow - colB1 - startCol) * ldSubA;

                        // 将同步条件写进去--不能写入同步条件,因为要保证数据一致性
                        // 这儿不判断退出--也是因为数据同步,其他的warp可能还在处理

                        // 取B2到S1中
                        for (int opColB1 = j; opColB1 < colB1;
                             opColB1 += warpGroupThdCount) {
                            for (int opRowB1 = i; opRowB1 < rowB1;
                                 opRowB1 += Nx) {
                                // SS[opRowB1][opColB1] = B1[opRowB1][opColB1] =
                                // B2[opColB1][opRowB1]
                                S1[opRowB1 + opColB1 * ldSS] =
                                    B1[(opRowB1 - opColB1) + opColB1 * ldSubA];
                            }
                        }
                    } else {
                        // 第2组warp: 对S2进行处理--对S进行Householder变换

                        for (int opColS = j - warpGroupThdCount; opColS < colS;
                             opColS += warpGroupThdCount) {
                            T nu = 0.0;
                            // 先计算u'x

                            for (int opRowS = i; opRowS < rowS; opRowS += Nx) {
                                nu += u[opRowS] * S2[opRowS + opColS * ldSS];
                            }

                            T utx = warpAllReduceSum(nu);

                            // 计算x-uu'x
                            for (int opRowS = i; opRowS < rowS; opRowS += Nx) {
                                S2[opRowS + opColS * ldSS] -= utx * u[opRowS];
                            }

                            __syncwarp();
                        }

                        // 这里面的线程也需要更新这些局部变量
                        opRow += rowB1;  // 更新下一次A的长度
                        rowB1 = min(
                            b,
                            (int)(n - opRow));  // 因为opRow是起始位置,所以是n-1
                                                // - opRow +1 = n - opRow;
                        colB1 = colS;
                        // B1    = dSubA + colB1 + (opRow - colB1) * ldSubA;

                        B1 =
                            dSubA + colB1 + (opRow - colB1 - startCol) * ldSubA;
                    }

                    __syncthreads();

                    // 3.3 两组warp一起对S2的转置进行进行Householder变换
                    for (int opRowS = j; opRowS < rowS; opRowS += Ny) {
                        T nu = 0.0;
                        // 先计算u'x

                        for (int opColS = i; opColS < colS; opColS += Nx) {
                            nu += u[opColS] * S2[opRowS + opColS * ldSS];
                        }

                        T utx = warpAllReduceSum(nu);

                        // 计算x-uu'x
                        for (int opColS = i; opColS < colS; opColS += Nx) {
                            S2[opRowS + opColS * ldSS] -= utx * u[opColS];
                        }

                        __syncwarp();
                    }

                    __syncthreads();

#if MY_DEBUG_NO_COM
                    // if ((DEBUG_BLOCK == bInx) && (0 == i) && (0 == j))
                    if ((31 == i) && (31 == j)) {
                        // printf("block[%d] [%d][%d] come line=%d.\n",
                        //       bInx, i, j, __LINE__);
                        printf(
                            "line= %d,BInx: %d, PEIndex: %d, PENum: %d, "
                            "cuProcSweepIndex: %d, startCol: %d, endCol: %d, "
                            "k1=%d, k2=%d, k=%d,"
                            "rowS: %d, colS: %d, opRow: %d\n",
                            __LINE__, bInx, PEIndex, PENum, cuProcSweepIndex,
                            startCol, endCol, k1, k2, k, rowS, colS, opRow);

                        for (int h = 0; h < colS; h++) {
                            for (int k = 0; k < rowS; k++) {
                                printf("S[%d][%d]=%f,", k, h, S2[k + h * ldSS]);
                            }
                            printf("\n");
                        }
                        // printf("\b");
                    }
                    __syncthreads();
#endif

                    // 3.3 两组warp一起对B2的转置进行Householder变换
                    for (int opRowB1 = j; opRowB1 < rowB1; opRowB1 += Ny) {
                        T nu = 0.0;
                        // 先计算u'x
                        for (int opColB1 = i; opColB1 < colB1; opColB1 += Nx) {
                            nu += u[opColB1] * S1[opRowB1 + opColB1 * ldSS];
                        }

                        T utx = warpAllReduceSum(nu);

                        // 计算x-uu'x
                        for (int opColB1 = i; opColB1 < colB1; opColB1 += Nx) {
                            S1[opRowB1 + opColB1 * ldSS] -= utx * u[opColB1];
                        }

                        __syncwarp();
                    }

                    __syncthreads();

                    if (0 == i && 0 == j) {
                        com[cuProcSweepIndex] =
                            opRow;  // 更新当前PE当前block的处理位置
                    }

                    __syncthreads();

                    if (0 < PEIndex && cuProcSweepIndex == g_tailSweepIndex) {
                        if (0 == i && 0 == j) {
                            nvshmem_int_p(nextPEWriteTailSweepProcRow, opRow,
                                          PEIndex - 1);

                            nvshmem_quiet();
                        }
                    }

                    k++;

                    // 需要写入最后1次的S和B2
                    if (k2 - crossPECount == k) {
                        // 写入S的数据
                        for (int opColS = j; opColS < colS; opColS += Ny) {
                            for (int opRowS = i;
                                 (opColS <= opRowS) && (opRowS < rowS);
                                 opRowS += Nx) {
                                // pB2[opRowB2][opColB2] = B1[opColB2][opRowB2]
                                // = SS[opColB2][opRowB2]
                                // 由于是阶梯状,所以S每列会多减去列数
                                S[(opRowS - opColS) + opColS * ldSubA] =
                                    S2[opRowS + opColS * ldSS];
                            }
                        }
                        __syncthreads();

                        // 写入B2的数据
                        for (int opColB1 = j; opColB1 < colB1; opColB1 += Ny) {
                            for (int opRowB1 = i; opRowB1 < rowB1;
                                 opRowB1 += Nx) {
                                B1[(opRowB1 - opColB1) + opColB1 * ldSubA] =
                                    S1[opRowB1 + opColB1 * ldSS];

                                // 写出B2转置
                                // B2_T[opColB1 + opRowB1 * ldS] = S1[opRowB1 +
                                // opColB1 * ldSS];
                            }
                        }
                        __syncthreads();
                    }

                    __syncthreads();
                }
            } else if (k < k2) {
                // 处理最后1次需要remote数据的情况
                // 只有非最后1个PE需要处理
                // if( PEIndex != PENum - 1)  //根据上面的处理,不存在PEIndex ==
                // PENum - 1
                {
                    if ((0 == cuProcSweepIndex) ||
                        (g_headSweepIndex == cuProcSweepIndex &&
                         opRow + 2 * b < *nextPEWriteTailSweepProcRow) ||
                        (g_headSweepIndex != cuProcSweepIndex &&
                         opRow + 2 * b < com[(cuProcSweepIndex - 1)]))
                    // (opRow + 2 * b < com[(cuProcSweepIndex-1)]))
                    {
// #if MY_DEBUG
#if 0
            if(0==i && 0==j)
            {
              printf("[Begin] line= %d,BInx: %d, PEIndex: %d, PENum: %d, cuProcSweepIndex: %d, startCol: %d, endCol: %d, k1=%d, k2=%d, k=%d, opRow: %d\n", 
                    __LINE__, bInx, PEIndex, PENum, cuProcSweepIndex, startCol, endCol, k1, k2, k, opRow);
            }
            __syncthreads();
#endif

                        rowB1 = min(
                            b,
                            (int)(n -
                                  opRow));  // 因为要处理B1矩阵的起始位置为(opRow,
                                            // opRow-b)
                        colB1 = colS;  // B1的列数就是上1个S的列数

                        // 计算在本PE上的B1的列数
                        // int localColB1 = min(b, int(endCol - (opRow - b)));
                        int localColB1 =
                            min(colB1, int(endCol - (opRow - colB1)));

                        // int remoteColB1 = b - localColB1;

                        // if(DEBUG_BLOCK == bInx && 0==i && 0==j)
                        // {
                        //   printf("line= %d,BInx: %d, rowB1: %d, colB1: %d,
                        //   opRow: %d, localColB1: %d\n",
                        //         __LINE__, bInx, rowB1, colB1, opRow,
                        //         localColB1);
                        // }

                        __syncthreads();

                        // 加载B1到S1中, 包括本地数据和nextPE的数据
                        // 本地B1的起始位置
                        T* localB1 =
                            dSubA + colB1 + (opRow - colB1 - startCol) * ldSubA;
                        for (int opColB1 = j; opColB1 < colB1; opColB1 += Ny) {
                            for (int opRowB1 = i; opRowB1 < rowB1;
                                 opRowB1 += Nx) {
                                if (opColB1 < localColB1) {
                                    // 由于是阶梯状,所以S每列会多减去列数
                                    S1[opRowB1 + opColB1 * ldSS] =
                                        localB1[(opRowB1 - opColB1) +
                                                opColB1 * ldSubA];
                                } else {
                                    // 远程数据,需要进行远程加载
                                    // nvshmem_double_get(&S1[opRowB1 + opColB1
                                    // * ldSS],
                                    //                     &dSubA[(opRowB1 -
                                    //                     opColB1) + colB1 +
                                    //                     (opColB1 -
                                    //                     localColB1) *
                                    //                     ldSubA], 1,
                                    //                     PEIndex+1);

                                    // S1[opRowB1 + opColB1 * ldSS] =
                                    // nvshmem_double_g(dSubA + (opRowB1 -
                                    // opColB1) + colB1 + (opColB1 - localColB1)
                                    // * ldSubA, PEIndex+1);
                                    int offset =
                                        colB1 +
                                        (opRow - colB1 - endCol) * ldSubA +
                                        (opRowB1 - opColB1) + opColB1 * ldSubA;
                                    if constexpr (std::is_same_v<T, double>) {
                                        S1[opRowB1 + opColB1 * ldSS] =
                                            nvshmem_double_g(dSubA + offset,
                                                             PEIndex + 1);
                                    } else if constexpr (std::is_same_v<
                                                             T, float>) {
                                        S1[opRowB1 + opColB1 * ldSS] =
                                            nvshmem_float_g(dSubA + offset,
                                                            PEIndex + 1);
                                    }
                                }
                            }
                        }

                        // 进行同步
                        __syncthreads();

// #if MY_DEBU
#if 0
            if ((31 == i) && (31 == j))
            {
              printf("line= %d,BInx: %d, PEIndex: %d, PENum: %d, cuProcSweepIndex: %d, startCol: %d, endCol: %d, k1=%d, k2=%d, k=%d,"
                      "rowB1: %d, colB1: %d, opRow: %d, localColB1: %d\n", 
                    __LINE__, bInx, PEIndex, PENum, cuProcSweepIndex, startCol, endCol, k1, k2, k, rowB1, colB1, opRow, localColB1);
              for(int h = 0; h < colB1; h++)
              {
                for(int k = 0; k < rowB1; k++ )
                {
                  printf("B1[%d][%d] = %f,",
                      k, h, S1[k + h*ldSS]);
                }
                printf("\n");
              }
                  // printf("\b");
            }
            __syncthreads();
#endif

                        colS = rowS = rowB1;

                        // 计算在本PE上的S的列数
                        int localColS;
                        T* localS;
                        // long remoteSOffset;

                        if (endCol > opRow) {
                            localColS = endCol - opRow;
                            localS = dSubA + (opRow - startCol) * ldSubA;
                        } else {
                            localColS = 0;
                        }

                        // 本地S的起始位置
                        // T *localS = dSubA + (opRow - startCol) * ldSubA;

                        // nextPE中S的起始位置
                        // T *remoteS = dSubA + (colB1 - localColB1) *
                        // ldSubA;

                        int remoteSOffset = (opRow - endCol) * ldSubA;
                        int offset;

                        // 3.1 第1组warp:
                        // 对S2进行处理--从S2中写回preS,取新的A到S2中 3.1
                        // 第2组warp(j==0):
                        // 对S1进行处理--取B1第1列到U中;求Householder向量
                        if (0 != j) {
                            // 读取矩阵S的数据存放到S2中

                            // 利用对称性,只拷贝下三角部分的元素
                            for (int opColS = j - 1; opColS < colS;
                                 opColS += (Ny - 1)) {
                                for (int opRowS = i;
                                     (opColS <= opRowS) && (opRowS < rowS);
                                     opRowS += Nx) {
                                    // pB2[opRowB2][opColB2] =
                                    // B1[opColB2][opRowB2] =
                                    // SS[opColB2][opRowB2]
                                    // 由于是阶梯状,所以S每列会多减去列数
                                    // S2[opColS + opRowS * ldSS] =
                                    if (opColS < localColS) {
                                        S2[opRowS + opColS * ldSS] =
                                            localS[(opRowS - opColS) +
                                                   opColS * ldSubA];
                                    } else {
                                        // 远程数据,需要进行远程加载
                                        // nvshmem_double_get(&S2[opRowS +
                                        // opColS * ldSS],
                                        //                     &remoteS[(opRowS
                                        //                     - opColS ) +
                                        //                     opColS * ldSubA],
                                        //                     1, PEIndex+1);

                                        // S2[opRowS + opColS * ldSS] =
                                        // nvshmem_double_g(remoteS + (opRowS -
                                        // opColS ) + opColS * ldSubA,
                                        // PEIndex+1);

                                        // 就从整体位置上去计算目前需要读取的位置
                                        // dSubA + (opRow-endCol)*ldSubA+
                                        // (opRowS - opColS ) + opColS * ldSubA
                                        offset = remoteSOffset +
                                                 (opRowS - opColS) +
                                                 opColS * ldSubA;

                                        if constexpr (std::is_same_v<T,
                                                                     double>) {
                                            S2[opRowS + opColS * ldSS] =
                                                nvshmem_double_g(dSubA + offset,
                                                                 PEIndex + 1);
                                        } else if constexpr (std::is_same_v<
                                                                 T, float>) {
                                            S2[opRowS + opColS * ldSS] =
                                                nvshmem_float_g(dSubA + offset,
                                                                PEIndex + 1);
                                        }
                                    }

                                    // 利用对称性
                                    S2[opColS + opRowS * ldSS] =
                                        S2[opRowS + opColS * ldSS];
                                }
                            }
                        } else {
                            // 第2组warp:
                            // 对S1进行处理--取B1第1列到U中;求Householder向量;对B1进行Householder变换
                            // 2.1将B1中第一列的数据拷贝到u中
                            for (int opRowB1 = i; opRowB1 < rowB1;
                                 opRowB1 += Nx) {
                                u[opRowB1] = S1[opRowB1];
                            }

                            __syncwarp();

                            // 2.2 求出norm_x
                            // T nu = 0.0;
                            T nu = 0.0;

                            for (int opRowB1 = i; opRowB1 < rowB1;
                                 opRowB1 += Nx) {
                                //  u[opRowB1] = B1[opRowB1][0]
                                nu += u[opRowB1] * u[opRowB1];
                            }

                            // 需要将1个lane中所有线程求出的norm_squre加到一起,同时进行同步
                            T norm_x_squre = warpAllReduceSum(nu);
                            T norm_x = sqrt(norm_x_squre);

                            // 2.3、求u=x/norm(x);
                            T scale = 1.0 / norm_x;
                            for (int opRowB1 = i; opRowB1 < rowB1;
                                 opRowB1 += Nx) {
                                //  u[opRowB1] = B1[opRowB1][0]
                                u[opRowB1] *= scale;
                            }

                            __syncwarp();

                            // 2.4、求u(0)= u(0)+sign(u(0));
                            // 每列找一个线程来计算即可
                            if (0 == i) {
                                T u1 = u[0];

                                u[0] += (u1 >= 0) ? 1 : -1;

                                // 把normx存放到RR中，也就是对角线的元素
                                // 使用这个值可以少进行一步计算,暂时没考虑,后期考虑
                                // T RR = (u1 >= 0) ? -norm_x : norm_x;
                            }

                            __syncwarp();

                            // 2.5、u=u/sqrt(abs(u(0))),计算HouseHolder向量
                            scale = 1 / (sqrt(abs(u[0])));

                            for (int opRowB1 = i; opRowB1 < rowB1;
                                 opRowB1 += Nx) {
                                //  u[opRowB1] = B1[opRowB1][0]
                                u[opRowB1] *= scale;
                            }

                            // 将求出的u向量放置到uB中
                            uB = dU + cuProcSweepIndex *
                                          ldU;  // uB的起始位置为(opRow, 0)
                            for (int opRowB1 = i; opRowB1 < rowB1;
                                 opRowB1 += Nx) {
                                //  u[opRowB1] = B1[opRowB1][0]
                                uB[opRow - startCol + opRowB1] = u[opRowB1];
                            }
                        }
                        __syncthreads();

// #if MY_DEBUG
#if 1
                        if ((31 == i) && (31 == j)) {
                            // printf("line= %d,BInx: %d, PEIndex: %d, PENum:
                            // %d, cuProcSweepIndex: %d, opRow: %d.\n",
                            //       __LINE__, bInx, PEIndex, PENum,
                            //       cuProcSweepIndex, opRow);
                            // for(int k = 0; k < rowB1; k++ )
                            // {
                            //   printf("u[%d] = %f,",
                            //       k, u[k]);
                            // }
                            printf("\n");
                        }

                        __syncthreads();
#endif

// if(DEBUG_BLOCK == bInx && 0==i && 0==j)
// {
//   printf("line= %d,BInx: %d, rowS: %d, colS: %d, opRow: %d, localColS: %d,
//   remoteSOffset: %d\n",
//         __LINE__, bInx, rowS, colS, opRow, localColS, remoteSOffset);
// }

// #if MY_DEBUG
#if 0
            // if ((DEBUG_BLOCK == bInx) && (0 == i) && (0 == j))
            if ((31 == i) && (31 == j))
            {
              // printf("block[%d] [%d][%d] come line=%d.\n",
              //       bInx, i, j, __LINE__);
              printf("line= %d,BInx: %d, PEIndex: %d, PENum: %d, cuProcSweepIndex: %d, startCol: %d, endCol: %d, k1=%d, k2=%d, k=%d,"
                      "rowS: %d, colS: %d, opRow: %d, localColS: %d, remoteSOffset: %d, offset: %d\n", 
                    __LINE__, bInx, PEIndex, PENum, cuProcSweepIndex, startCol, endCol, k1, k2, k, rowS, colS, opRow, localColS, remoteSOffset, offset);
              
              for(int h = 0; h < colS; h++)
              {
                for(int k = 0; k < rowS; k++ )
                {
                  printf("S[%d][%d]=%f,",
                      k, h, S2[k + h*ldSS]);
                }
                printf("\n");
              }
                  // printf("\b");
            }
            __syncthreads();
#endif

                        // 3.1.2 一起对B1进行Householder变换
                        for (int opColB1 = j; opColB1 < colB1; opColB1 += Ny) {
                            T nu = 0.0;
                            // 先计算u'x

                            for (int opRowB1 = i; opRowB1 < rowB1;
                                 opRowB1 += Nx) {
                                nu += u[opRowB1] * S1[opRowB1 + opColB1 * ldSS];
                            }

                            T utx = warpAllReduceSum(nu);

                            // 计算x-uu'x

                            for (int opRowB1 = i; opRowB1 < rowB1;
                                 opRowB1 += Nx) {
                                S1[opRowB1 + opColB1 * ldSS] -=
                                    utx * u[opRowB1];
                            }

                            __syncwarp();
                        }

                        __syncthreads();

// #if MY_DEBUG
#if 0
            if ((31 == i) && (31 == j))
            {
              printf("line= %d,BInx: %d, PEIndex: %d, PENum: %d, cuProcSweepIndex: %d, startCol: %d, endCol: %d, k1=%d, k2=%d, k=%d,"
                      "rowB1: %d, colB1: %d, opRow: %d, localColB1: %d\n", 
                    __LINE__, bInx, PEIndex, PENum, cuProcSweepIndex, startCol, endCol, k1, k2, k, rowB1, colB1, opRow, localColB1);
              for(int h = 0; h < colB1; h++)
              {
                for(int k = 0; k < rowB1; k++ )
                {
                  printf("B1[%d][%d] = %f,",
                      k, h, S1[k + h*ldSS]);
                }
                printf("\n");
              }
                  // printf("\b");
            }
            __syncthreads();
#endif

                        // 3.2 第1组warp:
                        // 对S1进行处理--写回B1(包括B1和其转置位置),取B2到S1中
                        // 注意:
                        // 取B2前都要判断数据同步条件--其实就是在B2变化的时候进行修改和判断同步条件
                        // 3.2 第2组warp: 对S2进行处理--对S进行Householder变换
                        if (j < warpGroupThdCount) {
                            // 第1组warp:
                            // 对S2进行处理--写回B1(包括B1和其转置位置),取B2到S1中
                            for (int opColB1 = j; opColB1 < colB1;
                                 opColB1 += warpGroupThdCount) {
                                for (int opRowB1 = i; opRowB1 < rowB1;
                                     opRowB1 += Nx) {
                                    if (opColB1 < localColB1) {
                                        // 由于是阶梯状,所以S每列会多减去列数
                                        localB1[(opRowB1 - opColB1) +
                                                opColB1 * ldSubA] =
                                            S1[opRowB1 + opColB1 * ldSS];
                                    } else {
                                        // 远程数据,需要进行远程加载
                                        // nvshmem_double_put(&dSubA[(opRowB1 -
                                        // opColB1) + colB1 +(opColB1 -
                                        // localColB1) * ldSubA],
                                        //                     &S1[opRowB1 +
                                        //                     opColB1 * ldSS],
                                        //                     1, PEIndex+1);
                                        int offset =
                                            colB1 +
                                            (opRow - colB1 - endCol) * ldSubA +
                                            (opRowB1 - opColB1) +
                                            opColB1 * ldSubA;

                                        if constexpr (std::is_same_v<T,
                                                                     double>) {
                                            nvshmem_double_p(
                                                dSubA + offset,
                                                S1[opRowB1 + opColB1 * ldSS],
                                                PEIndex + 1);
                                        } else if constexpr (std::is_same_v<
                                                                 T, float>) {
                                            nvshmem_float_p(
                                                dSubA + offset,
                                                S1[opRowB1 + opColB1 * ldSS],
                                                PEIndex + 1);
                                        }

                                        nvshmem_quiet();
                                    }
                                }
                            }

                            opRow += rowB1;  // 更新下一次A的长度

                            rowB1 = min(
                                b,
                                (int)(n -
                                      opRow));  // 因为opRow是起始位置,所以是n-1
                                                // - opRow +1 = n - opRow;
                            colB1 = colS;

                            // 计算在本PE上的B1的列数
                            localColB1 = localColS;
                            // int remoteColB1 = b - localColB1;

                            // 加载本PE上的B1到S1中
                            // 本地B1的起始位置
                            localB1 = dSubA + colB1 +
                                      (opRow - colB1 - startCol) * ldSubA;

                            for (int opColB1 = j; opColB1 < colB1;
                                 opColB1 += warpGroupThdCount) {
                                for (int opRowB1 = i; opRowB1 < rowB1;
                                     opRowB1 += Nx) {
                                    if (opColB1 < localColB1) {
                                        // 由于是阶梯状,所以S每列会多减去列数
                                        S1[opRowB1 + opColB1 * ldSS] =
                                            localB1[(opRowB1 - opColB1) +
                                                    opColB1 * ldSubA];
                                    } else {
                                        // 远程数据,需要进行远程加载
                                        // nvshmem_double_get(&S1[opRowB1 +
                                        // opColB1 * ldSS],
                                        //                     &dSubA[(opRowB1 -
                                        //                     opColB1) + colB1
                                        //                     + (opColB1 -
                                        //                     localColB1) *
                                        //                     ldSubA], 1,
                                        //                     PEIndex+1);

                                        int offset =
                                            colB1 +
                                            (opRow - colB1 - endCol) * ldSubA +
                                            (opRowB1 - opColB1) +
                                            opColB1 * ldSubA;

                                        if constexpr (std::is_same_v<T,
                                                                     double>) {
                                            S1[opRowB1 + opColB1 * ldSS] =
                                                nvshmem_double_g(dSubA + offset,
                                                                 PEIndex + 1);
                                        } else if constexpr (std::is_same_v<
                                                                 T, float>) {
                                            S1[opRowB1 + opColB1 * ldSS] =
                                                nvshmem_float_g(dSubA + offset,
                                                                PEIndex + 1);
                                        }
                                    }
                                }
                            }

                            // nvshmem_quiet(); // 保证远程数据能够到达

                        } else {
                            // 第2组warp: 对S2进行处理--对S进行Householder变换

                            for (int opColS = j - warpGroupThdCount;
                                 opColS < colS; opColS += warpGroupThdCount) {
                                T nu = 0.0;
                                // 先计算u'x
                                for (int opRowS = i; opRowS < rowS;
                                     opRowS += Nx) {
                                    nu +=
                                        u[opRowS] * S2[opRowS + opColS * ldSS];
                                }

                                T utx = warpAllReduceSum(nu);

                                // 计算x-uu'x

                                for (int opRowS = i; opRowS < rowS;
                                     opRowS += Nx) {
                                    S2[opRowS + opColS * ldSS] -=
                                        utx * u[opRowS];
                                }

                                __syncwarp();
                            }

                            opRow += rowB1;  // 更新下一次A的长度

                            rowB1 = min(
                                b,
                                (int)(n -
                                      opRow));  // 因为opRow是起始位置,所以是n-1
                                                // - opRow +1 = n - opRow;
                            colB1 = colS;

                            // 计算在本PE上的B1的列数
                            localColB1 = localColS;
                            // int remoteColB1 = b - localColB1;

                            // 加载本PE上的B1到S1中
                            // 本地B1的起始位置
                            localB1 = dSubA + colB1 +
                                      (opRow - colB1 - startCol) * ldSubA;
                        }

                        __syncthreads();

// #if MY_DEBUG
#if 0
            // if ((DEBUG_BLOCK == bInx) && (0 == i) && (0 == j))
            if ((31 == i) && (31 == j))
            {
              // printf("block[%d] [%d][%d] come line=%d.\n",
              //       bInx, i, j, __LINE__);
              printf("line= %d,BInx: %d, PEIndex: %d, PENum: %d, cuProcSweepIndex: %d, startCol: %d, endCol: %d, k1=%d, k2=%d, k=%d,"
                      "rowB2: %d, colB2: %d, opRow: %d, localColB2: %d\n", 
                    __LINE__, bInx, PEIndex, PENum, cuProcSweepIndex, startCol, endCol, k1, k2, k, rowB1, colB1, opRow, localColB1);
              
              for(int h = 0; h < colB1; h++)
              {
                for(int k = 0; k < rowB1; k++ )
                {
                  printf("B2[%d][%d]=%f,",
                      k, h, S1[k + h*ldSS]);
                }
                printf("\n");
              }
                  // printf("\b");
            }
            __syncthreads();
#endif

// 对S2进行转置方向的Householder变换
#pragma unroll
                        for (int opRowS = j; opRowS < rowS; opRowS += Ny) {
                            T nu = 0.0;
                            // 先计算u'x

                            for (int opColS = i; opColS < colS; opColS += Nx) {
                                nu += u[opColS] * S2[opRowS + opColS * ldSS];
                            }

                            T utx = warpAllReduceSum(nu);

                            // 计算x-uu'x

                            for (int opColS = i; opColS < colS; opColS += Nx) {
                                S2[opRowS + opColS * ldSS] -= utx * u[opColS];
                            }

                            __syncwarp();
                        }

                        __syncthreads();

#if MY_DEBUG
                        // #if 0
                        // if ((DEBUG_BLOCK == bInx) && (0 == i) && (0 == j))
                        if ((31 == i) && (31 == j)) {
                            // printf("block[%d] [%d][%d] come line=%d.\n",
                            //       bInx, i, j, __LINE__);
                            // printf("line= %d,BInx: %d, PEIndex: %d, PENum:
                            // %d, cuProcSweepIndex: %d, startCol: %d, endCol:
                            // %d, k1=%d, k2=%d, k=%d,"
                            //         "rowS: %d, colS: %d, opRow: %d,
                            //         localColS: %d, remoteSOffset: %d, offset:
                            //         %d\n",
                            //       __LINE__, bInx, PEIndex, PENum,
                            //       cuProcSweepIndex, startCol, endCol, k1, k2,
                            //       k, rowS, colS, opRow, localColS,
                            //       remoteSOffset, offset);

                            printf(
                                "line= %d,BInx: %d, PEIndex: %d, PENum: %d, "
                                "cuProcSweepIndex: %d, startCol: %d, endCol: "
                                "%d, k1=%d, k2=%d, k=%d,"
                                "rowS: %d, colS: %d, opRow: %d, localColS: "
                                "%d\n",
                                __LINE__, bInx, PEIndex, PENum,
                                cuProcSweepIndex, startCol, endCol, k1, k2, k,
                                rowS, colS, opRow, localColS);

                            for (int h = 0; h < colS; h++) {
                                for (int k = 0; k < rowS; k++) {
                                    printf("S[%d][%d]=%f,", k, h,
                                           S2[k + h * ldSS]);
                                }
                                printf("\n");
                            }
                            // printf("\b");
                        }
                        __syncthreads();
#endif

                        // 3.2 第1组warp: 对S2进行处理--将S2写入到S中
                        // 3.2 第2组warp: 对S1进行处理--对S1进行Householder变换
                        if (j < warpGroupThdCount) {
                            // 第1组warp: 对S2进行处理--将S2写入到S中
                            // 利用对称性,只写入下半部分矩阵
                            for (int opColS = j; opColS < colS;
                                 opColS += warpGroupThdCount) {
                                for (int opRowS = i;
                                     (opColS <= opRowS) && (opRowS < rowS);
                                     opRowS += Nx) {
                                    // pB2[opRowB2][opColB2] =
                                    // B1[opColB2][opRowB2] =
                                    // SS[opColB2][opRowB2]
                                    // 由于是阶梯状,所以S每列会多减去列数
                                    // S2[opColS + opRowS * ldSS] =
                                    if (opColS < localColS) {
                                        localS[(opRowS - opColS) +
                                               opColS * ldSubA] =
                                            S2[opRowS + opColS * ldSS];
                                    } else {
                                        // 远程数据,需要进行远程加载
                                        // nvshmem_double_put(&remoteS[(opRowS -
                                        // opColS) + opColS * ldSubA],
                                        //                     &S2[opRowS +
                                        //                     opColS * ldSS],
                                        //                     1, PEIndex+1);

                                        // nvshmem_double_p(remoteS + (opRowS -
                                        // opColS) + opColS * ldSubA,
                                        //                     S2[opRowS +
                                        //                     opColS * ldSS],
                                        //                     PEIndex+1);

                                        int offset = remoteSOffset +
                                                     (opRowS - opColS) +
                                                     opColS * ldSubA;

                                        if constexpr (std::is_same_v<T,
                                                                     double>) {
                                            nvshmem_double_p(
                                                dSubA + offset,
                                                S2[opRowS + opColS * ldSS],
                                                PEIndex + 1);
                                        } else if constexpr (std::is_same_v<
                                                                 T, float>) {
                                            nvshmem_float_p(
                                                dSubA + offset,
                                                S2[opRowS + opColS * ldSS],
                                                PEIndex + 1);
                                        }

                                        nvshmem_quiet();  // 保证远程数据能够到达
                                    }
                                }
                            }

                        } else {
                            // 第2组warp:
                            // 对S1进行处理--对B2的转置进行Householder变换
                            for (int opRowB1 = j - warpGroupThdCount;
                                 opRowB1 < rowB1;
                                 opRowB1 += warpGroupThdCount) {
                                T nu = 0.0;
                                // 先计算u'x
                                for (int opColB1 = i; opColB1 < colB1;
                                     opColB1 += Nx) {
                                    nu += u[opColB1] *
                                          S1[opRowB1 + opColB1 * ldSS];
                                }

                                T utx = warpAllReduceSum(nu);

                                // 计算x-uu'x
                                for (int opColB1 = i; opColB1 < colB1;
                                     opColB1 += Nx) {
                                    S1[opRowB1 + opColB1 * ldSS] -=
                                        utx * u[opColB1];
                                }

                                __syncwarp();
                            }
                        }

                        __syncthreads();

// #if MY_DEBUG
#if 0
            // if ((DEBUG_BLOCK == bInx) && (0 == i) && (0 == j))
            if ((31 == i) && (31 == j))
            {
              // printf("block[%d] [%d][%d] come line=%d.\n",
              //       bInx, i, j, __LINE__);
              printf("line= %d,BInx: %d, PEIndex: %d, PENum: %d, cuProcSweepIndex: %d, startCol: %d, endCol: %d, k1=%d, k2=%d, k=%d,"
                      "rowB2: %d, colB2: %d, opRow: %d, localColB2: %d\n", 
                    __LINE__, bInx, PEIndex, PENum, cuProcSweepIndex, startCol, endCol, k1, k2, k, rowB1, colB1, opRow, localColB1);
              
              for(int h = 0; h < colB1; h++)
              {
                for(int k = 0; k < rowB1; k++ )
                {
                  printf("B2[%d][%d]=%f,",
                      k, h, S1[k + h*ldSS]);
                }
                printf("\n");
              }
                  // printf("\b");
            }
            __syncthreads();
#endif

                        // 将B1写回去: 包括本地和remote的数据
                        for (int opColB1 = j; opColB1 < colB1; opColB1 += Ny) {
                            for (int opRowB1 = i; opRowB1 < rowB1;
                                 opRowB1 += Nx) {
                                if (opColB1 < localColB1) {
                                    // 由于是阶梯状,所以S每列会多减去列数
                                    localB1[(opRowB1 - opColB1) +
                                            opColB1 * ldSubA] =
                                        S1[opRowB1 + opColB1 * ldSS];
                                } else {
                                    // 远程数据,需要进行远程加载
                                    // nvshmem_double_put(&dSubA[(opRowB1 -
                                    // opColB1) + colB1 + (opColB1 - localColB1)
                                    // * ldSubA],
                                    //                     &S1[opRowB1 + opColB1
                                    //                     * ldSS], 1,
                                    //                     PEIndex+1);

                                    // 先计算在整个dA中的位置,
                                    // 再求出在netxPE的dSubA中的位置
                                    int offset =
                                        colB1 +
                                        (opRow - colB1 - endCol) * ldSubA +
                                        (opRowB1 - opColB1) + opColB1 * ldSubA;

                                    if constexpr (std::is_same_v<T, double>) {
                                        nvshmem_double_p(
                                            dSubA + offset,
                                            S1[opRowB1 + opColB1 * ldSS],
                                            PEIndex + 1);
                                    } else if constexpr (std::is_same_v<
                                                             T, float>) {
                                        nvshmem_float_p(
                                            dSubA + offset,
                                            S1[opRowB1 + opColB1 * ldSS],
                                            PEIndex + 1);
                                    }

                                    nvshmem_quiet();
                                }
                            }
                        }

                        __syncthreads();

                        // 进行同步: 写入现在正在进行处理的opRow,通知remote
                        // PE可以开始处理

                        // nvshmem_quiet(); // 保证远程数据能够到达

                        k++;

                        if (0 == i && 0 == j) {
                            com[cuProcSweepIndex] =
                                opRow;  // 更新当前PE当前block的处理位置
                        }

                        if (0 < PEIndex &&
                            cuProcSweepIndex == g_tailSweepIndex) {
                            if (0 == i && 0 == j) {
                                nvshmem_int_p(nextPEWriteTailSweepProcRow,
                                              opRow, PEIndex - 1);

                                nvshmem_quiet();
                            }
                        }
                    }
                }
            }

            __syncthreads();

            // 最后waitPE++; 判断是否处理完成前面所有PE的BC Sweep
            if (k2 == k) {
                waitFlag = true;

                // 最后1个PE的g_headSweepIndex不需要改变,才能保证判断的正确性
                if ((PEIndex < PENum - 1) && (0 == i && 0 == j)) {
                    // 处理完成前面所有PE的BC Sweep,通知下1个PE
                    g_headSweepIndex++;
                }

                // 通知下1个PE, 本PE的waitStartSweepIndex趟处理完毕
                if (PEIndex < PENum - 1) {
                    // 通知下1个PE, 本PE的waitStartSweepIndex趟处理完毕
                    nvshmem_int_p(prePEWriteCom + cuProcSweepIndex, 1,
                                  PEIndex + 1);
                    nvshmem_quiet();
                }

                if (PEIndex == (PENum - 1)) {
                    // 最后1个PE,需要将g_tailSweepIndex更新为当前处理位置
                    if (0 == i && 0 == j) {
                        com[cuProcSweepIndex] =
                            opRow + 3 * b;  // 更新当前PE当前block的处理位置

                        if (0 < PEIndex &&
                            cuProcSweepIndex == g_tailSweepIndex) {
                            nvshmem_int_p(nextPEWriteTailSweepProcRow,
                                          opRow + 3 * b, PEIndex - 1);
                            nvshmem_quiet();
                        }
                    }
                }

                cuProcSweepIndex += blockNum;  // 更新当前PE的处理位置
                if (0 == procState && cuProcSweepIndex >= startCol) {
                    // 说明已经处理完了前面所有PE的BC Sweep,可以进入下一个状态
                    procState = 1;
                    // initFlag = true; // 重新初始化
                } else if (1 == procState &&
                           cuProcSweepIndex > endBCSweepIndex) {
                    // 说明已经处理完了前面所有PE的BC
                    // Sweep,也处理了当前PE发起的BC Sweep,可以进入下一个状态
                    procState = 2;

                    // if((PEIndex == PENum-1) && (blockNum - 1) == bInx )
                    if ((cuProcSweepIndex - blockNum) == endBCSweepIndex) {
                        // 说明已经处理完了前面所有PE的BC Sweep,可以退出循环
                        if (0 == i && 0 == j) {
                            g_cycleFlag = false;

// #if MY_DEBUG
#if 1
                            printf(
                                "PE %d, block %d, cuProcSweepIndex = %d, "
                                "cycleFlag = false\n",
                                PEIndex, bInx, cuProcSweepIndex);
#endif

                            // nvshmem_int_p(g_cycleFlag, 1, 0);
                        }
                    }
                }
            }

            // __syncthreads();
        }

        // nvshmem_quiet(); // 保证远程数据能够到达
        grid.sync();
        // 进行同步

        // nvshmem_barrier_all();
    }

    // nvshmem_barrier_all();

    // 处理完所有的BC Sweep,退出循环
    // if(0 == i && 0 == j)
    // {
    //   printf("PE %d, block %d, exit BC Sweep\n", PEIndex, bInx);
    // }

    // grid.sync();
}

// 添加函数的说明
template <typename T>
void sb2tr(int n, int b, int ns, T* dSubA, int ldSubA, T* dU, int ldU,
           int PEIndex, int PENum, int* com, int* prePEWriteCom,
           int* nextPEWriteTailSweepProcRow) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    dim3 dimBlock(32, 32, 1);
#else
    dim3 dimBlock(32, 16, 1);
#endif

    int device = -1;
    cudaGetDevice(&device);
    util::MpiLogger::println("BC device: {}, dimBlock.x: {}, dimBlock.y: {}",
                             device, dimBlock.x, dimBlock.y);

    int blockNum;

    // 4. 创建ns个线程块, 每个线程块进行1趟BC处理
    void* kernelArgs[] = {
        (void*)&n,
        (void*)&b,
        (void*)&ns,
        (void*)&dSubA,
        (void*)&ldSubA,
        (void*)&dU,
        (void*)&ldU,
        (void*)&blockNum,
        (void*)&PEIndex,
        (void*)&PENum,
        (void*)&com,
        (void*)&prePEWriteCom,
        (void*)&nextPEWriteTailSweepProcRow,
        //   (void *)&g_cycleFlag,
    };

    // int supportsCoopLaunch = 0;
    // cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch,
    //                        rank);

    int res = nvshmemx_collective_launch_query_gridsize(
        (void*)kernel_BC_NVShmem_V3<32, T>, dimBlock, kernelArgs, 0, &blockNum);

    util::MpiLogger::println("res :{} BC can start blockNum: {}", res,
                             blockNum);

    blockNum = std::min(128, blockNum);

    dim3 dimGrid(blockNum, 1, 1);

    util::MpiLogger::tic("sb2tr_BC");
    res = nvshmemx_collective_launch((void*)kernel_BC_NVShmem_V3<32, T>,
                                     dimGrid, dimBlock, kernelArgs, 0, 0);

    cudaDeviceSynchronize();

    util::MpiLogger::toc("sb2tr_BC");

    return;
}

template <typename T>
__global__ void kernel_bugle_chasing_cpydA2dSubA(int n, int b, int cols_perPE,
                                                 int rank, T* dA, long ldA,
                                                 T* dSubA, int ldSubA) {
    // int bInx = blockIdx.y * gridDim.x + blockIdx.x;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // 伪代码
    // j索引等于本线程复制的A的列索引
    //

    // 起始位置和结束位置都是在同列中,计算的是行数
    // 1.找到A的起始复制位置--就是j

    // if (i < (b + 1) && j < n)
    if ((i < 2 * b) && j < cols_perPE) {
        // 2.找到A的结束位置
        // int end = min(n, j + 2b);
        int end = min(n, rank * cols_perPE + j + b +
                             1);  // 开始时,下面的b-1个元素是0,不用进行拷贝

        // 3.计算复制个数
        int count = end - j;

        // printf("block[%d] [%d][%d] come line=%d,count=%d.\n", bInx, i, j,
        // __LINE__, count);

        if (i < count) {
            dSubA[i + j * ldSubA] = dA[j + i + j * ldA];
        } else {
            dSubA[i + j * ldSubA] = 0.0;
        }
    }
}

}  // namespace sb2tr
}  // namespace matrix_ops

template void matrix_ops::sb2tr::sb2tr<float>(int n, int b, int ns,
                                              float* dSubA, int ldSubA,
                                              float* dU, int ldU, int PEIndex,
                                              int PENum, int* com,
                                              int* prePEWriteCom,
                                              int* nextPEWriteTailSweepProcRow);

template void matrix_ops::sb2tr::sb2tr<double>(
    int n, int b, int ns, double* dSubA, int ldSubA, double* dU, int ldU,
    int PEIndex, int PENum, int* com, int* prePEWriteCom,
    int* nextPEWriteTailSweepProcRow);

template __global__ void matrix_ops::sb2tr::kernel_bugle_chasing_cpydA2dSubA<
    float>(int n, int b, int cols_perPE, int rank, float* dA, long ldA,
           float* dSubA, int ldSubA);

template __global__ void matrix_ops::sb2tr::kernel_bugle_chasing_cpydA2dSubA<
    double>(int n, int b, int cols_perPE, int rank, double* dA, long ldA,
            double* dSubA, int ldSubA);