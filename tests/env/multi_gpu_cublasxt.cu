#include "fmt/base.h"
#include "gpu_handle_wrappers.h"
#include "log.h"
#include "matrix_ops.cuh"

int main() {
    util::Logger::init_timer(true);

    fmt::println("===test normal cublas for single gpu max===");

    {
        int size_cublas = 24576;

        auto A =
            matrix_ops::create_normal_random<float>(size_cublas, size_cublas);
        auto B =
            matrix_ops::create_normal_random<float>(size_cublas, size_cublas);
        auto C = thrust::device_vector<float>(size_cublas * size_cublas, 0.0);

        common::CublasHandle cublasHandle;

        util::Logger::tic("cublas gemm");

        matrix_ops::gemm<float>(cublasHandle, size_cublas, size_cublas,
                                size_cublas, 1, A.data(), size_cublas, false,
                                B.data(), size_cublas, false, 0, C.data(),
                                size_cublas);
        util::Logger::toc("cublas gemm");
    }

    fmt::println("===test cublasXt for multi gpu===");

    {
        int size_cublasxt = 24576;

        auto A = thrust::host_vector<float>(size_cublasxt * size_cublasxt);
        auto B = thrust::host_vector<float>(size_cublasxt * size_cublasxt);
        auto C = thrust::host_vector<float>(size_cublasxt * size_cublasxt);
        common::CublasXtHandle cublasXtHandle;

        int gpu_num = 1;
        int xt_devices[1] = {0};

        if (cublasXtDeviceSelect(cublasXtHandle, gpu_num, xt_devices) !=
            CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("cublasXtDeviceSelect failed");
        } else {
            fmt::println("===  success select gpus ===");
        }

        {
            auto Ad = matrix_ops::create_normal_random<float>(size_cublasxt,
                                                              size_cublasxt);
            A = Ad;
        }
        {
            auto Bd = matrix_ops::create_normal_random<float>(size_cublasxt,
                                                              size_cublasxt);
            B = Bd;
        }
        float alpha = 1.0f;
        float beta = 0.0f;
        util::Logger::tic("cublasXt gemm");

        cublasXtSgemm(cublasXtHandle, CUBLAS_OP_N, CUBLAS_OP_N, size_cublasxt,
                      size_cublasxt, size_cublasxt, &alpha, A.data(),
                      size_cublasxt, B.data(), size_cublasxt, &beta, C.data(),
                      size_cublasxt);

        util::Logger::toc("cublasXt gemm");

        matrix_ops::print(C.data() + size_cublasxt * (size_cublasxt - 10) + size_cublasxt - 10, 10,
        10, "cublasXt Test");
    }

    return 0;
}