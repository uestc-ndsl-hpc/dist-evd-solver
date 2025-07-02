#pragma once

#include <curand_kernel.h>
#include <fmt/format.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>
#include <time.h>
#include <curand.h>

#include <string>
#include <type_traits>

#include "fmt/base.h"
#include "log.h"

namespace matrix_ops {

namespace detail {

template <typename T>
struct symmetrize_functor {
    T* C;
    const int n;
    symmetrize_functor(T* C, int n) : C(C), n(n) {}

    __device__ void operator()(const int i) const {
        int row = i / n;
        int col = i % n;
        if (col >= row) {
            T val_ij = C[row * n + col];
            T val_ji = C[col * n + row];
            T avg = (val_ij + val_ji) / 2.0;
            C[row * n + col] = avg;
            C[col * n + row] = avg;
        }
    }
};
}  // namespace detail

template <typename T>
void print(const thrust::device_vector<T>& d_vec, int n,
           const std::string& title) {
    fmt::println("\n--- {} ({}x{}) ---", title, n, n);
    thrust::host_vector<T> h_vec = d_vec;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if constexpr (std::is_floating_point_v<T>) {
                fmt::print("{:8.4f} ", h_vec[i * n + j]);
            } else {
                fmt::print("{} ", h_vec[i * n + j]);
            }
        }
        fmt::println("");
    }
    fmt::println("---------------------------\n");
}

template <typename T>
thrust::device_vector<T> create_symmetric_random(int n) {
    util::Logger::println("Creating test device C of size {}x{}", n, n);
    auto C = thrust::device_vector<T>(n * n);
    auto C_ptr = thrust::raw_pointer_cast(C.data());
    
    util::Logger::println("Step 1: Filling C with random values using cuRAND library");
    
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(nullptr));

    if constexpr (std::is_same_v<T, float>) {
        curandGenerateUniform(gen, C_ptr, static_cast<size_t>(n) * n);
    } else if constexpr (std::is_same_v<T, double>) {
        curandGenerateUniformDouble(gen, C_ptr, static_cast<size_t>(n) * n);
    } else {
        fmt::report_error("Unsupported type");
    }
    
    curandDestroyGenerator(gen);
    cudaDeviceSynchronize();

    if (util::Logger::is_verbose()) {
        print(C, n, "Matrix C after random fill");
    }

    util::Logger::println("Step 2: Symmetrizing C via (C + C^T) / 2");
    thrust::for_each(thrust::counting_iterator<int>(0),
                     thrust::counting_iterator<int>(n * n),
                     detail::symmetrize_functor<T>(C_ptr, n));
    cudaDeviceSynchronize();

    util::Logger::println("Finished creating symmetric random matrix C");
    return C;
}

}  // namespace matrix_ops