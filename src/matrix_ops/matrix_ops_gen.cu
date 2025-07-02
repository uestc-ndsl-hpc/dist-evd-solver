#include "../include/matrix_ops.cuh"
#include "../include/log.h"

#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <curand.h>
#include <time.h>
#include <cstddef>
#include <type_traits>
#include <fmt/core.h>

namespace matrix_ops {

namespace detail {

template <typename T>
struct symmetrize_functor {
    T* C;
    const size_t n;
    symmetrize_functor(T* C, size_t n) : C(C), n(n) {}

    __device__ void operator()(const size_t i) const {
        size_t row = i / n;
        size_t col = i % n;
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
thrust::device_vector<T> create_symmetric_random(size_t n) {
    util::Logger::tic("create_symmetric_random");
    util::Logger::println("Creating test device C of size {}x{}", n, n);
    auto C = thrust::device_vector<T>(n * n);
    auto C_ptr = thrust::raw_pointer_cast(C.data());
    
    util::Logger::println("Step 1: Filling C with random values using cuRAND library");
    
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(nullptr));

    if constexpr (std::is_same_v<T, float>) {
        curandGenerateUniform(gen, C_ptr, n * n);
    } else if constexpr (std::is_same_v<T, double>) {
        curandGenerateUniformDouble(gen, C_ptr, n * n);
    } else {
        fmt::print(stderr, "Unsupported type for random generation\n");
    }
    
    curandDestroyGenerator(gen);
    cudaDeviceSynchronize();

    if (util::Logger::is_verbose()) {
        print(C, n, "Matrix C after random fill");
    }

    util::Logger::println("Step 2: Symmetrizing C via (C + C^T) / 2");
    thrust::for_each(thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(n * n),
                     detail::symmetrize_functor<T>(C_ptr, n));
    cudaDeviceSynchronize();

    util::Logger::println("Finished creating symmetric random matrix C");
    util::Logger::toc("create_symmetric_random");
    return C;
}

// Explicitly instantiate the templates
template thrust::device_vector<float> create_symmetric_random<float>(size_t);
template thrust::device_vector<double> create_symmetric_random<double>(size_t);

} // namespace matrix_ops
