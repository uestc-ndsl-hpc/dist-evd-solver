#include <curand.h>
#include <fmt/core.h>
#include <thrust/device_vector.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <time.h>

#include <cstddef>
#include <type_traits>

#include "log.h"
#include "matrix_ops.cuh"

namespace matrix_ops {

namespace detail {

enum class RandomDistribution { UNIFORM, NORMAL };

template <typename T>
void generate_random_inplace(T* C_ptr, size_t n_elements,
                             RandomDistribution dist, T arg1, T arg2) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, time(nullptr));

    if (dist == RandomDistribution::UNIFORM) {
        if constexpr (std::is_same_v<T, float>) {
            curandGenerateUniform(gen, C_ptr, n_elements);
        } else if constexpr (std::is_same_v<T, double>) {
            curandGenerateUniformDouble(gen, C_ptr, n_elements);
        } else {
            fmt::print(stderr, "Unsupported type for random generation\n");
        }
    } else if (dist == RandomDistribution::NORMAL) {
        if constexpr (std::is_same_v<T, float>) {
            curandGenerateNormal(gen, C_ptr, n_elements, arg1, arg2);
        } else if constexpr (std::is_same_v<T, double>) {
            curandGenerateNormalDouble(gen, C_ptr, n_elements, arg1, arg2);
        } else {
            fmt::print(stderr, "Unsupported type for random generation\n");
        }
    }

    curandDestroyGenerator(gen);
}

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
thrust::device_vector<T> create_uniform_random(size_t m, size_t n) {
    util::Logger::tic("create_uniform_random");
    auto C = thrust::device_vector<T>(m * n);
    auto C_ptr = thrust::raw_pointer_cast(C.data());

    detail::generate_random_inplace(
        C_ptr, m * n, detail::RandomDistribution::UNIFORM, (T)0.0, (T)1.0);
    cudaDeviceSynchronize();
    util::Logger::toc("create_uniform_random");
    return C;
}

template <typename T>
thrust::device_vector<T> create_uniform_random(size_t n) {
    return create_uniform_random<T>(n, n);
}

template <typename T>
thrust::device_vector<T> create_normal_random(size_t m, size_t n, T mean,
                                              T stddev) {
    util::Logger::tic("create_normal_random");
    auto C = thrust::device_vector<T>(m * n);
    auto C_ptr = thrust::raw_pointer_cast(C.data());

    detail::generate_random_inplace(
        C_ptr, m * n, detail::RandomDistribution::NORMAL, mean, stddev);
    cudaDeviceSynchronize();
    util::Logger::toc("create_normal_random");
    return C;
}

template <typename T>
thrust::device_vector<T> create_normal_random(size_t n, T mean, T stddev) {
    return create_normal_random<T>(n, n, mean, stddev);
}

template <typename T>
thrust::device_vector<T> create_symmetric_random(size_t n) {
    util::Logger::tic("create_symmetric_random");
    util::Logger::println("Creating test device C of size {}x{}", n, n);
    auto C = thrust::device_vector<T>(n * n);
    auto C_ptr = thrust::raw_pointer_cast(C.data());
    detail::generate_random_inplace(
        C_ptr, n * n, detail::RandomDistribution::UNIFORM, (T)0.0, (T)1.0);
    cudaDeviceSynchronize();
    thrust::for_each(thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(n * n),
                     detail::symmetrize_functor<T>(C_ptr, n));
    cudaDeviceSynchronize();
    util::Logger::toc("create_symmetric_random");
    return C;
}

// Explicitly instantiate the templates
template thrust::device_vector<float> create_uniform_random<float>(size_t);
template thrust::device_vector<double> create_uniform_random<double>(size_t);
template thrust::device_vector<float> create_uniform_random<float>(size_t,
                                                                   size_t);
template thrust::device_vector<double> create_uniform_random<double>(size_t,
                                                                     size_t);
template thrust::device_vector<float> create_normal_random<float>(size_t, float,
                                                                  float);
template thrust::device_vector<double> create_normal_random<double>(size_t,
                                                                    double,
                                                                    double);
template thrust::device_vector<float> create_normal_random<float>(size_t,
                                                                  size_t, float,
                                                                  float);
template thrust::device_vector<double> create_normal_random<double>(size_t,
                                                                    size_t,
                                                                    double,
                                                                    double);
template thrust::device_vector<float> create_symmetric_random<float>(size_t);
template thrust::device_vector<double> create_symmetric_random<double>(size_t);

}  // namespace matrix_ops
