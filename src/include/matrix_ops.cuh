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

#include <cstddef>
#include <string>

namespace matrix_ops {

template <typename T>
void print(const thrust::device_vector<T>& d_vec, size_t n,
           const std::string& title);

template <typename T>
thrust::device_vector<T> create_symmetric_random(size_t n);

}  // namespace matrix_ops