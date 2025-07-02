#include "matrix_ops.cuh"

#include <fmt/format.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <string>
#include <type_traits>

namespace matrix_ops {

template <typename T>
void print(const thrust::device_vector<T>& d_vec, size_t m, size_t n,
           const std::string& title) {
    fmt::println("\n--- {} ({}x{}) ---", title, m, n);
    thrust::host_vector<T> h_vec = d_vec;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
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
void print(const thrust::device_vector<T>& d_vec, size_t n,
           const std::string& title) {
    print(d_vec, n, n, title);
}

// Explicitly instantiate the templates to allow for separate compilation
template void print<float>(const thrust::device_vector<float>&, size_t, size_t,
                         const std::string&);
template void print<double>(const thrust::device_vector<double>&, size_t, size_t,
                          const std::string&);
template void print<float>(const thrust::device_vector<float>&, size_t,
                         const std::string&);
template void print<double>(const thrust::device_vector<double>&, size_t,
                          const std::string&);

}  // namespace matrix_ops
