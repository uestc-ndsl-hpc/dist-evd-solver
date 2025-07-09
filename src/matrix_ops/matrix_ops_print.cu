#include <fmt/format.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <string>
#include <type_traits>

#include "matrix_ops.cuh"

namespace matrix_ops {

template <typename T>
void print(thrust::device_ptr<T> data, size_t m, size_t n,
           const std::string& title) {
    size_t size = m * n;
    thrust::host_vector<typename std::remove_const<T>::type> h_data(size);
    thrust::copy(data, data + size, h_data.begin());

    if (!title.empty()) {
        fmt::println("\n--- {} ({}x{}) ---", title, m, n);
    }
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if constexpr (std::is_floating_point_v<T>) {
                fmt::print("{:8.4f} ", h_data[i * n + j]);
            } else {
                fmt::print("{} ", h_data[i * n + j]);
            }
        }
        fmt::println("");
    }
    if (!title.empty()) {
        fmt::println("---------------------------\n");
    }
}

template <typename T>
void print(const thrust::device_vector<T>& data, size_t m, size_t n,
           const std::string& title) {
    print(data.data(), m, n, title);
}

template <typename T>
void print(const thrust::device_vector<T>& data, size_t n,
           const std::string& title) {
    print(data, n, n, title);
}

// Explicitly instantiate the templates to allow for separate compilation
template void print<float>(const thrust::device_vector<float>&, size_t, size_t,
                           const std::string&);
template void print<double>(const thrust::device_vector<double>&, size_t,
                            size_t, const std::string&);
template void print<float>(const thrust::device_vector<float>&, size_t,
                           const std::string&);
template void print<double>(const thrust::device_vector<double>&, size_t,
                            const std::string&);
template void print<float>(thrust::device_ptr<float>, size_t, size_t,
                           const std::string&);
template void print<double>(thrust::device_ptr<double>, size_t, size_t,
                            const std::string&);

}  // namespace matrix_ops
