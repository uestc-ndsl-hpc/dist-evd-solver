#include <fmt/format.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <string>
#include <type_traits>

#include "matrix_ops.cuh"

namespace matrix_ops {

template <typename T>
void print(T* data, size_t m, size_t n, size_t lda, const std::string& title) {
    if (!title.empty()) {
        fmt::println("\n--- {} ({}x{}) ---", title, m, n);
    }
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            if constexpr (std::is_floating_point_v<T>) {
                fmt::print("{:8.4f} ", data[j * m + i]);
            } else {
                fmt::print("{} ", data[j * m + i]);
            }
        }
        fmt::println("");
    }
    if (!title.empty()) {
        fmt::println("---------------------------\n");
    }
}

template <typename T>
void print(T* data, size_t m, size_t n, const std::string& title) {
    print(data, m, n, m, title);
}

template <typename T>
void print(thrust::device_ptr<T> data, size_t m, size_t n, size_t lda,
           const std::string& title) {
    if (m == 0 || n == 0) {
        return;
    }
    thrust::host_vector<T> h_data(m * n);
    matrix_ops::matrix_copy<thrust::device_ptr<T>, T*, T>(
        data, lda, thrust::raw_pointer_cast(h_data.data()), m, m, n);

    print(h_data.data(), m, n, lda, title);
}

template <typename T>
void print(thrust::device_ptr<T> data, size_t m, size_t n,
           const std::string& title) {
    print(data, m, n, m, title);
}

template <typename T>
void print(thrust::device_vector<T>& data, size_t m, size_t n,
           const std::string& title) {
    print(data.data(), m, n, title);
}

template <typename T>
void print(thrust::device_vector<T>& data, size_t n,
           const std::string& title) {
    print(data, n, n, title);
}

// Explicitly instantiate the templates to allow for separate compilation
template void print<float>(thrust::device_vector<float>&, size_t, size_t,
                           const std::string&);
template void print<double>(thrust::device_vector<double>&, size_t,
                            size_t, const std::string&);
template void print<float>(thrust::device_vector<float>&, size_t,
                           const std::string&);
template void print<double>(thrust::device_vector<double>&, size_t,
                            const std::string&);
template void print<float>(thrust::device_ptr<float>, size_t, size_t,
                           const std::string&);
template void print<double>(thrust::device_ptr<double>, size_t, size_t,
                            const std::string&);
template void print<float>(thrust::device_ptr<float>, size_t, size_t, size_t,
                           const std::string&);
template void print<double>(thrust::device_ptr<double>, size_t, size_t, size_t,
                            const std::string&);
template void print<float>(float*, size_t, size_t, size_t, const std::string&);
template void print<double>(double*, size_t, size_t, size_t,
                            const std::string&);
template void print<float>(float*, size_t, size_t, const std::string&);
template void print<double>(double*, size_t, size_t, const std::string&);

}  // namespace matrix_ops
