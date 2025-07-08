#include <cstddef>

#include "matrix_ops.cuh"

namespace matrix_ops {

namespace internal {
    template <typename T>
    void tsqr_wrapper() {
        
    }
}

template <typename T>
void sy2sb(const CublasHandle& handle, size_t n,
           thrust::device_vector<T>& A_inout) {
    // the size of the matrix A
    const auto m = (size_t)n;
    // the panel size
    const auto b = (size_t)32;
    // the block size
    const auto nb = (size_t)b * 4;
}

}  // namespace matrix_ops