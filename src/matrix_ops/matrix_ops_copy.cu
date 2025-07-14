#include <type_traits>

#include "matrix_ops.cuh"

namespace matrix_ops {

template <typename srcPtr, typename dstPtr, typename T>
void matrix_copy(srcPtr src, size_t src_ld, dstPtr dst, size_t dst_ld, size_t m,
                 size_t n) {
    // copy kind switch
    auto kind = cudaMemcpyKind::cudaMemcpyDeviceToDevice;
    if (std::is_same_v<srcPtr, thrust::device_ptr<T>> &&
        std::is_same_v<dstPtr, thrust::device_ptr<T>>) {
        kind = cudaMemcpyKind::cudaMemcpyDeviceToDevice;
    } else if (std::is_same_v<srcPtr, thrust::device_ptr<T>> &&
               std::is_same_v<dstPtr, T*>) {
        kind = cudaMemcpyKind::cudaMemcpyDeviceToHost;
    } else if (std::is_same_v<srcPtr, T*> &&
               std::is_same_v<dstPtr, thrust::device_ptr<T>>) {
        kind = cudaMemcpyKind::cudaMemcpyHostToDevice;
    } else if (std::is_same_v<srcPtr, T*> && std::is_same_v<dstPtr, T*>) {
        kind = cudaMemcpyKind::cudaMemcpyHostToHost;
    }

    cudaMemcpy2D(thrust::raw_pointer_cast(dst), dst_ld * sizeof(T),
                 thrust::raw_pointer_cast(src), src_ld * sizeof(T),
                 m * sizeof(T), n, kind);
}

// Explicitly instantiate the templates to allow for separate compilation
template void matrix_copy<thrust::device_ptr<float>, thrust::device_ptr<float>,
                          float>(thrust::device_ptr<float>, size_t,
                                 thrust::device_ptr<float>, size_t, size_t,
                                 size_t);
template void
    matrix_copy<thrust::device_ptr<double>, thrust::device_ptr<double>, double>(
        thrust::device_ptr<double>, size_t, thrust::device_ptr<double>, size_t,
        size_t, size_t);

template void matrix_copy<thrust::device_ptr<float>, float*, float>(
    thrust::device_ptr<float>, size_t, float*, size_t, size_t, size_t);
template void matrix_copy<thrust::device_ptr<double>, double*, double>(
    thrust::device_ptr<double>, size_t, double*, size_t, size_t, size_t);

}  // namespace matrix_ops
