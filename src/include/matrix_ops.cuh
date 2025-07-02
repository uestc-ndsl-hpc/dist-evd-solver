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
#include <cublas_v2.h>

#include <cstddef>
#include <stdexcept>
#include <string>

namespace matrix_ops {

/**
 * @class CublasHandle
 * @brief A RAII wrapper for a cuBLAS handle.
 *
 * This class ensures that a cuBLAS handle is properly created and destroyed.
 * It follows the Rule of Five: move semantics are enabled, but copy
 * semantics are disabled to prevent double-destruction of the handle.
 */
class CublasHandle {
 public:
  CublasHandle() {
    if (cublasCreate(&handle_) != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error("Failed to create cuBLAS handle.");
    }
  }

  ~CublasHandle() {
    if (handle_) {
      // It's generally safe to call cublasDestroy on a valid handle.
      // We don't need to check the return status in a destructor.
      cublasDestroy(handle_);
    }
  }

  // Disable copy constructor and copy assignment operator
  CublasHandle(const CublasHandle&) = delete;
  CublasHandle& operator=(const CublasHandle&) = delete;

  // Enable move constructor
  CublasHandle(CublasHandle&& other) noexcept : handle_(other.handle_) {
    other.handle_ = nullptr;
  }

  // Enable move assignment operator
  CublasHandle& operator=(CublasHandle&& other) noexcept {
    if (this != &other) {
      if (handle_) {
        cublasDestroy(handle_);
      }
      handle_ = other.handle_;
      other.handle_ = nullptr;
    }
    return *this;
  }

  /**
   * @brief Implicit conversion to the underlying cuBLAS handle.
   *
   * This allows the object to be used directly in cuBLAS API calls.
   */
  operator cublasHandle_t() const { return handle_; }

 private:
  cublasHandle_t handle_{nullptr};
};

template <typename T>
void print(const thrust::device_vector<T>& d_vec, size_t n,
           const std::string& title);

template <typename T>
void print(const thrust::device_vector<T>& d_vec, size_t m, size_t n,
           const std::string& title);

template <typename T>
thrust::device_vector<T> create_symmetric_random(size_t n);

template <typename T>
thrust::device_vector<T> create_uniform_random(size_t n);

template <typename T>
thrust::device_vector<T> create_uniform_random(size_t m, size_t n);

template <typename T>
thrust::device_vector<T> create_normal_random(size_t n, T mean = 0.0,
                                              T stddev = 1.0);

template <typename T>
thrust::device_vector<T> create_normal_random(size_t m, size_t n, T mean = 0.0,
                                              T stddev = 1.0);

/**
 * @brief In-place Tall-and-Skinny QR decomposition on a single GPU.
 *
 * This function computes the QR decomposition of a tall-and-skinny matrix A.
 * The input matrix A is overwritten by the orthogonal matrix Q.
 *
 * @tparam T The data type of the matrix elements (e.g., float, double).
 * @param handle A handle to the cuBLAS library context.
 * @param m The number of rows of matrix A.
 * @param n The number of columns of matrix A.
 * @param A_inout On input, the m x n matrix A. On output, the m x n orthogonal matrix Q.
 * @param R On output, the n x n upper triangular matrix R.
 */
template <typename T>
void tsqr(const CublasHandle& handle, size_t m, size_t n,
          thrust::device_vector<T>& A_inout, thrust::device_vector<T>& R);

}  // namespace matrix_ops