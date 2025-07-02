#pragma once

#include <cublas_v2.h>

#include <stdexcept>

namespace common {

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
     * @brief Get the underlying cuBLAS handle.
     * @return The raw cublasHandle_t.
     */
    cublasHandle_t get() const { return handle_; }

    /**
     * @brief Implicit conversion to the underlying cuBLAS handle.
     *
     * This allows the object to be used directly in cuBLAS API calls.
     */
    operator cublasHandle_t() const { return handle_; }

   private:
    cublasHandle_t handle_{nullptr};
};

}  // namespace common