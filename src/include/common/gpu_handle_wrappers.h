#pragma once

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cublasXt.h>
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

/**
 * @class CusolverDnHandle
 * @brief A RAII wrapper for a cuSOLVER DN handle.
 *
 * This class ensures that a cuSOLVER DN handle is properly created and
 * destroyed. It follows the Rule of Five: move semantics are enabled, but copy
 * semantics are disabled to prevent double-destruction of the handle.
 */
class CusolverDnHandle {
   public:
    CusolverDnHandle() {
        if (cusolverDnCreate(&handle_) != CUSOLVER_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuSOLVER DN handle.");
        }
    }

    ~CusolverDnHandle() {
        if (handle_) {
            // It's generally safe to call cusolverDnDestroy on a valid handle.
            // We don't need to check the return status in a destructor.
            cusolverDnDestroy(handle_);
        }
    }

    // Disable copy constructor and copy assignment operator
    CusolverDnHandle(const CusolverDnHandle&) = delete;
    CusolverDnHandle& operator=(const CusolverDnHandle&) = delete;

    // Enable move constructor
    CusolverDnHandle(CusolverDnHandle&& other) noexcept
        : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    // Enable move assignment operator
    CusolverDnHandle& operator=(CusolverDnHandle&& other) noexcept {
        if (this != &other) {
            if (handle_) {
                cusolverDnDestroy(handle_);
            }
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief Get the underlying cuSOLVER DN handle.
     * @return The raw cusolverDnHandle_t.
     */
    cusolverDnHandle_t get() const { return handle_; }

    /**
     * @brief Implicit conversion to the underlying cuSOLVER DN handle.
     *
     * This allows the object to be used directly in cuSOLVER DN API calls.
     */
    operator cusolverDnHandle_t() const { return handle_; }

   private:
    cusolverDnHandle_t handle_{nullptr};
};

/**
 * @class CublasXtHandle
 * @brief A RAII wrapper for a cuBLAS Xt handle.
 *
 * This class ensures that a cuBLAS Xt handle is properly created and destroyed.
 * It follows the Rule of Five: move semantics are enabled, but copy
 * semantics are disabled to prevent double-destruction of the handle.
 */
class CublasXtHandle {
   public:
    CublasXtHandle() {
        if (cublasXtCreate(&handle_) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create cuBLAS Xt handle.");
        }
    }

    ~CublasXtHandle() {
        if (handle_) {
            // It's generally safe to call cublasXtDestroy on a valid handle.
            // We don't need to check the return status in a destructor.
            cublasXtDestroy(handle_);
        }
    }

    // Disable copy constructor and copy assignment operator
    CublasXtHandle(const CublasXtHandle&) = delete;
    CublasXtHandle& operator=(const CublasXtHandle&) = delete;

    // Enable move constructor
    CublasXtHandle(CublasXtHandle&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    // Enable move assignment operator
    CublasXtHandle& operator=(CublasXtHandle&& other) noexcept {
        if (this != &other) {
            if (handle_) {
                cublasXtDestroy(handle_);
            }
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief Get the underlying cuBLAS Xt handle.
     * @return The raw cublasXtHandle_t.
     */
    cublasXtHandle_t get() const { return handle_; }

    /**
     * @brief Implicit conversion to the underlying cuBLAS Xt handle.
     *
     * This allows the object to be used directly in cuBLAS Xt API calls.
     */
    operator cublasXtHandle_t() const { return handle_; }

   private:
    cublasXtHandle_t handle_{nullptr};
};

}  // namespace common