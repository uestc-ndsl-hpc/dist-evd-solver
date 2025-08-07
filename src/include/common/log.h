#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/color.h>

#include <map>
#include <string>

#include "fmt/base.h"

namespace util {

inline const char* cublasGetErrorString(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
    }
    return "Unknown cublas status";
}

class Logger {
   public:
    static void init(bool verbose) { get()._verbose = verbose; }

    static void init_timer(bool print_time) { get()._print_time = print_time; }

    static bool is_verbose() { return get()._verbose; }

    static void print_environment_info() {
        int nDevices;
        cudaError_t err = cudaGetDeviceCount(&nDevices);
        if (err != cudaSuccess) {
            error("Failed to get CUDA device count: {}",
                  cudaGetErrorString(err));
            return;
        }

        println("--- GPU Environment Info ---");
        println("Number of CUDA devices: {}", nDevices);

        for (int i = 0; i < nDevices; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            println("  Device {}: {}", i, prop.name);
            println("    Compute Capability: {}.{}", prop.major, prop.minor);
            println("    Total Global Memory: {:.2f} GiB",
                    prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        }
        println("--------------------------");
    }

    template <typename S, typename... Args>
    static void println(const S& format_str, Args&&... args) {
        if (get()._verbose) {
            fmt::println(format_str, std::forward<Args>(args)...);
        }
    }

    template <typename... Args>
    static void error(const std::string& format_str, Args&&... args) {
        fmt::print(stderr, ("[ERROR] " + format_str + "\n").c_str(),
                   std::forward<Args>(args)...);
    }

    static void tic(const std::string& name) {
        auto it = get()._timers.find(name);
        if (it != get()._timers.end()) {
            cudaEventDestroy(it->second);
            get()._timers.erase(it);
        }
        cudaEvent_t start;
        cudaEventCreate(&start);
        cudaEventRecord(start, 0);
        get()._timers[name] = start;
    }

    static void toc(const std::string& name) {
        auto it = get()._timers.find(name);
        if (it == get()._timers.end()) {
            error("Timer '{}' not found for toc.", name);
            return;
        }

        cudaEvent_t start = it->second;
        cudaEvent_t stop;
        cudaEventCreate(&stop);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        if (get()._print_time) {
            fmt::println("[TIMER] {}: {:.4f} ms", name, milliseconds);
        } else {
            println("[TIMER] {}: {:.4f} ms", name, milliseconds);
        }

        cudaEventDestroy(stop);
    }

    static void toc(const std::string& name, float ops) {
        auto it = get()._timers.find(name);
        if (it == get()._timers.end()) {
            error("Timer '{}' not found for toc.", name);
            return;
        }

        cudaEvent_t start = it->second;
        cudaEvent_t stop;
        cudaEventCreate(&stop);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        if (get()._print_time) {
            fmt::println("[TIMER] {}: {:.4f} ms", name, milliseconds);
            auto tflops = (ops / 1e12f) / (milliseconds / 1000.0f);
            fmt::println("[TFLOPS] {}: {:.4f} TFLOPS", name, tflops);
        } else {
            println("[TIMER] {}: {:.4f} ms", name, milliseconds);
            auto tflops = (ops / 1e12f) / (milliseconds / 1000.0f);
            fmt::println("[TFLOPS] {}: {:.4f} TFLOPS", name, tflops);
        }
        cudaEventDestroy(stop);
    }

   private:
    Logger() = default;
    ~Logger() {
        for (auto const& [name, event] : _timers) {
            cudaEventDestroy(event);
        }
    }

    static Logger& get() {
        static Logger instance;
        return instance;
    }

    bool _verbose = false;
    std::map<std::string, cudaEvent_t> _timers;
    bool _print_time = false;
};

class MpiLogger {
   public:
    static void init(bool verbose, int rank) { 
        get()._verbose = verbose; 
        get()._rank = rank;
    }

    static void init_timer(bool print_time) { get()._print_time = print_time; }

    static bool is_verbose() { return get()._verbose; }

    static void print_environment_info() {
        // Only rank 0 should print environment info
        if (get()._rank != 0) return;
        
        int nDevices;
        cudaError_t err = cudaGetDeviceCount(&nDevices);
        if (err != cudaSuccess) {
            error("Failed to get CUDA device count: {}",
                  cudaGetErrorString(err));
            return;
        }

        println("--- GPU Environment Info ---");
        println("Number of CUDA devices: {}", nDevices);

        for (int i = 0; i < nDevices; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            println("  Device {}: {}", i, prop.name);
            println("    Compute Capability: {}.{}", prop.major, prop.minor);
            println("    Total Global Memory: {:.2f} GiB",
                    prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        }
        println("--------------------------");
    }

    template <typename S, typename... Args>
    static void println(const S& format_str, Args&&... args) {
        if (get()._verbose) {
            fmt::print(fmt::fg(fmt::color::yellow), "[rank: {}] ", get()._rank);
            fmt::println(format_str, std::forward<Args>(args)...);
        }
    }

    template <typename... Args>
    static void error(const std::string& format_str, Args&&... args) {
        fmt::print(stderr, fmt::fg(fmt::color::yellow), "[rank: {}] ", get()._rank);
        fmt::print(stderr, fmt::fg(fmt::color::red), "[ERROR] ");
        fmt::print(stderr, format_str + "\n", std::forward<Args>(args)...);
    }

    static void tic(const std::string& name) {
        auto it = get()._timers.find(name);
        if (it != get()._timers.end()) {
            cudaEventDestroy(it->second);
            get()._timers.erase(it);
        }
        cudaEvent_t start;
        cudaEventCreate(&start);
        cudaEventRecord(start, 0);
        get()._timers[name] = start;
    }

    static void toc(const std::string& name) {
        auto it = get()._timers.find(name);
        if (it == get()._timers.end()) {
            error("Timer '{}' not found for toc.", name);
            return;
        }

        cudaEvent_t start = it->second;
        cudaEvent_t stop;
        cudaEventCreate(&stop);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        if (get()._print_time) {
            fmt::print(fmt::fg(fmt::color::yellow), "[rank: {}] ", get()._rank);
            fmt::println("[TIMER] {}: {:.4f} ms", name, milliseconds);
        } else {
            println("[TIMER] {}: {:.4f} ms", name, milliseconds);
        }

        cudaEventDestroy(stop);
    }

    static void toc(const std::string& name, float ops) {
        auto it = get()._timers.find(name);
        if (it == get()._timers.end()) {
            error("Timer '{}' not found for toc.", name);
            return;
        }

        cudaEvent_t start = it->second;
        cudaEvent_t stop;
        cudaEventCreate(&stop);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        if (get()._print_time) {
            fmt::print(fmt::fg(fmt::color::yellow), "[rank: {}] ", get()._rank);
            fmt::println("[TIMER] {}: {:.4f} ms", name, milliseconds);
            auto tflops = (ops / 1e12f) / (milliseconds / 1000.0f);
            fmt::print(fmt::fg(fmt::color::yellow), "[rank: {}] ", get()._rank);
            fmt::println("[TFLOPS] {}: {:.4f} TFLOPS", name, tflops);
        } else {
            println("[TIMER] {}: {:.4f} ms", name, milliseconds);
            auto tflops = (ops / 1e12f) / (milliseconds / 1000.0f);
            fmt::print(fmt::fg(fmt::color::yellow), "[rank: {}] ", get()._rank);
            fmt::println("[TFLOPS] {}: {:.4f} TFLOPS", name, tflops);
        }
        cudaEventDestroy(stop);
    }

   private:
    MpiLogger() = default;
    ~MpiLogger() {
        for (auto const& [name, event] : _timers) {
            cudaEventDestroy(event);
        }
    }

    static MpiLogger& get() {
        static MpiLogger instance;
        return instance;
    }

    bool _verbose = false;
    std::map<std::string, cudaEvent_t> _timers;
    bool _print_time = false;
    int _rank = 0;
};

}  // namespace util