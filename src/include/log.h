#pragma once

#include <fmt/core.h>
#include <fmt/format.h>
#include <string>
#include <map>
#include <cuda_runtime.h>

namespace util {

class Logger {
   public:
    static void init(bool verbose) { get()._verbose = verbose; }

    static bool is_verbose() { return get()._verbose; }

    template <typename S, typename... Args>
    static void println(const S& format_str, Args&&... args) {
        if (get()._verbose) {
            fmt::println(format_str, std::forward<Args>(args)...);
        }
    }

    template <typename... Args>
    static void error(const std::string& format_str, Args&&... args) {
        fmt::print(stderr, ("[ERROR] " + format_str + "\n").c_str(), std::forward<Args>(args)...);
    }
    
    static void tic(const std::string& name) {
        cudaEvent_t start;
        cudaEventCreate(&start);
        cudaEventRecord(start, 0);
        get()._timers[name] = start;
    }

    static void toc(const std::string& name, const bool print_time = true) {
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
        
        if (print_time) {
            fmt::println("[TIMER] {}: {:.4f} ms", name, milliseconds);
        } else {
            println("[TIMER] {}: {:.4f} ms", name, milliseconds);
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        get()._timers.erase(it);
    }

   private:
    Logger() = default;

    static Logger& get() {
        static Logger instance;
        return instance;
    }

    bool _verbose = false;
    std::map<std::string, cudaEvent_t> _timers;
};

}  // namespace util