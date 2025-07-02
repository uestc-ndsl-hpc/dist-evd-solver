#pragma once

#include <cuda_runtime.h>
#include <fmt/core.h>
#include <fmt/format.h>

#include <map>
#include <string>

namespace util {

class Logger {
   public:
    static void init(bool verbose) { get()._verbose = verbose; }

    static void init_timer(bool print_time) { get()._print_time = print_time; }

    static bool is_verbose() { return get()._verbose; }

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

}  // namespace util