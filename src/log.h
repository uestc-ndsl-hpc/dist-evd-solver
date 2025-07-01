#pragma once

#include <fmt/core.h>
#include <fmt/format.h>

#include <utility>

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

   private:
    Logger() = default;

    static Logger& get() {
        static Logger instance;
        return instance;
    }

    bool _verbose = false;
};

}  // namespace util