#include <fmt/format.h>
#include <cstddef>

#include "argh.h"
#include "log.h"
#include "matrix_ops.cuh"

template <typename T>
void run_workflow(int n) {
    if constexpr (std::is_same_v<T, float>) {
        util::Logger::println("Using float precision");
    } else {
        util::Logger::println("Using double precision");
    }

    auto C = matrix_ops::create_symmetric_random<T>(n);

    if (util::Logger::is_verbose()) {
        matrix_ops::print(C, n, "Final Symmetric Matrix C");
    }
}

int main(int argc, char** argv) {
    argh::parser cmdl(argv);

    const bool verbose = cmdl[{"-v", "--verbose"}];
    util::Logger::init(verbose);
    const bool print_time = cmdl[{"-t", "--time"}];
    util::Logger::init_timer(print_time);
    util::Logger::println("Starting dist-evd-solver");

    auto n = (size_t) 4;
    cmdl({"-n", "--size"}, 4) >> n;

    if (cmdl[{"--double"}]) {
        util::Logger::println("Using double precision");
        run_workflow<double>(n);
    } else if (cmdl[{"--float"}]) {
        util::Logger::println("Using single precision");
        run_workflow<float>(n);
    } else {
        util::Logger::println("Using default precision");
        run_workflow<float>(n);
    }

    return 0;
}