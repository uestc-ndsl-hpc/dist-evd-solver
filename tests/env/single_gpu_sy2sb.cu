#include <cusolverDn.h>
#include <fmt/format.h>

#include <cstddef>

#include "argh.h"
#include "log.h"
#include "workflow.cuh"

int main(int argc, char** argv) {
    argh::parser cmdl(argv);

    const bool verbose = cmdl[{"-v", "--verbose"}];
    util::Logger::init(verbose);
    const bool print_time = cmdl[{"-t", "--time"}];
    util::Logger::init_timer(print_time);
    util::Logger::print_environment_info();
    util::Logger::println("Starting dist-evd-solver");

    const bool validate = cmdl[{"--validate"}];

    auto n = (size_t)4;
    cmdl({"-n", "--size"}, 4) >> n;
    auto m = n;
    cmdl({"-m", "--m"}, n) >> m;

    if (cmdl[{"--double"}]) {
        util::Logger::println("Using double precision");
        run_workflow_sy2sb<double>(n, validate);
    } else if (cmdl[{"--float"}]) {
        util::Logger::println("Using single precision");
        run_workflow_sy2sb<float>(n, validate);
    } else {
        util::Logger::println("Using default precision");
        run_workflow_sy2sb<float>(n, validate);
    }

    return 0;
}