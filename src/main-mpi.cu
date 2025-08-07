#include <mpi.h>
#include <cusolverDn.h>
#include <fmt/format.h>

#include <cstddef>

#include "argh.h"
#include "log.h"
#include "workflow.cuh"

int main(int argc, char** argv) {
    // Parse command line arguments first
    argh::parser cmdl(argv);

    const bool verbose = cmdl[{"-v", "--verbose"}];
    const bool print_time = cmdl[{"-t", "--time"}];
    const bool validate = cmdl[{"--validate"}];

    auto n = (size_t)4;
    cmdl({"-n", "--size"}, 4) >> n;
    auto m = n;
    cmdl({"-m", "--m"}, n) >> m;

    auto gpu_num = 2;
    cmdl({"-g", "--gpu-num"}, 2) >> gpu_num;

    // Initialize logger (will be used by all ranks, but main info printed only by rank 0)
    util::Logger::init(verbose);
    util::Logger::init_timer(print_time);

    if (cmdl[{"--double"}]) {
        run_workflow_sy2sb_mpi<double>(n, validate, gpu_num);
    } else if (cmdl[{"--float"}]) {
        run_workflow_sy2sb_mpi<float>(n, validate, gpu_num);
    } else {
        run_workflow_sy2sb_mpi<float>(n, validate, gpu_num);
    }

    return 0;
}
