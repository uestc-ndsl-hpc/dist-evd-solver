#include <mpi.h>
#include <cusolverDn.h>
#include <fmt/format.h>

#include <cstddef>
#include <cstdlib>

#include "argh.h"
#include "log.h"
#include "workflow.cuh"

int main(int argc, char** argv) {
    // Parse command line arguments first
    argh::parser cmdl(argv);

    // Show help if requested
    if (cmdl[{"-h", "--help"}]) {
        fmt::print("Usage: {} [options]\n"
                  "Options:\n"
                  "  -n, --size <size>      Matrix size (default: 4)\n"
                  "  -m, --m <size>         Secondary matrix size (default: same as n)\n"
                  "  -g, --gpu-num <num>    Number of GPUs (default: 2)\n"
                  "  -nb, --nb <size>       Block size for recursion (default: 64)\n"
                  "  -b, --b <size>         Panel size (default: 16)\n"
                  "  --dist <mode>          Distribution: blockwise | cyclic (default: blockwise)\n"
                  "  -v, --verbose          Enable verbose output\n"
                  "  -t, --time             Enable timing output\n"
                  "  --validate             Enable result validation\n"
                  "  --double               Use double precision\n"
                  "  --float                Use single precision (default)\n"
                  "  -h, --help             Show this help message\n", argv[0]);
        return 0;
    }

    const bool verbose = cmdl[{"-v", "--verbose"}];
    const bool print_time = cmdl[{"-t", "--time"}];
    const bool validate = cmdl[{"--validate"}];

    auto n = (size_t)4;
    cmdl({"-n", "--size"}, 4) >> n;
    auto m = n;
    cmdl({"-m", "--m"}, n) >> m;

    auto gpu_num = 2;
    cmdl({"-g", "--gpu-num"}, 2) >> gpu_num;

    auto nb = (size_t)64;
    cmdl({"-nb", "--nb"}, 64) >> nb;

    auto b = (size_t)16;
    cmdl({"-b", "--b"}, 16) >> b;

    // Optional distribution selection (propagated via env var understood by sy2sb)
    std::string dist;
    cmdl({"--dist"}, std::string()) >> dist;
    if (!dist.empty()) {
        // Normalize aliases
        if (dist == "cyclic" || dist == "blockcyclic" || dist == "bc") {
            setenv("EVD_DIST", "cyclic", 1);
        } else if (dist == "blockwise" || dist == "bw") {
            setenv("EVD_DIST", "blockwise", 1);
        }
    }

    // Initialize logger (will be used by all ranks, but main info printed only by rank 0)
    util::Logger::init(verbose);
    util::Logger::init_timer(print_time);

    if (cmdl[{"--double"}]) {
        run_workflow_sy2sb_mpi<double>(n, validate, gpu_num, nb, b);
    } else if (cmdl[{"--float"}]) {
        run_workflow_sy2sb_mpi<float>(n, validate, gpu_num, nb, b);
    } else {
        run_workflow_sy2sb_mpi<float>(n, validate, gpu_num, nb, b);
    }

    return 0;
}
