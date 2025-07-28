cmake_minimum_required(VERSION 3.25)

project(dist-evd-solver LANGUAGES CXX CUDA)

# --- 基本设置 ---
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "all")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)


# --- 外部依赖管理 (FetchContent) ---
include(FetchContent)

# 1. 声明 fmt 库
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG 11.2.0
)

# 2. 声明 argh 库
FetchContent_Declare(
    argh
    GIT_REPOSITORY https://github.com/adishavit/argh.git
    GIT_TAG v1.3.2
)

# 3. 声明 googletest 库
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
)

# 下载并使依赖可用 (将 googletest 添加到列表中)
FetchContent_MakeAvailable(fmt argh googletest)


# --- 核心依赖 ---
find_package(NVHPC REQUIRED)


# --- 项目库定义 ---
add_library(dist-evd-solver-lib
    src/matrix_ops/matrix_ops_print.cu
    src/matrix_ops/matrix_ops_gen.cu
    src/matrix_ops/matrix_ops_tsqr.cu
    src/matrix_ops/matrix_ops_sy2sb.cu
    src/matrix_ops/matrix_ops_copy.cu
    src/matrix_ops/matrix_ops_gemm.cu
    src/matrix_ops/matrix_ops_syr2k.cu
    src/matrix_ops/internal/sy2sb/sy2sb_panelqr.cu
    src/matrix_ops/dist/matrix_ops_dist_sy2sb.cu
)

target_include_directories(dist-evd-solver-lib
    PUBLIC
        src/include
        src/include/common
        src/include/matrix_ops/internal/sy2sb
)

find_package(OpenMP REQUIRED)

target_link_libraries(dist-evd-solver-lib
    PUBLIC
        NVHPC::CUDART
        NVHPC::CUSPARSE
        NVHPC::CUBLAS
        NVHPC::CUSOLVER
        NVHPC::CURAND
        NVHPC::NCCL
        fmt::fmt
        OpenMP::OpenMP_CXX
        argh
)

target_link_options(dist-evd-solver-lib PRIVATE "-Xcompiler=-fopenmp")
target_compile_options(dist-evd-solver-lib PRIVATE "-Xcompiler=-fopenmp")

# --- 可执行文件定义 ---
add_executable(dist-evd-solver
    src/main.cu
    src/workflow/workflow_sy2sb_dist.cu
)

target_link_options(dist-evd-solver PRIVATE "-Xcompiler=-fopenmp")
target_compile_options(dist-evd-solver PRIVATE "-Xcompiler=-fopenmp")

target_link_libraries(dist-evd-solver
    PRIVATE
        dist-evd-solver-lib
)

add_subdirectory(tests)