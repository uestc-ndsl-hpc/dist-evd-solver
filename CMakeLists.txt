cmake_minimum_required(VERSION 3.22)

project(dist-evd-solver LANGUAGES CXX CUDA)

# --- 基本设置 ---
# CUDA architectures
# NVSHMEM and some CUDA features used here (system-scope atomics, cooperative groups reduce, etc.)
# require newer SMs. Avoid compiling for legacy architectures that will fail to build.
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "" OR CMAKE_CUDA_ARCHITECTURES STREQUAL "all")
    # Default to Ampere+; adjust via -DCMAKE_CUDA_ARCHITECTURES=... if needed
    set(CMAKE_CUDA_ARCHITECTURES "80;90")
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
# FetchContent_Declare(
#     googletest
#     GIT_REPOSITORY https://github.com/google/googletest.git
#     GIT_TAG v1.17.0
# )

# 下载并使依赖可用 (将 googletest 添加到列表中)
# FetchContent_MakeAvailable(fmt argh googletest)
FetchContent_MakeAvailable(fmt argh)


# --- 核心依赖 ---
find_package(NVHPC REQUIRED)

find_package(OpenMP REQUIRED)

find_package(MPI REQUIRED)

find_package(MKL REQUIRED)

# --- 项目库定义 ---
add_library(dist-evd-solver-lib
    src/matrix_ops/matrix_ops_print.cu
    src/matrix_ops/matrix_ops_gen.cu
    src/matrix_ops/matrix_ops_tsqr.cu
    src/matrix_ops/matrix_ops_sy2sb.cu
    src/matrix_ops/matrix_ops_copy.cu
    src/matrix_ops/matrix_ops_gemm.cu
    src/matrix_ops/matrix_ops_sb2tr.cu
    src/matrix_ops/matrix_ops_syr2k.cu
    src/matrix_ops/matrix_ops_tr2sb.cu
    src/matrix_ops/internal/sy2sb/sy2sb_panelqr.cu
    src/matrix_ops/dist/matrix_ops_dist_sy2sb.cu
    src/matrix_ops/mpi/matrix_ops_mpi_sy2sb.cu
    src/matrix_ops/mpi/matrix_ops_mpi_sb2sy.cu
    src/matrix_ops/mpi/matrix_ops_mpi_sb2tr.cu
    src/matrix_ops/mpi/matrix_ops_mpi_tr2sb.cu
)

target_include_directories(dist-evd-solver-lib
    PUBLIC
        src/include
        src/include/common
        src/include/matrix_ops/internal/sy2sb
        src/include/matrix_ops
    PRIVATE
        ${MPI_CXX_INCLUDE_DIRS}
)


target_link_libraries(dist-evd-solver-lib
    PUBLIC
        NVHPC::CUDART
        NVHPC::CUSPARSE
        NVHPC::CUBLAS
        NVHPC::CUSOLVER
        NVHPC::CURAND
        NVHPC::NCCL
        NVHPC::NVSHMEM
        fmt::fmt
        OpenMP::OpenMP_CXX
        MPI::MPI_CXX
        argh
        MKL::MKL
)

# 为库目标设置分离编译
set_property(TARGET dist-evd-solver-lib PROPERTY CUDA_SEPARABLE_COMPILATION ON)
# 对于静态库，不解析设备符号，让最终的可执行文件来解析
set_property(TARGET dist-evd-solver-lib PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)


# 分别为 CUDA 和 CXX 设置编译选项
target_compile_options(dist-evd-solver-lib PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>
    $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
    ${MPI_CXX_COMPILE_FLAGS}
)

# 链接选项只用于 CXX，不用 CUDA 特有的选项
target_link_options(dist-evd-solver-lib PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
)

# --- MPI 可执行文件定义 ---
add_executable(dist-evd-solver-mpi
    src/main-mpi.cu
    src/workflow/workflow_tr2sb_mpi.cu
)

target_link_libraries(dist-evd-solver-mpi
    PRIVATE
        dist-evd-solver-lib
)

set_property(TARGET dist-evd-solver-lib PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET dist-evd-solver-lib PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS OFF)

target_compile_options(dist-evd-solver-mpi PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
    $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>
    $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
    ${MPI_CXX_COMPILE_FLAGS}
)

target_link_options(dist-evd-solver-mpi PRIVATE 
    $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
)

# add_subdirectory(tests)