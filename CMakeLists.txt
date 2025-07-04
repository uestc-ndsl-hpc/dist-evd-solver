cmake_minimum_required(VERSION 3.25)

project(dist-evd-solver LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES "native")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(NVHPC REQUIRED COMPONENTS MATH CUDA)

add_subdirectory(fmt)
add_subdirectory(argh)

add_executable(dist-evd-solver
        src/main.cu
        src/matrix_ops/matrix_ops_print.cu
        src/matrix_ops/matrix_ops_gen.cu
        src/matrix_ops/matrix_ops_tsqr.cu
        src/workflow/workflow_tsqr.cu)

target_include_directories(dist-evd-solver
        PRIVATE
        src/include)

target_link_libraries(dist-evd-solver
        PRIVATE
        NVHPC::CUDART
        NVHPC::CUSPARSE
        NVHPC::CUBLAS
        NVHPC::CUSOLVER
        NVHPC::CURAND
        fmt::fmt
        argh)