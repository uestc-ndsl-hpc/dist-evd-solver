#pragma once

#include <cstddef>

template <typename T>
void run_workflow_tsqr(size_t m, size_t n, bool validate); 

template <typename T>
void run_workflow_sy2sb(size_t n, bool validate);

template <typename T>
void run_workflow_sy2sb_dist(size_t n, bool validate, int gpu_num = 1);

template <typename T>
void run_workflow_sy2sb_mpi(size_t n, bool validate, int num_gpus = 1, size_t nb = 64, size_t b = 16, bool debug = false);

template <typename T>
void run_workflow_sb2sy_mpi(size_t n, bool validate, int num_gpus = 1, size_t nb = 64, size_t b = 16, bool debug = false);