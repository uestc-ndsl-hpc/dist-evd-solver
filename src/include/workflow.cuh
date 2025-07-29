#pragma once

#include <cstddef>

template <typename T>
void run_workflow_tsqr(size_t m, size_t n, bool validate); 

template <typename T>
void run_workflow_sy2sb(size_t n, bool validate);

template <typename T>
void run_workflow_sy2sb_dist(size_t n, bool validate, int gpu_num = 1);