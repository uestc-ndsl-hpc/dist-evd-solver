#pragma once

#include <cstddef>

#include "gpu_handle_wrappers.h"

namespace matrix_ops {
namespace dist {
template <typename T>
void sy2sb(const common::CublasHandle& handle, size_t n, T* A, size_t lda, T* W,
           size_t ldw, T* Y, size_t ldy, size_t nb = 1024, size_t b = 32);
}  // namespace dist
}  // namespace matrix_ops