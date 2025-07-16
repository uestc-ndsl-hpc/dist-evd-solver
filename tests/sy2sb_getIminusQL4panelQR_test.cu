#include <gpu_handle_wrappers.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "matrix_ops.cuh"
#include "sy2sb_panelqr.cuh"

// CPU implementation of LU factorization (Doolittle, no pivoting)
// for a m x n matrix. The result is stored in-place in A.
// A is column-major.
template <typename T>
void cpu_getrf_no_piv(size_t m, size_t n, std::vector<T>& A, size_t lda) {
    for (size_t k = 0; k < std::min(m, n); ++k) {
        for (size_t i = k + 1; i < m; ++i) {
            A[k * lda + i] /= A[k * lda + k];
        }
        for (size_t j = k + 1; j < n; ++j) {
            for (size_t i = k + 1; i < m; ++i) {
                A[j * lda + i] -= A[k * lda + i] * A[j * lda + k];
            }
        }
    }
}

template <typename T>
class GetIminusQL4panelQRTest : public ::testing::Test {
   protected:
    void SetUp() override {
        handle_ = std::make_unique<common::CusolverDnHandle>();
    }

    void TearDown() override { handle_.reset(); }

    void run_test(size_t m, size_t n, size_t lda) {
        auto d_A = matrix_ops::create_normal_random<T>(m, n, lda);
        thrust::host_vector<T> h_A = d_A;

        // Make matrix diagonally dominant to ensure getrf without pivoting is
        // stable
        for (size_t i = 0; i < std::min(m, n); ++i) {
            h_A[i * lda + i] += static_cast<T>(2.0 * std::max(m, n));
        }
        d_A = h_A;

        thrust::host_vector<T> h_A_original = h_A;

        auto A_ptr = d_A.data();
        matrix_ops::internal::sy2sb::getIminusQL4panelQR<T>(*handle_, m, n,
                                                           A_ptr, lda);

        thrust::host_vector<T> h_result = d_A;

        std::vector<T> h_A_vec(h_A_original.begin(), h_A_original.end());
        cpu_getrf_no_piv<T>(m, n, h_A_vec, lda);

        thrust::host_vector<T> h_expected(lda * n);
        // The iterator in getIminusQL4panelQR goes up to lda * n.
        thrust::transform(
            thrust::host, h_A_vec.begin(), h_A_vec.end(), h_expected.begin(),
            [=](T) { return static_cast<T>(0.0); });

        for (size_t j = 0; j < n; ++j) {
            for (size_t i = 0; i < m; ++i) {
                if (i > j) {
                    h_expected[j * lda + i] = h_A_vec[j * lda + i];
                } else if (i == j) {
                    h_expected[j * lda + i] = static_cast<T>(1.0);
                } else {  // i < j
                    h_expected[j * lda + i] = static_cast<T>(0.0);
                }
            }
        }

        for (size_t i = 0; i < h_result.size(); ++i) {
            ASSERT_NEAR(h_result[i], h_expected[i], 1e-5);
        }
    }

    std::unique_ptr<common::CusolverDnHandle> handle_;
};

using MyTypes = ::testing::Types<float, double>;

TYPED_TEST_SUITE(GetIminusQL4panelQRTest, MyTypes);

TYPED_TEST(GetIminusQL4panelQRTest, Square) { this->run_test(16, 16, 16); }

TYPED_TEST(GetIminusQL4panelQRTest, SquareLda) { this->run_test(17, 17, 20); }

TYPED_TEST(GetIminusQL4panelQRTest, Tall) { this->run_test(32, 16, 32); }

TYPED_TEST(GetIminusQL4panelQRTest, TallLda) { this->run_test(32, 16, 40); } 