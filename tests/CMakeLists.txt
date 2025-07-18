# --- 测试目标配置 ---
enable_testing()

add_executable(matrix_ops_gemm_test matrix_ops_gemm_test.cu)

target_link_libraries(matrix_ops_gemm_test
    PRIVATE
        dist-evd-solver-lib
        GTest::gtest
        GTest::gtest_main
)

add_executable(matrix_ops_print_test matrix_ops_print_test.cu)

target_link_libraries(matrix_ops_print_test
    PRIVATE
        dist-evd-solver-lib
        GTest::gtest
        GTest::gtest_main
)


add_test(NAME matrix_ops_gemm_test COMMAND matrix_ops_gemm_test)
add_test(NAME matrix_ops_print_test COMMAND matrix_ops_print_test)


add_executable(matrix_ops_syr2k_test matrix_ops_syr2k_test.cu)

target_link_libraries(matrix_ops_syr2k_test
    PRIVATE
        dist-evd-solver-lib
        GTest::gtest
        GTest::gtest_main
)

add_test(NAME matrix_ops_syr2k_test COMMAND matrix_ops_syr2k_test)


add_executable(matrix_ops_tsqr_test matrix_ops_tsqr_test.cu)

target_link_libraries(matrix_ops_tsqr_test
    PRIVATE
        dist-evd-solver-lib
        GTest::gtest
        GTest::gtest_main
)

add_test(NAME matrix_ops_tsqr_test COMMAND matrix_ops_tsqr_test)

add_executable(matrix_ops_sy2sb_test matrix_ops_sy2sb_test.cu)

target_link_libraries(matrix_ops_sy2sb_test
    PRIVATE
        dist-evd-solver-lib
        GTest::gtest
        GTest::gtest_main
)

add_test(NAME matrix_ops_sy2sb_test COMMAND matrix_ops_sy2sb_test)

add_executable(matrix_ops_IminusA_test matrix_ops_IminusA_test.cu)

target_link_libraries(matrix_ops_IminusA_test
    PRIVATE
        dist-evd-solver-lib
        GTest::gtest
        GTest::gtest_main
)

add_executable(matrix_ops_copy_test matrix_ops_copy_test.cu)

target_link_libraries(matrix_ops_copy_test
    PRIVATE
        dist-evd-solver-lib
        GTest::gtest
        GTest::gtest_main
)

add_test(NAME matrix_ops_copy_test COMMAND matrix_ops_copy_test)

add_executable(sy2sb_getIminusQL4panelQR_test sy2sb_getIminusQL4panelQR_test.cu)

target_link_libraries(sy2sb_getIminusQL4panelQR_test
    PRIVATE
        dist-evd-solver-lib
        GTest::gtest
        GTest::gtest_main
)

target_include_directories(sy2sb_getIminusQL4panelQR_test PRIVATE ${CMAKE_SOURCE_DIR}/src/matrix_ops/internal/sy2sb)

add_test(NAME sy2sb_getIminusQL4panelQR_test COMMAND sy2sb_getIminusQL4panelQR_test)

add_executable(sy2sb_WY_construction_test sy2sb_WY_construction_test.cu)

target_link_libraries(sy2sb_WY_construction_test
    PRIVATE
        dist-evd-solver-lib
        GTest::gtest
        GTest::gtest_main
)

target_include_directories(sy2sb_WY_construction_test PRIVATE ${CMAKE_SOURCE_DIR}/src/matrix_ops/internal/sy2sb)

add_test(NAME sy2sb_WY_construction_test COMMAND sy2sb_WY_construction_test)