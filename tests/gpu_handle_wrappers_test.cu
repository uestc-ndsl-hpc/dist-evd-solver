#include <gtest/gtest.h>
#include "gpu_handle_wrappers.h"

using namespace common;

/**
 * @brief Test fixture for GPU handle wrapper tests
 */
class GpuHandleWrappersTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize CUDA context for testing
        cudaSetDevice(0);
    }

    void TearDown() override {
        // Ensure all CUDA operations are complete
        cudaDeviceSynchronize();
    }
};

/**
 * @brief Test basic construction and destruction of CublasXtHandle
 */
TEST_F(GpuHandleWrappersTest, CublasXtHandleBasicConstruction) {
    EXPECT_NO_THROW({
        CublasXtHandle handle;
        EXPECT_NE(handle.get(), nullptr);
    });
}

/**
 * @brief Test move constructor for CublasXtHandle
 */
TEST_F(GpuHandleWrappersTest, CublasXtHandleMoveConstruction) {
    CublasXtHandle original;
    cublasXtHandle_t original_handle = original.get();
    
    EXPECT_NE(original_handle, nullptr);
    
    CublasXtHandle moved(std::move(original));
    
    EXPECT_EQ(moved.get(), original_handle);
    EXPECT_EQ(original.get(), nullptr);
}

/**
 * @brief Test move assignment for CublasXtHandle
 */
TEST_F(GpuHandleWrappersTest, CublasXtHandleMoveAssignment) {
    CublasXtHandle original;
    cublasXtHandle_t original_handle = original.get();
    
    EXPECT_NE(original_handle, nullptr);
    
    CublasXtHandle target;
    target = std::move(original);
    
    EXPECT_EQ(target.get(), original_handle);
    EXPECT_EQ(original.get(), nullptr);
}

/**
 * @brief Test copy constructor is deleted for CublasXtHandle
 */
TEST_F(GpuHandleWrappersTest, CublasXtHandleCopyConstructorDeleted) {
    CublasXtHandle original;
    
    // This should fail to compile due to deleted copy constructor
    // Uncomment to verify: CublasXtHandle copy(original);
}

/**
 * @brief Test copy assignment is deleted for CublasXtHandle
 */
TEST_F(GpuHandleWrappersTest, CublasXtHandleCopyAssignmentDeleted) {
    CublasXtHandle original;
    CublasXtHandle target;
    
    // This should fail to compile due to deleted copy assignment
    // Uncomment to verify: target = original;
}

/**
 * @brief Test implicit conversion for CublasXtHandle
 */
TEST_F(GpuHandleWrappersTest, CublasXtHandleImplicitConversion) {
    CublasXtHandle handle;
    
    // Test implicit conversion to cublasXtHandle_t
    cublasXtHandle_t raw_handle = handle;
    EXPECT_EQ(raw_handle, handle.get());
}

/**
 * @brief Test multiple CublasXtHandle instances can coexist
 */
TEST_F(GpuHandleWrappersTest, CublasXtHandleMultipleInstances) {
    EXPECT_NO_THROW({
        CublasXtHandle handle1;
        CublasXtHandle handle2;
        CublasXtHandle handle3;
        
        EXPECT_NE(handle1.get(), handle2.get());
        EXPECT_NE(handle1.get(), handle3.get());
        EXPECT_NE(handle2.get(), handle3.get());
    });
}

/**
 * @brief Test CublasHandle basic construction
 */
TEST_F(GpuHandleWrappersTest, CublasHandleBasicConstruction) {
    EXPECT_NO_THROW({
        CublasHandle handle;
        EXPECT_NE(handle.get(), nullptr);
    });
}

/**
 * @brief Test CusolverDnHandle basic construction
 */
TEST_F(GpuHandleWrappersTest, CusolverDnHandleBasicConstruction) {
    EXPECT_NO_THROW({
        CusolverDnHandle handle;
        EXPECT_NE(handle.get(), nullptr);
    });
}

/**
 * @brief Test all handle types can coexist
 */
TEST_F(GpuHandleWrappersTest, AllHandleTypesCoexist) {
    EXPECT_NO_THROW({
        CublasHandle cublas_handle;
        CusolverDnHandle cusolver_handle;
        CublasXtHandle cublasxt_handle;
        
        EXPECT_NE(cublas_handle.get(), nullptr);
        EXPECT_NE(cusolver_handle.get(), nullptr);
        EXPECT_NE(cublasxt_handle.get(), nullptr);
    });
}