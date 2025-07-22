#include <iostream>
#include <cuda_runtime.h>

// 一个健壮的 CUDA 错误检查宏
#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
        exit(EXIT_FAILURE); \
    } \
}

int main() {
    // --- 1. 检查系统 ---
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount < 2) {
        std::cerr << "Error: Need at least 2 GPUs to run this P2P test." << std::endl;
        return EXIT_FAILURE;
    }
    std::cout << "Found " << deviceCount << " GPUs." << std::endl;

    // 我们将使用 GPU 0 和 GPU 1
    int gpu0 = 0;
    int gpu1 = 2;

    // --- 2. 检查 P2P 能力 ---
    int canAccessPeer = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&canAccessPeer, gpu0, gpu1));

    if (canAccessPeer) {
        std::cout << "GPU " << gpu0 << " and GPU " << gpu1 << " support Peer-to-Peer access." << std::endl;
    } else {
        std::cerr << "Error: P2P access is not supported between GPU " << gpu0 << " and GPU " << gpu1 << "." << std::endl;
        return EXIT_FAILURE;
    }

    // --- 3. 启用 P2P ---
    // 为了让 gpu0 能访问 gpu1，我们必须在 gpu0 的上下文中启用它
    CUDA_CHECK(cudaSetDevice(gpu0));
    CUDA_CHECK(cudaDeviceEnablePeerAccess(gpu1, 0)); // flags 通常为 0
    std::cout << "Enabled P2P access from GPU " << gpu0 << " to GPU " << gpu1 << "." << std::endl;

    // 同理，也启用反向访问，这在很多复杂应用中是好习惯
    CUDA_CHECK(cudaSetDevice(gpu1));
    CUDA_CHECK(cudaDeviceEnablePeerAccess(gpu0, 0));
    std::cout << "Enabled P2P access from GPU " << gpu1 << " to GPU " << gpu0 << "." << std::endl;


    // --- 4. 分配内存和准备数据 ---
    size_t dataSize = 1024 * 1024 * sizeof(float); // 4MB
    float *h_data = (float*)malloc(dataSize);
    if (h_data == nullptr) {
        std::cerr << "Failed to allocate host memory." << std::endl;
        return EXIT_FAILURE;
    }

    // 在 Host 端初始化数据，以便后续验证
    for (size_t i = 0; i < dataSize / sizeof(float); ++i) {
        h_data[i] = 3.14159f;
    }
    std::cout << "Host data initialized." << std::endl;

    // 在 GPU 0 上分配显存
    float *d_data_0;
    CUDA_CHECK(cudaSetDevice(gpu0));
    CUDA_CHECK(cudaMalloc(&d_data_0, dataSize));

    // 在 GPU 1 上分配显存
    float *d_data_1;
    CUDA_CHECK(cudaSetDevice(gpu1));
    CUDA_CHECK(cudaMalloc(&d_data_1, dataSize));
    std::cout << "Device memory allocated on both GPUs." << std::endl;


    // --- 5. 传输数据 ---
    // 步骤 A: Host -> GPU 0
    CUDA_CHECK(cudaSetDevice(gpu0));
    CUDA_CHECK(cudaMemcpy(d_data_0, h_data, dataSize, cudaMemcpyHostToDevice));
    std::cout << "Data copied from Host to GPU " << gpu0 << "." << std::endl;

    // 步骤 B: GPU 0 -> GPU 1 (核心的 P2P 步骤!)
    // 使用 cudaMemcpyPeer()，注意参数需要指明源和目标的设备ID
    std::cout << "Performing Peer-to-Peer copy from GPU " << gpu0 << " to GPU " << gpu1 << "..." << std::endl;
    CUDA_CHECK(cudaMemcpyPeer(d_data_1, gpu1, d_data_0, gpu0, dataSize));
    
    // 步骤 C: GPU 1 -> Host (用于验证)
    // 创建一个新的 host buffer 来接收结果
    float *h_result = (float*)malloc(dataSize);
    if (h_result == nullptr) {
        std::cerr << "Failed to allocate host memory for result." << std::endl;
        return EXIT_FAILURE;
    }
    CUDA_CHECK(cudaSetDevice(gpu1));
    CUDA_CHECK(cudaMemcpy(h_result, d_data_1, dataSize, cudaMemcpyDeviceToHost));
    std::cout << "Result data copied from GPU " << gpu1 << " back to Host." << std::endl;

    
    // --- 6. 验证结果 ---
    bool success = true;
    for (size_t i = 0; i < dataSize / sizeof(float); ++i) {
        if (h_result[i] != h_data[i]) {
            std::cerr << "Verification FAILED at index " << i << "! Expected: " << h_data[i] << ", Got: " << h_result[i] << std::endl;
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "\nVerification SUCCESSFUL! The P2P memory copy worked correctly." << std::endl;
    } else {
        std::cout << "\nVerification FAILED! The P2P memory copy did not work." << std::endl;
    }

    // --- 7. 清理资源 ---
    free(h_data);
    free(h_result);
    CUDA_CHECK(cudaSetDevice(gpu0));
    CUDA_CHECK(cudaFree(d_data_0));
    CUDA_CHECK(cudaSetDevice(gpu1));
    CUDA_CHECK(cudaFree(d_data_1));
    std::cout << "Cleaned up all resources." << std::endl;

    return EXIT_SUCCESS;
}