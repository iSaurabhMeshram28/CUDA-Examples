#include <iostream>
#include <cuda_runtime.h>

class CudaDeviceProperties {
public:
    void displayDeviceProperties();

private:
    void printHeader(const std::string& header);
    void printDriverAndRuntimeInfo();
    void printDeviceGeneralInfo(const cudaDeviceProp& devProp, int deviceId);
    void printDeviceMemoryInfo(const cudaDeviceProp& devProp);
    void printDeviceMultiprocessorInfo(const cudaDeviceProp& devProp);
    void printDeviceThreadInfo(const cudaDeviceProp& devProp);
};

void CudaDeviceProperties::displayDeviceProperties() {
    printHeader("CUDA INFORMATION");

    int deviceCount = 0;
    cudaError_t ret = cudaGetDeviceCount(&deviceCount);
    if (ret != cudaSuccess) {
        std::cerr << "CUDA Runtime API Error - cudaGetDeviceCount() Failed: " 
                  << cudaGetErrorString(ret) << std::endl;
        return;
    }

    if (deviceCount == 0) {
        std::cout << "There is no CUDA-supported device on this system." << std::endl;
        return;
    }

    std::cout << "Total Number of CUDA Supporting GPU Devices: " << deviceCount << std::endl;

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp devProp;
        ret = cudaGetDeviceProperties(&devProp, i);
        if (ret != cudaSuccess) {
            std::cerr << "Error retrieving properties for device " << i << ": " 
                      << cudaGetErrorString(ret) << std::endl;
            continue;
        }

        printDriverAndRuntimeInfo();
        printDeviceGeneralInfo(devProp, i);
        printDeviceMemoryInfo(devProp);
        printDeviceMultiprocessorInfo(devProp);
        printDeviceThreadInfo(devProp);
    }
}

void CudaDeviceProperties::printHeader(const std::string& header) {
    std::cout << "*********************************" << std::endl;
    std::cout << header << std::endl;
    std::cout << "*********************************" << std::endl;
}

void CudaDeviceProperties::printDriverAndRuntimeInfo() {
    int driverVersion = 0, runtimeVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printHeader("CUDA DRIVER AND RUNTIME INFORMATION");
    std::cout << "CUDA Driver Version: " << driverVersion / 1000 << "." 
              << (driverVersion % 100) / 10 << std::endl;
    std::cout << "CUDA Runtime Version: " << runtimeVersion / 1000 << "." 
              << (runtimeVersion % 100) / 10 << std::endl;
}

void CudaDeviceProperties::printDeviceGeneralInfo(const cudaDeviceProp& devProp, int deviceId) {
    printHeader("GPU DEVICE GENERAL INFORMATION");
    std::cout << "GPU Device Number: " << deviceId << std::endl;
    std::cout << "GPU Device Name: " << devProp.name << std::endl;
    std::cout << "GPU Device Compute Capability: " << devProp.major << "." << devProp.minor << std::endl;
    std::cout << "GPU Device Clock Rate: " << devProp.clockRate << " kHz" << std::endl;
    std::cout << "GPU Device Type: " << (devProp.integrated ? "Integrated (On-Board)" : "Discrete (Card)") << std::endl;
}

void CudaDeviceProperties::printDeviceMemoryInfo(const cudaDeviceProp& devProp) {
    printHeader("GPU DEVICE MEMORY INFORMATION");
    std::cout << "GPU Device Total Memory: " 
              << static_cast<float>(devProp.totalGlobalMem) / (1024.0f * 1024.0f * 1024.0f) << " GB" << std::endl;
    std::cout << "GPU Device Constant Memory: " << devProp.totalConstMem << " Bytes" << std::endl;
    std::cout << "GPU Device Shared Memory Per SMProcessor: " << devProp.sharedMemPerBlock << " Bytes" << std::endl;
}

void CudaDeviceProperties::printDeviceMultiprocessorInfo(const cudaDeviceProp& devProp) {
    printHeader("GPU DEVICE MULTIPROCESSOR INFORMATION");
    std::cout << "GPU Device Number of SMProcessors: " << devProp.multiProcessorCount << std::endl;
    std::cout << "GPU Device Number of Registers Per SMProcessor: " << devProp.regsPerBlock << std::endl;
}

void CudaDeviceProperties::printDeviceThreadInfo(const cudaDeviceProp& devProp) {
    printHeader("GPU DEVICE THREAD INFORMATION");
    std::cout << "GPU Device Maximum Number of Threads Per SMProcessor: " 
              << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "GPU Device Maximum Number of Threads Per Block: " << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "GPU Device Threads in Warp: " << devProp.warpSize << std::endl;
    std::cout << "GPU Device Maximum Thread Dimensions: (" 
              << devProp.maxThreadsDim[0] << ", " 
              << devProp.maxThreadsDim[1] << ", " 
              << devProp.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "GPU Device Maximum Grid Dimensions: (" 
              << devProp.maxGridSize[0] << ", " 
              << devProp.maxGridSize[1] << ", " 
              << devProp.maxGridSize[2] << ")" << std::endl;
    std::cout << "GPU Device has ECC support: " << (devProp.ECCEnabled ? "Enabled" : "Disabled") << std::endl;

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    std::cout << "GPU Device CUDA Driver Mode (TCC or WDDM): " 
              << (devProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)") 
              << std::endl;
#endif
}

int main() {
    CudaDeviceProperties cudaProps;
    cudaProps.displayDeviceProperties();
    return 0;
}
