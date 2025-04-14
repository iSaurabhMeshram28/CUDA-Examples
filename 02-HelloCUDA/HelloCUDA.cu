#include <iostream>
#include <cuda_runtime.h>

// Kernel function must be outside the class
__global__ void vecAddGPU(float *in1, float *in2, float *out, int len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len)
    {
        out[i] = in1[i] + in2[i];
    }
}

class VectorAddition
{
private:
    const int iNumberOfArrayElements;
    float *hostInput1, *hostInput2, *hostOutput;
    float *deviceInput1, *deviceInput2, *deviceOutput;

public:
    VectorAddition(int numElements)
        : iNumberOfArrayElements(numElements), hostInput1(nullptr), hostInput2(nullptr),
          hostOutput(nullptr), deviceInput1(nullptr), deviceInput2(nullptr), deviceOutput(nullptr) {}

    ~VectorAddition()
    {
        cleanup();
    }

    void initializeHostArrays()
    {
        int size = iNumberOfArrayElements * sizeof(float);

        hostInput1 = (float *)malloc(size);
        hostInput2 = (float *)malloc(size);
        hostOutput = (float *)malloc(size);

        if (!hostInput1 || !hostInput2 || !hostOutput)
        {
            std::cerr << "Host memory allocation failed.\n";
            cleanup();
            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < iNumberOfArrayElements; i++)
        {
            hostInput1[i] = 100.0f + i;
            hostInput2[i] = 200.0f + i;
        }
    }

    void allocateDeviceMemory()
    {
        int size = iNumberOfArrayElements * sizeof(float);
        cudaError_t result;

        result = cudaMalloc((void **)&deviceInput1, size);
        if (result != cudaSuccess)
        {
            std::cerr << "Device memory allocation failed for deviceInput1: " << cudaGetErrorString(result) << "\n";
            cleanup();
            exit(EXIT_FAILURE);
        }

        result = cudaMalloc((void **)&deviceInput2, size);
        if (result != cudaSuccess)
        {
            std::cerr << "Device memory allocation failed for deviceInput2: " << cudaGetErrorString(result) << "\n";
            cleanup();
            exit(EXIT_FAILURE);
        }

        result = cudaMalloc((void **)&deviceOutput, size);
        if (result != cudaSuccess)
        {
            std::cerr << "Device memory allocation failed for deviceOutput: " << cudaGetErrorString(result) << "\n";
            cleanup();
            exit(EXIT_FAILURE);
        }
    }

    void copyDataToDevice()
    {
        int size = iNumberOfArrayElements * sizeof(float);
        cudaError_t result;

        result = cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);
        if (result != cudaSuccess)
        {
            std::cerr << "Host to Device data copy failed for deviceInput1: " << cudaGetErrorString(result) << "\n";
            cleanup();
            exit(EXIT_FAILURE);
        }

        result = cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);
        if (result != cudaSuccess)
        {
            std::cerr << "Host to Device data copy failed for deviceInput2: " << cudaGetErrorString(result) << "\n";
            cleanup();
            exit(EXIT_FAILURE);
        }
    }

    void launchKernel()
    {
        dim3 dimBlock(256, 1, 1);
        dim3 dimGrid((iNumberOfArrayElements + dimBlock.x - 1) / dimBlock.x, 1, 1);

        vecAddGPU<<<dimGrid, dimBlock>>>(deviceInput1, deviceInput2, deviceOutput, iNumberOfArrayElements);

        cudaError_t result = cudaGetLastError();
        if (result != cudaSuccess)
        {
            std::cerr << "Kernel launch failed: " << cudaGetErrorString(result) << "\n";
            cleanup();
            exit(EXIT_FAILURE);
        }
    }

    void copyDataToHost()
    {
        int size = iNumberOfArrayElements * sizeof(float);
        cudaError_t result;

        result = cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);
        if (result != cudaSuccess)
        {
            std::cerr << "Device to Host data copy failed for hostOutput: " << cudaGetErrorString(result) << "\n";
            cleanup();
            exit(EXIT_FAILURE);
        }
    }

    void printResults() const
    {
        for (int i = 0; i < iNumberOfArrayElements; i++)
        {
            std::cout << hostInput1[i] << " + " << hostInput2[i] << " = " << hostOutput[i] << "\n";
        }
    }

private:
    void cleanup()
    {
        if (deviceOutput)
        {
            cudaFree(deviceOutput);
            deviceOutput = nullptr;
        }

        if (deviceInput2)
        {
            cudaFree(deviceInput2);
            deviceInput2 = nullptr;
        }

        if (deviceInput1)
        {
            cudaFree(deviceInput1);
            deviceInput1 = nullptr;
        }

        if (hostOutput)
        {
            free(hostOutput);
            hostOutput = nullptr;
        }

        if (hostInput2)
        {
            free(hostInput2);
            hostInput2 = nullptr;
        }

        if (hostInput1)
        {
            free(hostInput1);
            hostInput1 = nullptr;
        }
    }
};

int main()
{
    const int numElements = 5;
    VectorAddition vecAdd(numElements);

    vecAdd.initializeHostArrays();
    vecAdd.allocateDeviceMemory();
    vecAdd.copyDataToDevice();
    vecAdd.launchKernel();
    vecAdd.copyDataToHost();
    vecAdd.printResults();

    return 0;
}
