#include <stdio.h>
#include <cuda_runtime.h> // Use cuda_runtime.h for runtime APIs

const int iNumberOfArrayElements = 5;

float* hostInput1 = NULL;
float* hostInput2 = NULL;
float* hostOutput = NULL;
float* deviceInput1 = NULL;
float* deviceInput2 = NULL;
float* deviceOutput = NULL;

__global__ void vecAddGPU(float* in1, float* in2, float* out, int len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < len)
    {
        out[i] = in1[i] + in2[i];
    }
}

int main(void)
{
    void cleanup(void);
    int size = iNumberOfArrayElements * sizeof(float);
    cudaError_t result = cudaSuccess;

    // Allocate host memory
    hostInput1 = (float*)malloc(size);
    if (hostInput1 == NULL)
    {
        printf("Host Memory allocation failed for hostInput1 array.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostInput2 = (float*)malloc(size);
    if (hostInput2 == NULL)
    {
        printf("Host Memory allocation failed for hostInput2 array.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    hostOutput = (float*)malloc(size);
    if (hostOutput == NULL)
    {
        printf("Host Memory allocation failed for hostOutput array.\n");
        cleanup();
        exit(EXIT_FAILURE);
    }

    // Initialize host arrays
    for (int i = 0; i < iNumberOfArrayElements; i++)
    {
        hostInput1[i] = 100.0f + i;
        hostInput2[i] = 200.0f + i;
    }

    // Allocate device memory
    result = cudaMalloc((void**)&deviceInput1, size);
    if (result != cudaSuccess)
    {
        printf("Device Memory allocation failed for deviceInput1 array: %s\n", cudaGetErrorString(result));
        cleanup();
        exit(EXIT_FAILURE);
    }

    result = cudaMalloc((void**)&deviceInput2, size);
    if (result != cudaSuccess)
    {
        printf("Device Memory allocation failed for deviceInput2 array: %s\n", cudaGetErrorString(result));
        cleanup();
        exit(EXIT_FAILURE);
    }

    result = cudaMalloc((void**)&deviceOutput, size);
    if (result != cudaSuccess)
    {
        printf("Device Memory allocation failed for deviceOutput array: %s\n", cudaGetErrorString(result));
        cleanup();
        exit(EXIT_FAILURE);
    }

    // Copy data from host to device
    result = cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);
    if (result != cudaSuccess)
    {
        printf("Host to Device Data Copy failed for deviceInput1 array: %s\n", cudaGetErrorString(result));
        cleanup();
        exit(EXIT_FAILURE);
    }

    result = cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);
    if (result != cudaSuccess)
    {
        printf("Host to Device Data Copy failed for deviceInput2 array: %s\n", cudaGetErrorString(result));
        cleanup();
        exit(EXIT_FAILURE);
    }

    // Configure kernel launch parameters
    dim3 dimBlock = dim3(256, 1, 1); // 256 threads per block
    dim3 dimGrid = dim3((iNumberOfArrayElements + dimBlock.x - 1) / dimBlock.x, 1, 1);

    // Launch kernel
    vecAddGPU<<<dimGrid, dimBlock>>>(deviceInput1, deviceInput2, deviceOutput, iNumberOfArrayElements);

    // Check for kernel launch errors
    result = cudaGetLastError();
    if (result != cudaSuccess)
    {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(result));
        cleanup();
        exit(EXIT_FAILURE);
    }

    // Copy result from device to host
    result = cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);
    if (result != cudaSuccess)
    {
        printf("Device to Host Data Copy failed for hostOutput array: %s\n", cudaGetErrorString(result));
        cleanup();
        exit(EXIT_FAILURE);
    }

    // Print results
    for (int i = 0; i < iNumberOfArrayElements; i++)
    {
        printf("%f + %f = %f\n", hostInput1[i], hostInput2[i], hostOutput[i]);
    }

    cleanup();
    return 0;
}

void cleanup(void)
{
    if (deviceOutput)
    {
        cudaFree(deviceOutput);
        deviceOutput = NULL;
    }

    if (deviceInput2)
    {
        cudaFree(deviceInput2);
        deviceInput2 = NULL;
    }

    if (deviceInput1)
    {
        cudaFree(deviceInput1);
        deviceInput1 = NULL;
    }

    if (hostOutput)
    {
        free(hostOutput);
        hostOutput = NULL;
    }

    if (hostInput2)
    {
        free(hostInput2);
        hostInput2 = NULL;
    }

    if (hostInput1)
    {
        free(hostInput1);
        hostInput1 = NULL;
    }
}