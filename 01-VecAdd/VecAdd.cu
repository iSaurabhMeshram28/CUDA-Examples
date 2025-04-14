#include <stdio.h>
#include <cuda.h>
#include <cstdlib>
#include <cmath>
#include "helper_timer.h"

class VectorAddition
{
private:
	const int iNumberOfArrayElements;
	float *hostInput1, *hostInput2, *hostOutput, *gold;
	float *deviceInput1, *deviceInput2, *deviceOutput;
	float timeOnCPU, timeOnGPU;

	void allocateHostMemory();
	void allocateDeviceMemory();
	void fillFloatArrayWithRandomNumbers(float *arr, int len);
	void copyDataToDevice();
	void copyDataToHost();
	void vecAddCPU(const float *arr1, const float *arr2, float *out, int len);
	void cleanup();
	void compareResults();
	void printResults(dim3 dimGrid, dim3 dimBlock);

public:
	VectorAddition(int numElements);
	~VectorAddition();
	void execute();
};

VectorAddition::VectorAddition(int numElements)
	: iNumberOfArrayElements(numElements), hostInput1(nullptr), hostInput2(nullptr),
	  hostOutput(nullptr), gold(nullptr), deviceInput1(nullptr), deviceInput2(nullptr),
	  deviceOutput(nullptr), timeOnCPU(0.0f), timeOnGPU(0.0f) {}

VectorAddition::~VectorAddition()
{
	cleanup();
}

void VectorAddition::allocateHostMemory()
{
	int size = iNumberOfArrayElements * sizeof(float);

	hostInput1 = (float *)malloc(size);
	if (!hostInput1)
	{
		printf("Host Memory allocation failed for hostInput1 array.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	hostInput2 = (float *)malloc(size);
	if (!hostInput2)
	{
		printf("Host Memory allocation failed for hostInput2 array.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	hostOutput = (float *)malloc(size);
	if (!hostOutput)
	{
		printf("Host Memory allocation failed for hostOutput array.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	gold = (float *)malloc(size);
	if (!gold)
	{
		printf("Host Memory allocation failed for gold array.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}
}

void VectorAddition::allocateDeviceMemory()
{
	int size = iNumberOfArrayElements * sizeof(float);
	cudaError_t result;

	result = cudaMalloc((void **)&deviceInput1, size);
	if (result != cudaSuccess)
	{
		printf("Device Memory allocation failed for deviceInput1 array.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = cudaMalloc((void **)&deviceInput2, size);
	if (result != cudaSuccess)
	{
		printf("Device Memory allocation failed for deviceInput2 array.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = cudaMalloc((void **)&deviceOutput, size);
	if (result != cudaSuccess)
	{
		printf("Device Memory allocation failed for deviceOutput array.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}
}

void VectorAddition::fillFloatArrayWithRandomNumbers(float *arr, int len)
{
	const float fscale = 1.0f / (float)RAND_MAX;
	for (int i = 0; i < len; i++)
	{
		arr[i] = fscale * rand();
	}
}

void VectorAddition::copyDataToDevice()
{
	int size = iNumberOfArrayElements * sizeof(float);
	cudaError_t result;

	result = cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);
	if (result != cudaSuccess)
	{
		printf("Host to Device Data Copy failed for deviceInput1 array.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);
	if (result != cudaSuccess)
	{
		printf("Host to Device Data Copy failed for deviceInput2 array.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}
}

void VectorAddition::copyDataToHost()
{
	int size = iNumberOfArrayElements * sizeof(float);
	cudaError_t result;

	result = cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);
	if (result != cudaSuccess)
	{
		printf("Device to Host Data Copy failed for hostOutput array. Error: %s\n", cudaGetErrorString(result));
		cleanup();
		exit(EXIT_FAILURE);
	}
}

void VectorAddition::vecAddCPU(const float *arr1, const float *arr2, float *out, int len)
{
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	for (int i = 0; i < len; i++)
	{
		out[i] = arr1[i] + arr2[i];
	}

	sdkStopTimer(&timer);
	timeOnCPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);
}

void VectorAddition::cleanup()
{
	if (deviceOutput)
		cudaFree(deviceOutput);
	if (deviceInput2)
		cudaFree(deviceInput2);
	if (deviceInput1)
		cudaFree(deviceInput1);
	if (gold)
		free(gold);
	if (hostOutput)
		free(hostOutput);
	if (hostInput2)
		free(hostInput2);
	if (hostInput1)
		free(hostInput1);

	deviceOutput = deviceInput2 = deviceInput1 = nullptr;
	gold = hostOutput = hostInput2 = hostInput1 = nullptr;
}

void VectorAddition::compareResults()
{
	const float epsilon = 0.000001f;
	bool bAccuracy = true;
	int breakvalue = -1;

	for (int i = 0; i < iNumberOfArrayElements; i++)
	{
		if (fabs(gold[i] - hostOutput[i]) > epsilon)
		{
			bAccuracy = false;
			breakvalue = i;
			break;
		}
	}

	if (!bAccuracy)
	{
		printf("Comparison of CPU and GPU Vector Addition is not within accuracy of 0.000001 at array index %d\n", breakvalue);
	}
	else
	{
		printf("Comparison of CPU and GPU Vector Addition is within accuracy of 0.000001\n");
	}
}

void VectorAddition::printResults(dim3 dimGrid, dim3 dimBlock)
{
	printf("CUDA Kernel Grid dimension = %d,%d,%d and Block dimension = %d,%d,%d\n",
		   dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);
	printf("Time taken for Vector Addition on CPU = %.6f ms\n", timeOnCPU);
	printf("Time taken for Vector Addition on GPU = %.6f ms\n", timeOnGPU);
}

__global__ void vecAddGPU(float *in1, float *in2, float *out, int len)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len)
	{
		out[i] = in1[i] + in2[i];
	}
}

void VectorAddition::execute()
{
	allocateHostMemory();
	fillFloatArrayWithRandomNumbers(hostInput1, iNumberOfArrayElements);
	fillFloatArrayWithRandomNumbers(hostInput2, iNumberOfArrayElements);

	allocateDeviceMemory();
	copyDataToDevice();

	dim3 dimGrid((int)ceil((float)iNumberOfArrayElements / 256.0f), 1, 1);
	dim3 dimBlock(256, 1, 1);

	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	vecAddGPU<<<dimGrid, dimBlock>>>(deviceInput1, deviceInput2, deviceOutput, iNumberOfArrayElements);

	sdkStopTimer(&timer);
	timeOnGPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);

	copyDataToHost();
	vecAddCPU(hostInput1, hostInput2, gold, iNumberOfArrayElements);
	compareResults();
	printResults(dimGrid, dimBlock);
}

int main()
{
	const int numElements = 11444777;
	VectorAddition vectorAddition(numElements);
	vectorAddition.execute();
	return 0;
}
