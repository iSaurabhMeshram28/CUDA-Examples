#include <iostream>
#include <cuda.h>
#include "helper_timer.h"

#define BLOCK_WIDTH 32

// Declare the kernel function outside the class
__global__ void matMulGPU(int *A, int *B, int *C, int numARows, int numAColumns, int numBColumns, int numCColumns)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int column = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < numARows && column < numBColumns)
	{
		int value = 0;
		for (int k = 0; k < numAColumns; ++k)
		{
			value += A[row * numAColumns + k] * B[k * numBColumns + column];
		}
		C[row * numCColumns + column] = value;
	}
}

class MatrixMultiplication
{
private:
	int *hostA, *hostB, *hostC, *gold;
	int *deviceA, *deviceB, *deviceC;
	float timeOnCPU, timeOnGPU;

	int numARows, numAColumns, numBRows, numBColumns;
	int numCRows, numCColumns, numGoldRows, numGoldColumns;
	int sizeA, sizeB, sizeC, sizeGold;

	void allocateHostMemory();
	void allocateDeviceMemory();
	void initializeMatrices();
	void copyHostToDevice();
	void copyDeviceToHost();
	void cleanup();

	static void initMatrixA(int *data, int rows, int cols);
	static void initMatrixB(int *data, int rows, int cols);
	static void matMulCPU(int *A, int *B, int *C, int numARows, int numAColumns, int numBColumns, int numCColumns, float &timeOnCPU);

	void compareResults();

public:
	MatrixMultiplication();
	~MatrixMultiplication();
	void execute();
};

MatrixMultiplication::MatrixMultiplication()
	: hostA(nullptr), hostB(nullptr), hostC(nullptr), gold(nullptr),
	  deviceA(nullptr), deviceB(nullptr), deviceC(nullptr),
	  timeOnCPU(0.0f), timeOnGPU(0.0f)
{
	numARows = numAColumns = numBRows = numBColumns = 64;
	numCRows = numARows;
	numCColumns = numBColumns;
	numGoldRows = numARows;
	numGoldColumns = numBColumns;

	sizeA = numARows * numAColumns * sizeof(int);
	sizeB = numBRows * numBColumns * sizeof(int);
	sizeC = numCRows * numCColumns * sizeof(int);
	sizeGold = numGoldRows * numGoldColumns * sizeof(int);
}

MatrixMultiplication::~MatrixMultiplication()
{
	cleanup();
}

void MatrixMultiplication::allocateHostMemory()
{
	cudaMallocHost((void **)&hostA, sizeA);
	cudaMallocHost((void **)&hostB, sizeB);
	cudaMallocHost((void **)&hostC, sizeC);
	cudaMallocHost((void **)&gold, sizeGold);

	if (!hostA || !hostB || !hostC || !gold)
	{
		std::cerr << "Host memory allocation failed!" << std::endl;
		cleanup();
		exit(EXIT_FAILURE);
	}
}

void MatrixMultiplication::allocateDeviceMemory()
{
	cudaError_t result;

	result = cudaMalloc((void **)&deviceA, sizeA);
	if (result != cudaSuccess)
	{
		std::cerr << "Device memory allocation failed for deviceA!" << std::endl;
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = cudaMalloc((void **)&deviceB, sizeB);
	if (result != cudaSuccess)
	{
		std::cerr << "Device memory allocation failed for deviceB!" << std::endl;
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = cudaMalloc((void **)&deviceC, sizeC);
	if (result != cudaSuccess)
	{
		std::cerr << "Device memory allocation failed for deviceC!" << std::endl;
		cleanup();
		exit(EXIT_FAILURE);
	}
}

void MatrixMultiplication::initializeMatrices()
{
	initMatrixA(hostA, numARows, numAColumns);
	initMatrixB(hostB, numBRows, numBColumns);
}

void MatrixMultiplication::copyHostToDevice()
{
	cudaError_t result;

	result = cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
	if (result != cudaSuccess)
	{
		std::cerr << "Host to Device copy failed for deviceA!" << std::endl;
		cleanup();
		exit(EXIT_FAILURE);
	}

	result = cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);
	if (result != cudaSuccess)
	{
		std::cerr << "Host to Device copy failed for deviceB!" << std::endl;
		cleanup();
		exit(EXIT_FAILURE);
	}
}

void MatrixMultiplication::copyDeviceToHost()
{
	cudaError_t result;

	result = cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);
	if (result != cudaSuccess)
	{
		std::cerr << "Device to Host copy failed for hostC!" << std::endl;
		cleanup();
		exit(EXIT_FAILURE);
	}
}

void MatrixMultiplication::cleanup()
{
	if (deviceC)
		cudaFree(deviceC);
	if (deviceB)
		cudaFree(deviceB);
	if (deviceA)
		cudaFree(deviceA);
	if (gold)
		cudaFreeHost(gold);
	if (hostC)
		cudaFreeHost(hostC);
	if (hostB)
		cudaFreeHost(hostB);
	if (hostA)
		cudaFreeHost(hostA);

	deviceC = deviceB = deviceA = nullptr;
	gold = hostC = hostB = hostA = nullptr;
}

void MatrixMultiplication::initMatrixA(int *data, int rows, int cols)
{
	int num = 1;
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			data[i * cols + j] = num++;
		}
	}
}

void MatrixMultiplication::initMatrixB(int *data, int rows, int cols)
{
	int num = BLOCK_WIDTH;
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			data[i * cols + j] = num--;
		}
	}
}

void MatrixMultiplication::matMulCPU(int *A, int *B, int *C, int numARows, int numAColumns, int numBColumns, int numCColumns, float &timeOnCPU)
{
	StopWatchInterface *timer = nullptr;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	for (int i = 0; i < numARows; ++i)
	{
		for (int j = 0; j < numBColumns; ++j)
		{
			int value = 0;
			for (int k = 0; k < numAColumns; ++k)
			{
				value += A[i * numAColumns + k] * B[k * numBColumns + j];
			}
			C[i * numCColumns + j] = value;
		}
	}

	sdkStopTimer(&timer);
	timeOnCPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);
}

void MatrixMultiplication::compareResults()
{
	bool isAccurate = true;
	int breakIndex = -1;

	for (int i = 0; i < numCRows * numCColumns; ++i)
	{
		if (abs(gold[i] - hostC[i]) > 1e-5)
		{
			isAccurate = false;
			breakIndex = i;
			break;
		}
	}

	if (isAccurate)
	{
		std::cout << "Comparison of CPU and GPU results is accurate." << std::endl;
	}
	else
	{
		std::cerr << "Mismatch at index " << breakIndex << " between CPU and GPU results." << std::endl;
	}
}

void MatrixMultiplication::execute()
{
	allocateHostMemory();
	allocateDeviceMemory();
	initializeMatrices();

	std::cout << "Matrix Dimensions:" << std::endl;
	std::cout << "A: " << numARows << "x" << numAColumns << ", B: " << numBRows << "x" << numBColumns << std::endl;

	copyHostToDevice();

	dim3 dimGrid((numBColumns + BLOCK_WIDTH - 1) / BLOCK_WIDTH, (numARows + BLOCK_WIDTH - 1) / BLOCK_WIDTH, 1);
	dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);

	StopWatchInterface *timer = nullptr;
	sdkCreateTimer(&timer);

	// Measure only kernel execution time
	sdkStartTimer(&timer);
	matMulGPU<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBColumns, numCColumns);
	cudaDeviceSynchronize(); // Ensure kernel execution is complete
	sdkStopTimer(&timer);

	timeOnGPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);

	copyDeviceToHost();

	matMulCPU(hostA, hostB, gold, numARows, numAColumns, numBColumns, numCColumns, timeOnCPU);

	compareResults();

	std::cout << "Time on CPU: " << timeOnCPU << " ms" << std::endl;
	std::cout << "Time on GPU: " << timeOnGPU << " ms" << std::endl;

	cleanup();
}

int main()
{
	MatrixMultiplication matMul;
	matMul.execute();
	return 0;
}
