#include <stdio.h>
#include <cuda.h>
#include "helper_timer.h"
#define BLOCK_WIDTH 32

int *hostA = NULL;
int *hostB = NULL;
int *hostC = NULL;
int *gold = NULL;
int *deviceA = NULL;
int *deviceB = NULL;
int *deviceC = NULL;
float timeOnCPU = 0.0f;
float timeOnGPU = 0.0f;
__global__ void matMulGPU(int *A, int *B, int *C, int numARows, int numAColumns, int numBColumns, int numCColumns)
{

	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int column = blockIdx.x * blockDim.x + threadIdx.x;

	if ((row < numARows) && (column < numBColumns))
	{
		int value = 0;
		for (int k = 0; k < numAColumns; k++)
		{
			int a = A[row * numAColumns + k];
			int b = B[k * numBColumns + column];
			value += a * b;
		}
		C[row * numCColumns + column] = value;
	}
}

int main(int argc, char *argv[])
{
	void InitA(int *data, int, int);
	void InitB(int *data, int, int);
	void matMulCPU(int *, int *, int *, int, int, int, int);
	void cleanup(void);

	int numARows = BLOCK_WIDTH;
	int numAColumns = BLOCK_WIDTH;
	int numBRows = BLOCK_WIDTH;
	int numBColumns = BLOCK_WIDTH;

	int numCRows = numARows;
	int numCColumns = numBColumns;
	int numGoldRows = numARows;
	int numGoldColumns = numBColumns;
	int sizeA = numARows * numAColumns * sizeof(int);
	int sizeB = numBRows * numBColumns * sizeof(int);
	int sizeC = numCRows * numCColumns * sizeof(int);
	int sizeGold = numGoldRows * numGoldColumns * sizeof(int);
	cudaError_t result = cudaSuccess;
	hostA = (int *)malloc(sizeA);
	if (hostA == NULL)
	{
		printf("Host Memory allocation is failed for hostA Matrix.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}
	hostB = (int *)malloc(sizeB);
	if (hostB == NULL)
	{
		printf("Host Memory allocation is failed for hostB Matrix.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}
	hostC = (int *)malloc(sizeC);
	if (hostC == NULL)
	{
		printf("Host Memory allocation is failed for hostC Matrix.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}
	gold = (int *)malloc(sizeGold);
	if (gold == NULL)
	{
		printf("Host Memory allocation is failed for gold Matrix.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}
	// printing matrix dimensions and
	printf("The Dimensions Of Matrix for 'hostA' : %d x %d\n", numARows, numAColumns);
	printf("The Dimensions Of Matrix for 'hostB' : %d x %d\n", numBRows, numBColumns);
	printf("The Dimensions Of Matrix for 'hostC' : %d x %d\n", numCRows, numCColumns);
	printf("The Dimensions Of Matrix 'gold Are : %d x %d\n", numGoldRows, numGoldColumns);

	printf("Size Of Matrix hostA = %d\n", sizeA);
	printf("Size Of Matrix hostB = %d\n", sizeB);
	printf("Size Of Matrix hostC = %d\n", sizeC);
	printf("Size Of Matrix gold = %d\n", sizeGold);
	InitA(hostA, numARows, numAColumns);
	InitB(hostB, numBRows, numBColumns);
	result = cudaMalloc((void **)&deviceA, sizeA);
	if (result != cudaSuccess)
	{
		printf("Device Memory allocation is failed for deviceA matrix.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}
	result = cudaMalloc((void **)&deviceB, sizeB);
	if (result != cudaSuccess)
	{
		printf("Device Memory allocation is failed for deviceB matrix.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}
	result = cudaMalloc((void **)&deviceC, sizeC);
	if (result != cudaSuccess)
	{
		printf("Device Memory allocation is failed for deviceC matrix.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}
	result = cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice);
	if (result != cudaSuccess)
	{
		printf("Host to Device Data Copy is failed for deviceA matrix.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}
	result = cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice);
	if (result != cudaSuccess)
	{
		printf("Host to Device Data Copy is failed for deviceB matrix.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}
	dim3 dimGrid((numBColumns + BLOCK_WIDTH - 1) / BLOCK_WIDTH, (numARows + BLOCK_WIDTH - 1) / BLOCK_WIDTH, 1);
	dim3 dimBlock = dim3(BLOCK_WIDTH, BLOCK_WIDTH, 1);
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);

	matMulGPU<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBColumns, numCColumns);

	sdkStopTimer(&timer);
	timeOnGPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);
	timer = NULL;
	result = cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost);
	if (result != cudaSuccess)
	{
		printf("Device to Host Data Copy is failed for hostC matrix.\n");
		cleanup();
		exit(EXIT_FAILURE);
	}
	matMulCPU(hostA, hostB, gold, numARows, numAColumns, numBColumns, numCColumns);
	int breakvalue = -1;
	bool bAccuracy = true;
	for (int i = 0; i < numCRows * numCColumns; i++)
	{
		int vall = gold[i];
		int val2 = hostC[i];
		if (abs(vall - val2) > 1e-5)
		{
			bAccuracy = false;
			breakvalue = i;
			break;
		}
	}

	char str[128];
	if (bAccuracy == false)
		sprintf(str, "Comparison of CPU and GPU Matrix Multiplication is not accurate at array index % d", breakvalue);
	else
		sprintf(str, "Comparison of CPU and GPU Matrix Multiplication is accurate");

	printf("Time taken for Matrix Multiplication on CPU = %.6f\n", timeOnCPU);
	printf("Time taken for Matrix Multiplication on GPU = %.6f\n", timeOnGPU);
	printf("%s\n", str);
	cleanup();
	return (0);
}

void InitA(int *data, int row, int col)
{

	int num = 1;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			*(data + i * col + j) = num;
			num++;
		}
	}
}

void InitB(int *data, int row, int col)
{

	int num = BLOCK_WIDTH;
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
		{
			*(data + i * col + j) = num;
			num--;
		}
	}
}

void matMulCPU(int *A, int *B, int *C, int numARows, int numAColumns, int numBColumns, int numCColumns)
{
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);
	for (int i = 0; i < numARows; ++i)
	{
		for (int j = 0; j < numBColumns; ++j)
		{
			int value = 0;
			for (int k = 0; k < numAColumns; ++k)
			{
				int a = A[i * numAColumns + k];
				int b = B[k * numBColumns + j];
				value += a * b;
			}
			C[i * numCColumns + j] = value;
		}
	}
	sdkStopTimer(&timer);
	timeOnCPU = sdkGetTimerValue(&timer);
	sdkDeleteTimer(&timer);
	timer = NULL;
}

void cleanup(void)
{
	if (deviceC)
	{
		cudaFree(deviceC);
		deviceC = NULL;
	}
	if (deviceB)
	{
		cudaFree(deviceB);
		deviceB = NULL;
	}
	if (deviceA)
	{
		cudaFree(deviceA);
		deviceA = NULL;
	}
	if (gold)
	{
		free(gold);
		gold = NULL;
	}
	if (hostC)
	{
		free(hostC);
		hostC = NULL;
	}
	if (hostB)
	{
		free(hostB);
		hostB = NULL;
	}
	if (hostA)
	{
		free(hostA);
		hostA = NULL;
	}
}