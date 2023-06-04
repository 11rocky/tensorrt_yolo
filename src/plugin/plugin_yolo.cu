#include <cuda.h>
#include <cuda_runtime.h>
#include "util.h"

inline __device__ float sigmoidGPU(const float& x) { return 1.0f / (1.0f + __expf(-x)); }

__global__ void gpuYoloLayer(const float* input, float* output, uint32_t gridH, uint32_t gridW,
    uint32_t numClasses, uint32_t numBoxes)
{
    uint32_t xId = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t yId = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t zId = blockIdx.z * blockDim.z + threadIdx.z;

    if ((xId >= gridW) || (yId >= gridH) || (zId >= numBoxes)) {
        return;
    }

    const int numGridCells = gridH * gridW;
    const int bbindex = yId * gridW + xId;

    output[bbindex + numGridCells * (zId * (5 + numClasses) + 0)]
        = sigmoidGPU(input[bbindex + numGridCells * (zId * (5 + numClasses) + 0)]);

    output[bbindex + numGridCells * (zId * (5 + numClasses) + 1)]
        = sigmoidGPU(input[bbindex + numGridCells * (zId * (5 + numClasses) + 1)]);

    output[bbindex + numGridCells * (zId * (5 + numClasses) + 2)]
        = input[bbindex + numGridCells * (zId * (5 + numClasses) + 2)];

    output[bbindex + numGridCells * (zId * (5 + numClasses) + 3)]
        = input[bbindex + numGridCells * (zId * (5 + numClasses) + 3)];

    output[bbindex + numGridCells * (zId * (5 + numClasses) + 4)]
        = sigmoidGPU(input[bbindex + numGridCells * (zId * (5 + numClasses) + 4)]);

    for (uint32_t i = 0; i < numClasses; ++i) {
        output[bbindex + numGridCells * (zId * (5 + numClasses) + (5 + i))]
            = sigmoidGPU(input[bbindex + numGridCells * (zId * (5 + numClasses) + (5 + i))]);
    }
}

cudaError_t cudaYoloLayer(const void* input, void* output, uint32_t batchs, uint32_t batchSize,
    uint32_t gridH, uint32_t gridW, uint32_t numClasses, uint32_t numBoxes, cudaStream_t stream)
{
    dim3 threadD(16, 16, 4);
    dim3 blockD(CEIL(gridW, threadD.x), CEIL(gridH, threadD.y), CEIL(numBoxes, threadD.z));
    for (int batch = 0; batch < batchs; ++batch) {
        gpuYoloLayer<<<blockD, threadD, 0, stream>>>(
            reinterpret_cast<const float*>(input) + (batch * batchSize),
            reinterpret_cast<float*>(output) + (batch * batchSize), gridH, gridW, numClasses, numBoxes);
    }
    return cudaGetLastError();
}
