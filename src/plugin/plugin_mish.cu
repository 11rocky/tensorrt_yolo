#include <cuda.h>
#include <cuda_runtime.h>
#include "util.h"

__device__ float tanh_activate_kernel(float x) { return (2 / (1 + expf(-2 * x)) - 1); }

__device__ float softplus_kernel(float x, float threshold = 20) 
{
    if (x > threshold) {
        return x;
    } else if (x < -threshold) {
        return expf(x);
    }
    return logf(expf(x) + 1);
}

__global__ void gpuMishLayer(const float* input, float* output, uint32_t count)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= count) {
        return;
    }
    output[idx] = input[idx] * tanh_activate_kernel(softplus_kernel(input[idx]));
}

cudaError_t cudaMishLayer(const void* input, void* output, uint32_t count, cudaStream_t stream)
{
    int threadD = 16 * 16;
    int blockD = CEIL(count, threadD);

    gpuMishLayer<<<blockD, threadD, 0, stream>>>(
        reinterpret_cast<const float*>(input), reinterpret_cast<float*>(output), count);
    
    return cudaGetLastError();
}
