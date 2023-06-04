#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

cudaError_t cudaImagePreProc(uint8_t *src, int inH, int inW, float *dst, int outH, int outW);
