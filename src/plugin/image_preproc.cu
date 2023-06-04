#include <cuda.h>
#include <cuda_runtime.h>
#include "util.h"

__forceinline__ __device__ float3 get(uchar3* src, int x, int y, int w, int h)
{
    if(x < 0 || x >= w || y < 0 || y >= h) {
        return make_float3(0.5,0.5,0.5);
    }
    uchar3 temp = src[y * w + x];
    return make_float3(float(temp.x) / 255.,float(temp.y) / 255.,float(temp.z) / 255.);
}

__global__ void gpuPreProc(uchar3 *src, int inH, int inW, float *dst, int outH, int outW, float scale,
    int resizeH, int resizeW, int offH, int offW)
{
    uint32_t xId = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t yId = blockIdx.y * blockDim.y + threadIdx.y;

    if ((xId >= outW) || (yId >= outH)) {
        return;
    }
    
    const int outStride = outH * outW;
    const int outIdx = yId * outW + xId;
    if (yId < offH || yId >= offH + resizeH || xId < offW || xId >= offW + resizeW) {
        for (int i = 0; i < 3; i++) {
            dst[i * outStride + outIdx] = 0.5;
        }
        return;
    }

    int relH = yId - offH;
    int relW = xId - offW;
    float w = (relW + 0.5) * scale - 0.5;
    float h = (relH + 0.5) * scale - 0.5;
    int hLow = (int)h;
    int wLow = (int)w;
    int hHigh = hLow + 1;
    int wHigh = wLow + 1;
    float lh = h - hLow;
    float lw = w - wLow;
    float hh = 1 - lh, hw = 1 - lw;
    float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
    float3 v1 = get(src, wLow, hLow, inW, inH);
    float3 v2 = get(src, wHigh, hLow, inW, inH);
    float3 v3 = get(src, wLow, hHigh, inW, inH);
    float3 v4 = get(src, wHigh, hHigh, inW, inH);
    dst[outIdx] = w1 * v1.z + w2 * v2.z + w3 * v3.z + w4 * v4.z;
    dst[outStride + outIdx] = w1 * v1.y + w2 * v2.y + w3 * v3.y + w4 * v4.y;
    dst[2 * outStride + outIdx] = w1 * v1.x + w2 * v2.x + w3 * v3.x + w4 * v4.x;
}

cudaError_t cudaImagePreProc(uint8_t *src, int inH, int inW, float *dst, int outH, int outW)
{
    float scaleX = (outW * 1.0f / inW);
    float scaleY = (outH * 1.0f / inH);
    float scale = scaleX < scaleY ? scaleX : scaleY;
    int resizeH = (std::round(inH * scale));
    int resizeW = (std::round(inW * scale));
    if ((inW - resizeW) % 2) {
        resizeW--;
    }
    if ((inH - resizeH) % 2) {
        resizeH--;
    }
    int hOffset = (outH - resizeH) / 2;
    int wOffset = (outW - resizeW) / 2;
    scale = inH / (float)resizeH;
    dim3 threadD(32, 32, 1);
    dim3 blockD(CEIL(outW, threadD.x), CEIL(outH, threadD.y), 1);
    gpuPreProc<<<blockD, threadD, 0>>>((uchar3*)src, inH, inW, dst, outH, outW, scale,
        resizeH, resizeW, hOffset, wOffset);
    return cudaGetLastError();
}

