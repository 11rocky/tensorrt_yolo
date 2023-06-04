#pragma once
#include <NvInfer.h>
#include <cassert>
#include <cuda_runtime_api.h>
#include <iostream>
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <string>
#include <vector>
#include "utils.h"


template <typename AllocFunc, typename FreeFunc>
class GenericBuffer {
public:
    GenericBuffer(nvinfer1::DataType type = nvinfer1::DataType::kFLOAT)
        : mSize(0), mCapacity(0), mType(type), mBuffer(nullptr) {}

    GenericBuffer(size_t size, nvinfer1::DataType type) : mSize(size), mCapacity(size), mType(type)
    {
        if (!allocFn(&mBuffer, this->nbBytes())) {
            throw std::bad_alloc();
        }
    }

    GenericBuffer(GenericBuffer&& buf): mSize(buf.mSize), mCapacity(buf.mCapacity),
        mType(buf.mType), mBuffer(buf.mBuffer)
    {
        buf.mSize = 0;
        buf.mCapacity = 0;
        buf.mBuffer = nullptr;
    }

    GenericBuffer& operator=(GenericBuffer&& buf)
    {
        if (this != &buf) {
            freeFn(mBuffer);
            mSize = buf.mSize;
            mCapacity = buf.mCapacity;
            mType = buf.mType;
            mBuffer = buf.mBuffer;
            buf.mSize = 0;
            buf.mCapacity = 0;
            buf.mBuffer = nullptr;
        }
        return *this;
    }

    void* data() { return mBuffer;}
    const void* data() const { return mBuffer; }
    size_t eleNum() const { return mSize; }
    size_t nbBytes() const { return this->eleNum() * getElementSize(mType); }

    nvinfer1::DataType dataType() const { return mType; }

    ~GenericBuffer() { freeFn(mBuffer); }

private:
    size_t mSize{0}, mCapacity{0};
    nvinfer1::DataType mType;
    void* mBuffer;
    AllocFunc allocFn;
    FreeFunc freeFn;
};

class DeviceAllocator {
public:
    bool operator()(void** ptr, size_t size) const
    {
        return cudaMalloc(ptr, size) == cudaSuccess;
    }
};

class DeviceFree {
public:
    void operator()(void* ptr) const
    {
        cudaFree(ptr);
    }
};

class HostAllocator {
public:
    bool operator()(void** ptr, size_t size) const
    {
        *ptr = malloc(size);
        return *ptr != nullptr;
    }
};

class HostFree {
public:
    void operator()(void* ptr) const
    {
        free(ptr);
    }
};

using DeviceBuffer = GenericBuffer<DeviceAllocator, DeviceFree>;
using HostBuffer = GenericBuffer<HostAllocator, HostFree>;


class ManagedBuffer {
public:
    size_t getUsedSize()
    {
        return getElementSize(hostBuffer.dataType()) * volume(getInferDims());
    }

    size_t getMaxSize()
    {
        return hostBuffer.nbBytes();
    }

    int mInferBatch;
    std::string mName;
    nvinfer1::Dims mDims;
    DeviceBuffer deviceBuffer;
    HostBuffer hostBuffer;

private:
    nvinfer1::Dims getInferDims()
    {
        auto dim = mDims;
        dim.d[0] = mInferBatch;
        return dim;
    }
};

class BufferManager {
public:
    using BufferList = std::vector<std::unique_ptr<ManagedBuffer>>;

    BufferManager(nvinfer1::ICudaEngine* engine, int batchSize = 1)
        : mEngine(engine), mMaxBatch(batchSize), mInferBatch(1)
    {
        assert(batchSize > 0);
        for (int i = 0; i < mEngine->getNbIOTensors(); i++) {
            auto name = mEngine->getIOTensorName(i);
            auto ioMode = mEngine->getTensorIOMode(name);
            auto dims = mEngine->getTensorShape(name);
            dims.d[0] = mMaxBatch;
            nvinfer1::DataType type = mEngine->getTensorDataType(name);
            size_t vol = volume(dims);
            std::unique_ptr<ManagedBuffer> manBuf{new ManagedBuffer()};
            manBuf->deviceBuffer = DeviceBuffer(vol, type);
            manBuf->hostBuffer = HostBuffer(vol, type);
            manBuf->mName = name;
            manBuf->mDims = dims;
            mDeviceBindings.emplace_back(manBuf->deviceBuffer.data());
            if (ioMode == nvinfer1::TensorIOMode::kINPUT){
                mInputBuffers.emplace_back(std::move(manBuf));
            } else if (ioMode == nvinfer1::TensorIOMode::kOUTPUT) {
                mOutputBuffers.emplace_back(std::move(manBuf));
            } else {
                mOtherBuffers.emplace_back(std::move(manBuf));
            }
        }
    }

    void setInferBatch(int batch)
    {
        assert(batch <= mMaxBatch);
        mInferBatch = batch;
        for (auto &buf : mInputBuffers) {
            buf->mInferBatch = batch;
        }
        for (auto &buf : mOutputBuffers) {
            buf->mInferBatch = batch;
        }
        for (auto &buf : mOtherBuffers) {
            buf->mInferBatch = batch;
        }
    }

    std::vector<void*>& getDeviceBindings() { return mDeviceBindings; }

    const std::vector<void*>& getDeviceBindings() const { return mDeviceBindings; }

    size_t getInputNum() { return mInputBuffers.size(); }

    void* getInputBuffer(int idx, bool isHost = true)
    {
        return getBuffer(mInputBuffers, idx, isHost);
    }

    void* getInputBuffer(const std::string &tensorName, bool isHost = true)
    {
        return getBuffer(mInputBuffers, tensorName, isHost);
    }

    size_t getInputBufferSize(int idx)
    {
        return getBufferSize(mInputBuffers, idx);
    }

    size_t getInputBufferSize(const std::string &tensorName)
    {
        return getBufferSize(mInputBuffers, tensorName);
    }

    size_t getInputBufferMaxSize(const std::string &tensorName)
    {
        return getBufferSize(mInputBuffers, tensorName, true);
    }

    size_t getOutputNum() { return mOutputBuffers.size(); }

    const std::string& getOutputName(int idx) const
    {
        return getBufferName(mOutputBuffers, idx);
    }

    size_t getOutputBufferSize(int idx)
    {
        return getBufferSize(mOutputBuffers, idx);
    }

    void* getOutputBuffer(int idx, bool isHost = true)
    {
        return getBuffer(mOutputBuffers, idx, isHost);
    }

    void* getOutputBuffer(const std::string &tensorName, bool isHost = true)
    {
        return getBuffer(mOutputBuffers, tensorName, isHost);
    }

    std::vector<int> getOutputTensorDim(int idx)
    {
        if (idx >= mOutputBuffers.size()) {
            return {};
        }
        auto &buf = mOutputBuffers[idx];
        return dimsToVector(buf->mDims);
    }

    std::vector<int> getOutputTensorDim(const std::string &name)
    {
        for (int i = 0; i < mOutputBuffers.size(); i++) {
            auto &buf = mOutputBuffers[i];
            if (buf->mName == name) {
                return dimsToVector(buf->mDims);
            }
        }
        return {};
    }

    const ManagedBuffer* getOutputHostBuffer(int idx)
    {
        if (idx >= mOutputBuffers.size()) {
            return nullptr;
        }
        return mOutputBuffers[idx].get();
    }

    void copyInputToDevice()
    {
        memcpyBuffers(mInputBuffers, false, false);
    }

    void copyInputToHost()
    {
        memcpyBuffers(mInputBuffers, true, false);
    }

    void copyOutputToHost()
    {
        memcpyBuffers(mOutputBuffers, true, false);
    }

    void copyInputToDeviceAsync(const cudaStream_t& stream = 0)
    {
        memcpyBuffers(mInputBuffers, false, true, stream);
    }

    void copyOutputToHostAsync(const cudaStream_t& stream = 0)
    {
        memcpyBuffers(mOutputBuffers, true, true, stream);
    }

private:
    void* getBuffer(const BufferList &buffers, int idx, bool isHost) const
    {
        if (idx >= buffers.size()) {
            return nullptr;
        }
        auto &buf = buffers[idx];
        return isHost ? buf->hostBuffer.data() : buf->deviceBuffer.data();
    }

    void* getBuffer(const BufferList &buffers, const std::string &tensorName, bool isHost) const
    {
        for (int i = 0; i < buffers.size(); i++) {
            auto &buf = buffers[i];
            if (buf->mName == tensorName) {
                return isHost ? buf->hostBuffer.data() : buf->deviceBuffer.data();
            }
        }
        return nullptr;
    }

    const std::string& getBufferName(const BufferList &buffers, int idx) const
    {
        static const std::string dft;
        if (idx >= buffers.size()) {
            return dft;
        }
        return buffers[idx]->mName;
    }

    size_t getBufferSize(const BufferList &buffers, int idx, bool max = false) const
    {
        if (idx >= buffers.size()) {
            return 0;
        }
        return max ? buffers[idx]->getMaxSize() : buffers[idx]->getUsedSize();
    }

    size_t getBufferSize(const BufferList &buffers, const std::string &tensorName, bool max = false) const
    {
        for (int i = 0; i < buffers.size(); i++) {
            auto &buf = buffers[i];
            if (buf->mName == tensorName) {
                return max ? buf->getMaxSize() : buf->getUsedSize();
            }
        }
        return 0;
    }

    void memcpyBuffers(const BufferList &buffers, bool deviceToHost, bool async, const cudaStream_t& stream = 0)
    {
        const cudaMemcpyKind memcpyType = deviceToHost ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
        for (const auto &buf : buffers) {
            void *dst = deviceToHost ? buf->hostBuffer.data() : buf->deviceBuffer.data();
            void *src = deviceToHost ? buf->deviceBuffer.data() : buf->hostBuffer.data();
            size_t size = buf->getUsedSize();
            if (async) {
                cudaMemcpyAsync(dst, src, size, memcpyType, stream);
            } else {
                cudaMemcpy(dst, src, size, memcpyType);
            }
        }
    }


    BufferList mInputBuffers;
    BufferList mOutputBuffers;
    BufferList mOtherBuffers;
    nvinfer1::ICudaEngine *mEngine;
    int mMaxBatch;
    int mInferBatch;
    std::vector<void*> mDeviceBindings;
};

