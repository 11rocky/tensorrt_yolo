#pragma once
#include <NvInfer.h>
#include <string>
#include <vector>

class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8EntropyCalibrator(const uint32_t& batchSize, const std::string& calibImagesPath,
        const std::string& calibTableFilePath, const uint32_t& inputH, const uint32_t& inputW);
    virtual ~Int8EntropyCalibrator();

    int getBatchSize() const noexcept  override { return mBatchSize; }
    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept  override;
    const void* readCalibrationCache(size_t& length) noexcept  override;
    void writeCalibrationCache(const void* cache, size_t length) noexcept  override;

private:
    const uint32_t mBatchSize;
    const uint32_t mInputH;
    const uint32_t mInputW;
    const uint64_t mInputCount;
    const std::string mCalibTableFilePath;
    uint32_t mImageIndex;
    uint32_t mReservedSize;
    bool mReadCache{true};
    void* mDeviceInput{nullptr};
    void* mDevImage{nullptr};
    std::vector<std::string> mImageList;
    std::vector<char> mCalibrationCache;
};