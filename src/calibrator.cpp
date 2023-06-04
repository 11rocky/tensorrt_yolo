#include "calibrator.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <fmt/format.h>
#include <pystring/pystring.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <cuda.h>   
#include <sys/types.h>
#include <dirent.h>
#include "logger.h"
#include "yolo_engine.h"
#include "plugin/image_preproc.h"

static std::vector<std::string> loadImageList(const std::string &path)
{
    DIR *dir = opendir(path.c_str());
    if (dir == nullptr) {
        LOG_ERROR("path is invalid");
        return {};
    }
    std::vector<std::string> fileList;
    struct dirent *entry = readdir(dir);
    while (entry != nullptr) {
        auto file = fmt::format("{}/{}", path, entry->d_name);
        if (entry->d_type == 0x8) {
            LOG_INFO("image: {}", file);
            fileList.push_back(file);
        }
        entry = readdir(dir);
    }
    closedir(dir);
    return fileList;
}

Int8EntropyCalibrator::Int8EntropyCalibrator(const uint32_t& batchSize, const std::string& calibImagesPath,
    const std::string& calibTableFilePath, const uint32_t& inputH, const uint32_t& inputW)
    : mBatchSize(batchSize), mInputH(inputH), mInputW(inputW), mInputCount(batchSize * inputH * inputW * 3),
    mCalibTableFilePath(calibTableFilePath), mImageIndex(0), mReservedSize(960 * 720 * 3)
{
    mImageList = loadImageList(calibImagesPath);
    if (mImageList.empty()) {
        LOG_ERROR("no calibrator images");
    }
    mImageList.resize(static_cast<int>(mImageList.size() / mBatchSize) * mBatchSize);
    cudaMalloc(&mDeviceInput, mInputCount * sizeof(float));
    cudaMalloc(&mDevImage, mReservedSize);
}

Int8EntropyCalibrator::~Int8EntropyCalibrator()
{
    cudaFree(mDevImage);
    cudaFree(mDeviceInput);
}

bool Int8EntropyCalibrator::getBatch(void* bindings[], const char* names[], int nbBindings)noexcept
{
    if (mImageIndex + mBatchSize >= mImageList.size()) {
        return false;
    }
    if (strcmp(names[0], YoloEngine::inputName) != 0) {
        return false;
    }
    float *blob = (float*)mDeviceInput;
    for (uint32_t j = mImageIndex; j < mImageIndex + mBatchSize; ++j) {
        uint32_t i = j - mImageIndex;
        auto image = cv::imread(mImageList.at(j));
        size_t imageSize = image.total() * image.elemSize();
        if (imageSize > mReservedSize) {
            mReservedSize = imageSize;
            cudaFree(mDevImage);
            cudaMalloc(&mDevImage, mReservedSize);
        }
        auto dst = blob + i * 3 * mInputH * mInputW;
        cudaMemcpy(mDevImage, image.data, imageSize, cudaMemcpyHostToDevice);
        cudaImagePreProc((uint8_t*)mDevImage, image.rows, image.cols, dst, mInputH, mInputW);
    }
    mImageIndex += mBatchSize;
    bindings[0] = mDeviceInput;
    return true;
}

const void* Int8EntropyCalibrator::readCalibrationCache(size_t& length) noexcept
{
    void* output;
    mCalibrationCache.clear();
    std::ifstream input(mCalibTableFilePath, std::ios::binary);
    input >> std::noskipws;
    if (mReadCache && input.good()) {
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(),
            std::back_inserter(mCalibrationCache));
    }

    length = mCalibrationCache.size();
    if (length) {
        LOG_INFO("using cached calibration table to build the engine");
        output = &mCalibrationCache[0];
    } else {
        LOG_INFO("new calibration table will be created to build the engine");
        output = nullptr;
    }
    return output;
}

void Int8EntropyCalibrator::writeCalibrationCache(const void* cache, size_t length) noexcept
{
    std::ofstream output(mCalibTableFilePath, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
    output.close();
}
