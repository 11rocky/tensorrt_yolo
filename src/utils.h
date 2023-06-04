#pragma once
#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <random>
#include <NvInfer.h>

std::vector<int> dimsToVector(const nvinfer1::Dims &dim);
uint32_t getElementSize(nvinfer1::DataType t);
std::pair<void*, size_t> readFromFile(const std::string &file);
void writeToFile(const std::string &path, void *data, size_t size);

inline int randomColor()
{
    static std::default_random_engine r;
    return r() % 255;
}

inline int64_t volume(nvinfer1::Dims const& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, int64_t{1}, std::multiplies<int64_t>{});
}
