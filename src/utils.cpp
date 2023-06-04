#include "utils.h"

std::vector<int> dimsToVector(const nvinfer1::Dims &dim)
{
    std::vector<int> out(dim.nbDims);
    for (int i = 0; i < dim.nbDims; i++) {
        out[i] = dim.d[i];
    }
    return out;
}

uint32_t getElementSize(nvinfer1::DataType t)
{
    switch (t) {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kUINT8:
        case nvinfer1::DataType::kINT8:
        case nvinfer1::DataType::kFP8: return 1;
    }
    return 0;
}

void writeToFile(const std::string &path, void *data, size_t size)
{
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        return;
    }
    ofs.write((char*)data, size);
    ofs.close();
}

std::pair<void*, size_t> readFromFile(const std::string &file)
{
    std::ifstream ifs(file);
    if (!ifs) {
        return {nullptr, 0};
    }
    auto start = ifs.tellg();
    ifs.seekg(0, std::ios::end);
    auto size = ifs.tellg() - start;
    ifs.seekg(0, std::ios::beg);
    char *data = new char[size];
    ifs.read(data, size);
    ifs.close();
    return {data, size};
}
