#pragma once
#include <NvInfer.h>
#include "yolo_loader.h"

struct EngineInfo {
    int maxBatch;
    int inputC;
    int inputH;
    int inputW;
    int yoloNum;
    nvinfer1::DataType type;
};

class YoloEngine {
public:
    static bool build(const std::string &cfgFile, const std::string &weightFile, const std::string &enginePath,
        const std::string &calibPath, const std::string &calibTable, int maxBatch = 1, nvinfer1::DataType dataType = nvinfer1::DataType::kFLOAT,
        const std::vector<std::string> &dumps = {});
    static std::pair<nvinfer1::ICudaEngine*, nvinfer1::IRuntime*> load(const std::string &enginePath,
        EngineInfo &info, std::vector<Yolo> &yolos);
    static const char *inputName;
};

