#pragma once
#include <opencv2/opencv.hpp>
#include "yolo_engine.h"
#include "buffer.h"

struct Box {
    float x;
    float y;
    float w;
    float h;
};

struct Detection {
    Box box;
    int classes;
    std::vector<float> prob;
    std::vector<float> mask;
    float objectness;
    int sortClass;
};

class YoloInfer {
public:
    ~YoloInfer();
    bool init(const std::string &enginePath);
    void saveInferOutputs(const std::string &path);
    bool infer(const std::vector<cv::Mat> &images);
    std::vector<std::vector<Detection>> postProcess(const std::vector<cv::Mat> &images, float thresh = 0.5, float nms = 0.45);
    std::vector<std::pair<void*, size_t>> infer(const std::vector<std::pair<void*, size_t>> &inputs);
    void showDetection(const std::vector<Detection> &dets, cv::Mat &image, float thresh,
        const std::vector<std::string> &names, bool draw);

private:
    bool preProcess(const std::vector<cv::Mat> &images);
    nvinfer1::IRuntime* mRuntime{nullptr};
    nvinfer1::ICudaEngine* mEngine{nullptr};
    nvinfer1::IExecutionContext* mContext{nullptr};
    std::shared_ptr<BufferManager> mBuffers;
    uint8_t *devImage;
    EngineInfo mInfo;
    std::vector<Yolo> mYolos;
    size_t mMemSize{960 * 720 * 3};
};

