#include "yolo_infer.h"
#include <algorithm>
#include "plugin/image_preproc.h"
#include "logger.h"

inline int entryIndex(const Yolo &yolo, int location,  int entry)
{
    int n = location / (yolo.w * yolo.h);
    int loc = location % (yolo.w * yolo.h);
    return n * yolo.w * yolo.h * (4 + yolo.classes + 1) + entry * yolo.w * yolo.h + loc;
}

static int yoloNumDetections(const Yolo &yolo, float thresh)
{
    int h = yolo.h;
    int w = yolo.w;
    int count = 0;
    for (int i = 0; i < w * h; ++i){
        for(int n = 0; n < yolo.n; ++n){
            int idx = entryIndex(yolo, n * w * h + i, 4);
            if(yolo.output[idx] > thresh){
                ++count;
            }
        }
    }
    return count;
}

static Box getYoloBox(float *x, const std::vector<float> &biases, int n, int index, int i, int j,
    int lw, int lh, int w, int h, int stride)
{
    Box b;
    b.x = (i + x[index + 0 * stride]) / lw;
    b.y = (j + x[index + 1 * stride]) / lh;
    b.w = std::exp(x[index + 2 * stride]) * biases[2 * n] / w;
    b.h = std::exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
    return b;
}

static void correctYoloBoxes(Detection *dets, int n, int w, int h, int netw, int neth, bool relative)
{
    int new_w = 0;
    int new_h = 0;
    if (((float)netw / w) < ((float)neth / h)) {
        new_w = netw;
        new_h = (h * netw) / w;
    } else {
        new_h = neth;
        new_w = (w * neth) / h;
    }
    for (int i = 0; i < n; ++i) {
        Box &b = dets[i].box;
        b.x =  (b.x - (netw - new_w) / 2. / netw) / ((float)new_w / netw); 
        b.y =  (b.y - (neth - new_h) / 2./ neth) / ((float)new_h / neth); 
        b.w *= (float)netw / new_w;
        b.h *= (float)neth / new_h;
        if (!relative) {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
    }
}

static int getYoloDetections(const Yolo &yolo, int w, int h, int netw, int neth, float thresh,
    int *map, bool relative, Detection *dets)
{
    float *predictions = yolo.output;
    int count = 0;
    for (int i = 0; i < yolo.w * yolo.h; ++i){
        int row = i / yolo.w;
        int col = i % yolo.w;
        for(int n = 0; n < yolo.n; ++n){
            int objIndex = entryIndex(yolo, n * yolo.h * yolo.w + i, 4);
            float objectness = predictions[objIndex];
            if(objectness <= thresh) {
                continue;
            }
            int boxIndex  = entryIndex(yolo, n * yolo.h * yolo.w + i, 0);
            dets[count].box = getYoloBox(predictions, yolo.biases, yolo.mask[n], boxIndex, col, row,
                yolo.w, yolo.h, netw, neth, yolo.w * yolo.h);
            dets[count].objectness = objectness;
            dets[count].classes = yolo.classes;
            for(int j = 0; j < yolo.classes; ++j){
                int classIndex = entryIndex(yolo, n *yolo.h *yolo.w + i, 4 + 1 + j);
                float prob = objectness * predictions[classIndex];
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
            }
            ++count;
        }
    }
    correctYoloBoxes(dets, count, w, h, netw, neth, relative);
    return count;
}

static bool nmsComparator(const Detection &a, const Detection &b)
{
    float diff = 0;
    if (b.sortClass >= 0) {
        diff = a.prob[b.sortClass] - b.prob[b.sortClass];
    } else {
        diff = a.objectness - b.objectness;
    }
    return diff > 0;
}

static float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1 / 2;
    float l2 = x2 - w2 / 2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1 / 2;
    float r2 = x2 + w2 / 2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

static float boxIntersection(const Box &a, const Box &b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    return (w < 0 || h < 0) ? 0 : w * h;
}

static float boxUnion(const Box &a, const Box &b)
{
    float i = boxIntersection(a, b);
    float u = a.w * a.h + b.w * b.h - i;
    return u;
}

static float boxIou(const Box &a, const Box &b)
{
    return boxIntersection(a, b) / boxUnion(a, b);
}

static void doNmsSort(std::vector<Detection> &dets, int classes, float thresh)
{
    int total = dets.size();
    int k = total - 1;
    for(int i = 0; i <= k; ++i) {
        if(dets[i].objectness == 0) {
            std::swap(dets[i], dets[k]);
            --k;
            --i;
        }
    }

    for(k = 0; k < classes; ++k) {
        for(int i = 0; i < total; ++i) {
            dets[i].sortClass = k;
        }
        std::sort(dets.begin(), dets.end(), nmsComparator);
        for(int i = 0; i < total; ++i) {
            if (dets[i].prob[k] == 0) {
                continue;
            }
            Box a = dets[i].box;
            for(int j = i + 1; j < total; ++j) {
                Box b = dets[j].box;
                if (boxIou(a, b) > thresh){
                    dets[j].prob[k] = 0;
                }
            }
        }
    }
}

YoloInfer::~YoloInfer()
{
    if (devImage != nullptr) {
        cudaFree(devImage);
    }
    if (mEngine != nullptr) {
        delete mContext;
        delete mEngine;
        delete mRuntime;
    }
}

bool YoloInfer::preProcess(const std::vector<cv::Mat> &images)
{
    if (images.size() > mInfo.maxBatch) {
        LOG_ERROR("too many images, max: {}, current: {}", mInfo.maxBatch, images.size());
        return false;
    }
    size_t maxSize = 0;
    for (const auto &image : images) {
        if (image.empty()) {
            LOG_ERROR("invalid image");
            return false;
        }
        maxSize = std::max(image.total() * image.elemSize(), maxSize);
    }
    if (maxSize > mMemSize) {
        mMemSize = maxSize;
        cudaFree(devImage);
        if (cudaMalloc(&devImage, mMemSize) != cudaError_t::cudaSuccess) {
            LOG_ERROR("cuda malloc failed: size={}", mMemSize);
            return false;
        }
    }
    float *blob = reinterpret_cast<float*>(mBuffers->getInputBuffer(YoloEngine::inputName, false));
    for (int i = 0; i < images.size(); i++) {
        const auto &origin = images[i];
        auto dst = blob + i * mInfo.inputC * mInfo.inputH * mInfo.inputW;
        cudaMemcpy(devImage, origin.data, origin.total() * origin.elemSize(), cudaMemcpyHostToDevice);
        cudaImagePreProc(devImage, origin.rows, origin.cols, dst, mInfo.inputH, mInfo.inputW);
    }
    cudaDeviceSynchronize();
    return true;
}

bool YoloInfer::init(const std::string &engineFile)
{
    auto ret = YoloEngine::load(engineFile, mInfo, mYolos);
    mEngine = ret.first;
    mRuntime = ret.second;
    if (mEngine == nullptr || mRuntime == nullptr) {
        LOG_ERROR("load engine failed: {}", engineFile);
        return false;
    } else {
        LOG_INFO("load engine success");
    }
    mContext = mEngine->createExecutionContext();
    mBuffers = std::make_shared<BufferManager>(mEngine, mInfo.maxBatch);
    cudaMalloc(&devImage, mMemSize);
    return true;
}

void YoloInfer::saveInferOutputs(const std::string &path)
{
    for (int i = 0; i < mBuffers->getOutputNum(); i++) {
        auto outPath = fmt::format("{}/{}.bin", path, mBuffers->getOutputName(i));
        writeToFile(path, mBuffers->getOutputBuffer(i), mBuffers->getOutputBufferSize(i));
    }
}

bool YoloInfer::infer(const std::vector<cv::Mat> &images)
{
    if (!preProcess(images)) {
        return false;
    }
    mBuffers->setInferBatch(images.size());
    mContext->setInputShape(YoloEngine::inputName, nvinfer1::Dims4(images.size(), mInfo.inputC, mInfo.inputH, mInfo.inputW));
    mContext->executeV2(mBuffers->getDeviceBindings().data());
    mBuffers->copyOutputToHost();
    cudaDeviceSynchronize();
    return true;
}

std::vector<std::pair<void*, size_t>> YoloInfer::infer(const std::vector<std::pair<void*, size_t>> &inputs)
{
    if (inputs.size() > mInfo.maxBatch) {
        LOG_ERROR("too many inputs, max: {}, current: {}", mInfo.maxBatch, inputs.size());
        return {};
    }
    if (inputs.empty()) {
        return {};
    }
    size_t size = inputs.front().second;
    size_t totalSize = size;
    for (int i = 1; i < inputs.size(); i++) {
        if (size != inputs[i].second) {
            LOG_ERROR("input size not same, max: base = {}, inputs[{}] = ", size, i, inputs[i].second);
            return {};
        }
        totalSize += size;
    }
    size_t maxSize = mBuffers->getInputBufferMaxSize(YoloEngine::inputName);
    if (totalSize > maxSize) {
        LOG_ERROR("max total input size = {}, current = {}", maxSize, totalSize);
        return {};
    }
    mBuffers->setInferBatch(inputs.size());
    
    auto dst = (uint8_t*)mBuffers->getInputBuffer(YoloEngine::inputName, false);
    for (auto &i : inputs) {
        cudaMemcpy(dst, i.first, size, cudaMemcpyHostToDevice);
        dst += size;
    }
    cudaDeviceSynchronize();
    mContext->setInputShape(YoloEngine::inputName, nvinfer1::Dims4(inputs.size(), mInfo.inputC, mInfo.inputH, mInfo.inputW));
    mContext->executeV2(mBuffers->getDeviceBindings().data());
    mBuffers->copyOutputToHost();
    cudaDeviceSynchronize();
    std::vector<std::pair<void*, size_t>> result;
    for (int i = 0; i < mBuffers->getOutputNum(); i++) {
        result.push_back({mBuffers->getOutputBuffer(i), mBuffers->getOutputBufferSize(i)});
    }
    return result;
}

std::vector<std::vector<Detection>> YoloInfer::postProcess(const std::vector<cv::Mat> &images, float thresh, float nms)
{
    std::vector<std::vector<Detection>> result(images.size());
    for (int b = 0; b < result.size(); b++) {
        int nboxes = 0;
        for (int i = 0; i < mYolos.size(); i++) {
            auto &yolo = mYolos[i];
            yolo.output = reinterpret_cast<float*>(mBuffers->getOutputBuffer(yolo.name)) +
                b * yolo.n * (5 + yolo.classes) * yolo.h * yolo.w;
            nboxes += yoloNumDetections(yolo, thresh);
        }
        auto &dets = result[b];
        dets.resize(nboxes);
        for (int i = 0; i < dets.size(); i++) {
            dets[i].prob.resize(mYolos.front().classes);
        }
        if (dets.empty()) {
            LOG_WARN("detection is empty");
            return {};
        }

        Detection *detsOffset = dets.data();
        for(auto &yolo : mYolos) {
            int count = getYoloDetections(yolo, images[b].cols, images[b].rows, mInfo.inputW, mInfo.inputH, thresh,
                nullptr, true, detsOffset);
            detsOffset += count;
        }
        doNmsSort(dets, mYolos.front().classes, nms);
    }
    return result;
}

void YoloInfer::showDetection(const std::vector<Detection> &dets, cv::Mat &image, float thresh,
    const std::vector<std::string> &names, bool draw)
{
    if (dets.empty()) {
        return;
    }
    static std::vector<cv::Scalar> colorMap;
    if (colorMap.empty() && draw) {
        colorMap.resize(dets.front().classes);
        for (int i = 0; i < colorMap.size(); i++) {
            colorMap[i] = cv::Scalar(randomColor(), randomColor(), randomColor());
        }
    }
    
    for (const auto &det : dets) {
        std::string labelstr;
        int cls = -1;
        for (int j = 0; j < det.classes; ++j) {
            std::string label = std::to_string(j);
            if (names.size() == det.classes) {
                label = names[j];
            }
            if (det.prob[j] > thresh) {
                if (cls < 0) {
                    labelstr += fmt::format("{}({:.3f})", label, det.prob[j]);
                    cls = j;
                } else {
                    labelstr += ", ";
                    labelstr += fmt::format("{}({:.3f})", label, det.prob[j]);
                }
                auto &b = det.box;
                int left = (b.x - b.w / 2.) * image.cols;
                int right = (b.x + b.w / 2.) * image.cols;
                int top = (b.y - b.h / 2.) * image.rows;
                int bot = (b.y + b.h / 2.) * image.rows;
                if (draw) {
                    auto color = colorMap[j];
                    cv::putText(image, labelstr, cv::Point(left, top), cv::FONT_HERSHEY_SIMPLEX, 1, color, 2);
                    cv::rectangle(image, cv::Rect(left, top, right - left, bot - top), color, 2);
                } else {
                    LOG_INFO("{}: [({}, {}), ({}, {})]", labelstr, left, top, right, bot);
                }
            }
        }
    }
}
