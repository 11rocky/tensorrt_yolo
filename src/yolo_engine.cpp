#include "yolo_engine.h"
#include "logger.h"
#include "utils.h"
#include "calibrator.h"
#include <fstream>


const char* YoloEngine::inputName = "input";
static constexpr uint32_t MAGIC_NUMBER = 0x56781234;

static void yoloToStream(std::ofstream &ofs, const Yolo &yolo)
{
    int size = yolo.name.size();
    ofs.write((char*)&size, sizeof(int));
    ofs.write(yolo.name.data(), size);
    ofs.write((char*)&yolo.classes, sizeof(int));
    size = yolo.mask.size();
    ofs.write((char*)&size, sizeof(int));
    ofs.write((char*)yolo.mask.data(), size * sizeof(int));
    size = yolo.biases.size();
    ofs.write((char*)&size, sizeof(int));
    ofs.write((char*)yolo.biases.data(), size * sizeof(float));
    ofs.write((char*)&yolo.num, sizeof(int));
    ofs.write((char*)&yolo.n, sizeof(int));
    ofs.write((char*)&yolo.h, sizeof(int));
    ofs.write((char*)&yolo.w, sizeof(int));
}

static void yoloFromStream(std::ifstream &ifs, Yolo &yolo)
{
    int size;
    ifs.read((char*)&size, sizeof(int));
    yolo.name.resize(size);
    ifs.read((char*)yolo.name.data(), size);
    ifs.read((char*)&yolo.classes, sizeof(int));
    ifs.read((char*)&size, sizeof(int));
    yolo.mask.resize(size);
    ifs.read((char*)yolo.mask.data(), size * sizeof(int));
    ifs.read((char*)&size, sizeof(int));
    yolo.biases.resize(size);
    ifs.read((char*)yolo.biases.data(), size * sizeof(float));
    ifs.read((char*)&yolo.num, sizeof(int));
    ifs.read((char*)&yolo.n, sizeof(int));
    ifs.read((char*)&yolo.h, sizeof(int));
    ifs.read((char*)&yolo.w, sizeof(int));
}

static void yoloInfo(const Yolo &yolo)
{
    LOG_INFO("   name: {}", yolo.name);
    LOG_INFO("classes: {}", yolo.classes);
    std::string mask;
    for (auto &m : yolo.mask) {
        mask += std::to_string(m) + ", ";
    }
    if (!mask.empty()) {
        mask.pop_back();
    }
    LOG_INFO("   mask: {}", mask);
    std::string biases;
    for (auto &m : yolo.biases) {
        biases += std::to_string(m) + ", ";
    }
    if (!biases.empty()) {
        biases.pop_back();
    }
    LOG_INFO(" biases: {}", biases);
    LOG_INFO("    num: {}", yolo.num);
    LOG_INFO("      n: {}", yolo.n);
    LOG_INFO("      h: {}", yolo.h);
    LOG_INFO("      w: {}", yolo.w);
}

bool YoloEngine::build(const std::string &cfgFile, const std::string &weightFile, const std::string &enginePath,
    const std::string &calibPath, const std::string &calibTable, int maxBatch, nvinfer1::DataType dataType,
    const std::vector<std::string> &dumps)
{
    if (maxBatch < 1) {
        maxBatch = 1;
    }
    std::unordered_set<std::string> dumpTensors(dumps.begin(), dumps.end());
    std::shared_ptr<Network> net = YoloLoader::loadNetwork(cfgFile, weightFile);
    if (net == nullptr) {
        return false;
    }
    EngineInfo info;
    info.type = dataType;
    info.maxBatch = maxBatch;
    info.inputC = net->getAttr<int>("channels");
    info.inputH = net->getAttr<int>("height");
    info.inputW = net->getAttr<int>("width");
    auto builder = std::shared_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(Logger::getInstance()));
    if ((dataType == nvinfer1::DataType::kINT8 && !builder->platformHasFastInt8()) ||
        (dataType == nvinfer1::DataType::kHALF && !builder->platformHasFastFp16())) {
        LOG_ERROR("platform doesn't support this precision: {}", (int)dataType);
        return false;
    }
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    auto config = std::shared_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->setProfileStream(stream);
    std::shared_ptr<Int8EntropyCalibrator> calibrator;
    if (dataType == nvinfer1::DataType::kHALF) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    } else if (dataType == nvinfer1::DataType::kINT8) {
        calibrator = std::make_shared<Int8EntropyCalibrator>(maxBatch, calibPath, calibTable,
            info.inputH, info.inputW);
        config->setFlag(nvinfer1::BuilderFlag::kINT8);
        config->setInt8Calibrator(calibrator.get());
    }
    if (maxBatch > 1) {
        nvinfer1::IOptimizationProfile *profile = builder->createOptimizationProfile();
        profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMIN,
            nvinfer1::Dims4(1, info.inputC, info.inputH, info.inputW));
        profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kOPT,
            nvinfer1::Dims4(maxBatch / 2, info.inputC, info.inputH, info.inputW));
        profile->setDimensions(inputName, nvinfer1::OptProfileSelector::kMAX,
            nvinfer1::Dims4(maxBatch, info.inputC, info.inputH, info.inputW));
        config->addOptimizationProfile(profile);
    }
    auto network = std::shared_ptr<nvinfer1::INetworkDefinition>(
        builder->createNetworkV2(1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    bool isDynamic =  maxBatch > 1;
    auto yolos = net->toTrtNetwork(*network, inputName, isDynamic, dumpTensors);
    if (yolos.empty()) {
        return false;
    }
    auto plan = std::shared_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    info.yoloNum = yolos.size();
    std::ofstream ofs(enginePath, std::ios::binary);
    if (!ofs) {
        LOG_ERROR("open file failed: {}", enginePath);
        return false;
    }
    uint32_t magicNum = MAGIC_NUMBER;
    ofs.write((char*)&magicNum, sizeof(uint32_t));
    ofs.write(reinterpret_cast<char*>(&info), sizeof(info));
    for (int i = 0; i < yolos.size(); i++) {
        yoloToStream(ofs, yolos[i]);
    }
    ofs.write(reinterpret_cast<char*>(plan->data()), plan->size());
    ofs.close();
    LOG_INFO("save engine to: {}, size={}", enginePath, plan->size());
    return true;
}

std::pair<nvinfer1::ICudaEngine*, nvinfer1::IRuntime*> YoloEngine::load(const std::string &enginePath,
    EngineInfo &info, std::vector<Yolo> &yolos)
{
    std::ifstream ifs(enginePath, std::ios::binary);
    if (!ifs) {
        LOG_ERROR("open file failed: {}", enginePath);
        return {nullptr, nullptr};
    }
    uint32_t magicNum;
    ifs.read((char*)&magicNum, sizeof(uint32_t));
    if (magicNum != MAGIC_NUMBER) {
        LOG_ERROR("invalid engine file: {}", enginePath);
        return {};
    }
    ifs.read((char*)&info, sizeof(info));
    LOG_INFO("maxBatch: {}", info.maxBatch);
    LOG_INFO("  inputC: {}", info.inputC);
    LOG_INFO("  inputH: {}", info.inputH);
    LOG_INFO("  inputW: {}", info.inputW);
    LOG_INFO(" yoloNum: {}", info.yoloNum);
    LOG_INFO("    type: {}", static_cast<int>(info.type));
    yolos.resize(info.yoloNum);
    for (int i = 0; i < info.yoloNum; i++) {
        yoloFromStream(ifs, yolos[i]);
        yoloInfo(yolos[i]);
    }
    auto cur = ifs.tellg();
    ifs.seekg(0, std::ios::end);
    auto end = ifs.tellg();
    size_t size = end - cur;
    ifs.seekg(cur);
    std::vector<char> stream(size);
    ifs.read(stream.data(), size);
    ifs.close();
    auto runtime = nvinfer1::createInferRuntime(Logger::getInstance());
    auto engine = runtime->deserializeCudaEngine(stream.data(), stream.size());
    return {engine, runtime};
}
