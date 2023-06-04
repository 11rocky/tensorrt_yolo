#include "yolo_loader.h"
#include <fstream>
#include "plugin/plugin_yolo.h"
#include "plugin/plugin_mish.h"
#include "utils.h"
#include "logger.h"

void Layer::print() const
{
    LOG_INFO("{}", type);
    for (auto &iter : mAttrs) {
        LOG_INFO("  {}: {}", iter.first, getAttr<std::string>(iter.first));
    }
}

void Network::print() const
{
    LOG_INFO("net");
    for (auto &iter : mAttrs) {
        LOG_INFO("  {}: {}", iter.first, getAttr<std::string>(iter.first));
    }
    for (const auto &layer : mLayers) {
        LOG_INFO("----------------------------------------------------------------");
        layer.print();
    }
    LOG_INFO("----------------------------------------------------------------");
}

static nvinfer1::ILayer* addActivationLayer(nvinfer1::INetworkDefinition &trt,
    nvinfer1::ITensor &input, const std::string &activation, bool &ok)
{
    ok = true;
    if (activation == "leaky") {
        auto leakyLayer = trt.addActivation(input, nvinfer1::ActivationType::kLEAKY_RELU);
        leakyLayer->setAlpha(0.1);
        return leakyLayer;
    } else if (activation == "mish") {
        nvinfer1::IPluginV2DynamicExt *mishPlugin = new nvinfer1::MishLayerPlugin();
        nvinfer1::ITensor *inputTensors[] = {&input};
        auto mishLayer = trt.addPluginV2(&inputTensors[0], 1, *mishPlugin);
        ok = mishLayer != nullptr;
        return mishLayer;
    } else if (activation == "linear") {
        return nullptr;
    }
    ok = false;
    LOG_ERROR("unsurpported activation: {}", activation);
    return nullptr;
}

static std::string getActName(const std::string &activation)
{
    if (activation == "leaky") {
        return "lrelu";
    } else if (activation == "mish") {
        return "lmish";
    } else if (activation == "linear") {
        return "llinear";
    }
    return "";
}

static nvinfer1::IScaleLayer* addBatchNorm2d(nvinfer1::INetworkDefinition &trt, nvinfer1::ITensor& input,
    Layer &layer, int outChannel)
{
    auto &bnBias = layer.mData.at("bias");
    auto &bnRunningMean = layer.mData.at("mean");
    auto &bnScale = layer.mData.at("scale");
    auto &bnRunningVar = layer.mData.at("var");
    nvinfer1::Weights shift{nvinfer1::DataType::kFLOAT, nullptr, outChannel};
    nvinfer1::Weights scale{nvinfer1::DataType::kFLOAT, nullptr, outChannel};
    nvinfer1::Weights power{nvinfer1::DataType::kFLOAT, nullptr, 0};
    auto &shiftWt = layer.mData["shift"];
    shiftWt.resize(outChannel);
    auto &scaleWt = layer.mData["scalec"];
    scaleWt.resize(outChannel);
    float eps = 1e-6;
    for (int i = 0; i < outChannel; ++i) {
        scaleWt[i] = bnScale[i] / std::sqrt(bnRunningVar[i] + eps);
        shiftWt[i] = bnBias.at(i) - bnRunningMean[i] * scaleWt[i];
    }
    scale.values = scaleWt.data();
    shift.values = shiftWt.data();
    return trt.addScale(input, nvinfer1::ScaleMode::kCHANNEL, shift, scale, power);
}

static std::string dimsToStr(const nvinfer1::Dims &dims)
{
    std::string out;
    for (int i = 0; i < dims.nbDims; i++) {
        out += std::to_string(dims.d[i]);
        if (i < dims.nbDims - 1) {
            out += "x";
        }
    }
    return out;
}

std::vector<Yolo> Network::toTrtNetwork(nvinfer1::INetworkDefinition &trt, const std::string &inputName, bool dynamic,
    const std::unordered_set<std::string> &dumps)
{
    std::vector<nvinfer1::ITensor*> blkOutputs;
    std::vector<Yolo> yolos;
    auto input = trt.addInput(inputName.c_str(), nvinfer1::DataType::kFLOAT,
        nvinfer1::Dims4{dynamic ? -1 : 1, getAttr<int>("channels"), getAttr<int>("height"), getAttr<int>("width")});
    for (int l = 0; l < mLayers.size(); l++) {
        nvinfer1::ITensor *fm = blkOutputs.empty() ? input : blkOutputs.back();
        auto &layer = mLayers[l];
        if (layer.type == "convolutional") {
            int bn = layer.hasAttr("batch_normalize") ? layer.getAttr<int>("batch_normalize") : 0;
            int filters = layer.getAttr<int>("filters");
            int kernelSize = layer.getAttr<int>("size");
            int pad = layer.getAttr<int>("pad");
            int stride = layer.getAttr<int>("stride");
            int padding = pad != 0 ? (kernelSize - 1) / 2 : 0;
            std::string activation = layer.getAttr<std::string>("activation");
            nvinfer1::Weights bias{nvinfer1::DataType::kFLOAT, layer.mData["bias"].data(), filters};
            nvinfer1::Weights weight{nvinfer1::DataType::kFLOAT, layer.mData["weight"].data(),
                filters * fm->getDimensions().d[1] * kernelSize * kernelSize};
            auto *convLayer = trt.addConvolutionNd(*fm, filters, nvinfer1::Dims2{kernelSize, kernelSize},
                weight, bn == 1 ? nvinfer1::Weights{} : bias);
            convLayer->setPaddingNd(nvinfer1::Dims2{padding, padding});
            convLayer->setStrideNd(nvinfer1::Dims2{stride, stride});
            convLayer->setName(fmt::format("{:03d}_convolutional", l + 1).c_str());
            convLayer->getOutput(0)->setName(fmt::format("{:03d}_convolutional", l + 1).c_str());
            nvinfer1::ILayer *lastLayer = convLayer;
            if (bn == 1) {
                auto bnLayer = addBatchNorm2d(trt, *(lastLayer->getOutput(0)), layer, filters);
                bnLayer->setName(fmt::format("{:03d}_convolutional_bn", l + 1).c_str());
                bnLayer->getOutput(0)->setName(fmt::format("{:03d}_convolutional_bn", l + 1).c_str());
                lastLayer = bnLayer;
            }
            bool ok;
            auto actLayer = addActivationLayer(trt, *(lastLayer->getOutput(0)), activation, ok);
            if (!ok) {
                return {};
            }
            if (actLayer != nullptr) {
                actLayer->setName(fmt::format("{:03d}_convolutional_{}", l + 1, getActName(activation)).c_str());
                actLayer->getOutput(0)->setName(fmt::format("{:03d}_convolutional_{}", l + 1, getActName(activation)).c_str());
                lastLayer = actLayer;
            }
            blkOutputs.push_back(lastLayer->getOutput(0));
            LOG_INFO("{:20} | {} -> {}", "convolutional", dimsToStr(fm->getDimensions()),
                dimsToStr(blkOutputs.back()->getDimensions()));
        } else if (layer.type == "maxpool") {
            int size = layer.getAttr<int>("size");
            int stride = layer.getAttr<int>("stride");
            auto *poolLayer = trt.addPoolingNd(*fm, nvinfer1::PoolingType::kMAX, nvinfer1::Dims2{size, size});
            poolLayer->setStrideNd(nvinfer1::Dims2{stride, stride});
            poolLayer->setPaddingMode(nvinfer1::PaddingMode::kSAME_UPPER);
            blkOutputs.push_back(poolLayer->getOutput(0));
            poolLayer->setName(fmt::format("{:03d}_maxpool", l + 1).c_str());
            poolLayer->getOutput(0)->setName(fmt::format("{:03d}_maxpool", l + 1).c_str());
            LOG_INFO("{:20} | {} -> {}", "maxpool", dimsToStr(fm->getDimensions()),
                dimsToStr(blkOutputs.back()->getDimensions()));
        } else if (layer.type == "upsample") {
            int stride = layer.getAttr<int>("stride");
            auto resizeLayer = trt.addResize(*fm);
            layer.mData["scale"] = {1.0f, 1.0f, 2.0f, 2.0f};
            resizeLayer->setResizeMode(nvinfer1::ResizeMode::kNEAREST);
            resizeLayer->setScales(layer.mData["scale"].data(), layer.mData["scale"].size());
            blkOutputs.push_back(resizeLayer->getOutput(0));
            resizeLayer->setName(fmt::format("{:03d}_upsample", l + 1).c_str());
            resizeLayer->getOutput(0)->setName(fmt::format("{:03d}_upsample", l + 1).c_str());
            LOG_INFO("{:20} | {} -> {}", "upsample", dimsToStr(fm->getDimensions()),
                dimsToStr(blkOutputs.back()->getDimensions()));
        } else if (layer.type == "route") {
            std::vector<int> ids = layer.getAttr<std::vector<int>>("layers");
            if (ids.size() == 1) {
                blkOutputs.push_back(blkOutputs[ids[0] > 0 ? ids[0] : l + ids[0]]);
            } else if (ids.size() > 1) {
                std::vector<nvinfer1::ITensor*> tensors;
                std::string dims = "[";
                for (auto id : ids) {
                    tensors.push_back(blkOutputs[id > 0 ? id : l + id]);
                    dims += dimsToStr(tensors.back()->getDimensions());
                    dims += ", ";
                }
                dims += "]";
                auto concatLayer = trt.addConcatenation(tensors.data(), tensors.size());
                concatLayer->setAxis(1);
                blkOutputs.push_back(concatLayer->getOutput(0));
                concatLayer->setName(fmt::format("{:03d}_route", l + 1).c_str());
                concatLayer->getOutput(0)->setName(fmt::format("{:03d}_route", l + 1).c_str());
                LOG_INFO("{:20} | {} -> {}", "concat", dims, dimsToStr(blkOutputs.back()->getDimensions()));
            }
        } else if (layer.type == "shortcut") {
            int from = layer.getAttr<int>("from");
            std::string activation = layer.getAttr<std::string>("activation");
            if (from < 0) {
                from += blkOutputs.size();
            }
            auto in1Tensor = blkOutputs.back();
            auto in2Tensor = blkOutputs[from];
            auto eltLayer = trt.addElementWise(*in1Tensor, *in2Tensor, nvinfer1::ElementWiseOperation::kSUM);
            eltLayer->setName(fmt::format("{:03d}_shortcut", l + 1).c_str());
            eltLayer->getOutput(0)->setName(fmt::format("{:03d}_shortcut", l + 1).c_str());
            nvinfer1::ILayer *lastLayer = eltLayer;
            bool ok;
            auto actLayer = addActivationLayer(trt, *(lastLayer->getOutput(0)), activation, ok);
            if (!ok) {
                return {};
            }
            if (actLayer != nullptr) {
                actLayer->setName(fmt::format("{:03d}_shortcut_{}", l + 1, getActName(activation)).c_str());
                actLayer->getOutput(0)->setName(fmt::format("{:03d}_shortcut_{}", l + 1, getActName(activation)).c_str());
                lastLayer = actLayer;
            }
            blkOutputs.push_back(eltLayer->getOutput(0));
            LOG_INFO("{:20} | {} + {} -> {}", "shortcut", dimsToStr(in1Tensor->getDimensions()),
                dimsToStr(in2Tensor->getDimensions()), dimsToStr(blkOutputs.back()->getDimensions()));
        } else if (layer.type == "yolo") {
            int classes = layer.getAttr<int>("classes");
            int numBoxes = layer.getAttr<std::vector<int>>("mask").size();
            nvinfer1::IPluginV2DynamicExt* yoloPlugin = new nvinfer1::YoloLayerPlugin(classes, numBoxes);
            nvinfer1::IPluginV2Layer* yoloLayer = trt.addPluginV2(&fm, 1, *yoloPlugin);
            auto outName = fmt::format("{:03d}_yolo", l + 1);
            yoloLayer->setName(outName.c_str());
            yoloLayer->getOutput(0)->setName(outName.c_str());
            blkOutputs.push_back(yoloLayer->getOutput(0));
            trt.markOutput(*blkOutputs.back());
            auto dim = dimsToVector(yoloLayer->getOutput(0)->getDimensions());
            Yolo yolo;
            yolo.name = outName;
            yolo.classes = layer.getAttr<int>("classes");
            yolo.num = layer.getAttr<int>("num");
            yolo.h = dim[dim.size() - 2];
            yolo.w = dim[dim.size() - 1];
            yolo.mask = layer.getAttr<std::vector<int>>("mask");
            yolo.n = static_cast<int>(yolo.mask.size());
            yolo.biases = layer.getAttr<std::vector<float>>("anchors");
            yolos.push_back(yolo);
            LOG_INFO("{:20} | {} -> {}", "yolo", dimsToStr(fm->getDimensions()),
                dimsToStr(blkOutputs.back()->getDimensions()));
        } else {
            LOG_ERROR("unsurpported layer: {}", layer.type);
            return {};
        }
    }

    for (int i = 0; i < trt.getNbLayers(); i++) {
        auto layer = trt.getLayer(i);
        if (dumps.count(layer->getOutput(0)->getName()) > 0 || dumps.count("dump_all") > 0) {
            LOG_INFO("dump: {}", layer->getOutput(0)->getName());
            trt.markOutput(*(layer->getOutput(0)));
        }
    }
    return yolos;
}

void YoloLoader::skipCommentAndBlank(const std::vector<std::string> &lines, int &cur)
{
    while (cur < lines.size()) {
        const std::string &line = lines[cur];
        if (line.empty() || line[0] == '#') {
            cur++;
            continue;
        }
        break;
    }
}

Layer YoloLoader::parseLayer(const std::vector<std::string> &lines, int &cur)
{
    skipCommentAndBlank(lines, cur);
    Layer layer;
    if (cur >= lines.size()) {
        return layer;
    }
    std::string line = lines[cur++];
    if (line.size() < 2 || line.front() != '[' || line.back() != ']') {
        return layer;
    }
    layer.type = pystring::slice(line, 1, line.size() - 1);
    while (cur < lines.size()) {
        skipCommentAndBlank(lines, cur);
        if (cur >= lines.size() || lines[cur].front() == '[') {
            break;
        }
        std::vector<std::string> keyVal;
        pystring::split(lines[cur++], keyVal, "=");
        if (keyVal.size() != 2) {
            continue;
        }
        layer.setAttr(pystring::strip(keyVal[0]), pystring::strip(keyVal[1]));
    }
    return layer;
}

bool YoloLoader::initWeights(const std::shared_ptr<Network> &network, const std::string &weightFile)
{
    std::ifstream ifs(weightFile);
    if (!ifs) {
        return false;
    }
    int header[5];
    ifs.read((char*)header, 5 * sizeof(int));
    int inputChannel = network->getAttr<int>("channels");
    for (int l = 0; l < network->mLayers.size(); l++) {
        auto &layer = network->mLayers[l];
        int inChannel = l == 0 ? inputChannel : network->mLayers[l - 1].getAttr<int>("out_channels");
        int outChannel = inChannel;
        if (layer.type == "convolutional") {
            int bn = layer.hasAttr("batch_normalize") ? layer.getAttr<int>("batch_normalize") : 0;
            int filters = layer.getAttr<int>("filters");
            int kernelSize = layer.getAttr<int>("size");
            layer.mData["bias"].resize(filters);
            ifs.read((char*)layer.mData["bias"].data(), filters * sizeof(float));
            if (bn == 1) {
                layer.mData["scale"].resize(filters);
                ifs.read((char*)layer.mData["scale"].data(), filters * sizeof(float));
                layer.mData["mean"].resize(filters);
                ifs.read((char*)layer.mData["mean"].data(), filters * sizeof(float));
                layer.mData["var"].resize(filters);
                ifs.read((char*)layer.mData["var"].data(), filters * sizeof(float));
            }
            int filterSize = filters * inChannel * kernelSize * kernelSize;
            layer.mData["weight"].resize(filterSize);
            ifs.read((char*)layer.mData["weight"].data(), filterSize * sizeof(float));
            outChannel = filters;
        } else if (layer.type == "route") {
            std::vector<int> layers = layer.getAttr<std::vector<int>>("layers");
            outChannel = 0;
            for (auto &idx : layers) {
                if (idx < 0) {
                    idx = l + idx;
                }
                outChannel += network->mLayers[idx].getAttr<int>("out_channels");
            }
        }
        layer.setAttr("out_channels", outChannel);
    }
    ifs.close();
    return true;
}

std::shared_ptr<Network> YoloLoader::loadNetwork(const std::string &cfgFile, const std::string &weightFile)
{
    std::ifstream ifs(cfgFile);
    if (!ifs) {
        return nullptr;
    }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(ifs, line)) {
        line = pystring::strip(line);
        lines.push_back(line);
    }
    ifs.close();
    auto network = std::make_shared<Network>();
    int cur = 0;
    while (cur < lines.size()) {
        Layer layer = parseLayer(lines, cur);
        if (layer.type == "net") {
            network->setAttrs(layer.getAttrs());
        } else {
            network->mLayers.push_back(layer);
            layer.index = network->mLayers.size() + 1;
        }
    }
    network->print();
    if (!initWeights(network, weightFile)) {
        LOG_ERROR("load weight failed: {}", weightFile);
    }
    return network;
}