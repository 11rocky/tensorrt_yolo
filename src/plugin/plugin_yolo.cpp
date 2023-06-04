#include "plugin_yolo.h"
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>

cudaError_t cudaYoloLayer(const void* input, void* output, uint32_t batchs, uint32_t batchSize,
    uint32_t gridH, uint32_t gridW, uint32_t numClasses, uint32_t numBoxes, cudaStream_t stream);

namespace nvinfer1 {

int YoloLayerPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, 
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const auto &inDim = inputDesc[0].dims;
    uint32_t batchSize = inDim.d[1] * inDim.d[2] * inDim.d[3];
    cudaYoloLayer(inputs[0], outputs[0], inDim.d[0], batchSize, inDim.d[2], inDim.d[3], mClasses, mNumBoxes, stream);
    return 0;
}

PluginFieldCollection YoloLayerPluginCreator::mFC{};

std::vector<PluginField> YoloLayerPluginCreator::mPluginAttributes;

YoloLayerPluginCreator::YoloLayerPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("classes", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("num_boxes", nullptr, PluginFieldType::kINT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

IPluginV2DynamicExt* YoloLayerPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
{
    int classes = 0;
    int numBoxes = 0;
    auto fields = fc->fields;
    for (int i = 0; i < fc->nbFields; ++i) {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "classes")) {
            assert(fields[i].type == PluginFieldType::kINT32);
            classes = *(static_cast<const int*>(fields[i].data));
        } else if (!strcmp(attrName, "num_boxes")) {
            assert(fields[i].type == PluginFieldType::kINT32);
            numBoxes = *(static_cast<const int*>(fields[i].data));
        }
    }
    if (classes == 0 || numBoxes == 0) {
        return nullptr;
    }
    YoloLayerPlugin *obj = new YoloLayerPlugin(classes, numBoxes);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2DynamicExt* YoloLayerPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    YoloLayerPlugin *obj = new YoloLayerPlugin(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}
REGISTER_TENSORRT_PLUGIN(YoloLayerPluginCreator);
}
