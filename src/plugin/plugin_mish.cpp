#include "plugin_mish.h"
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>

cudaError_t cudaMishLayer(const void* input, void* output, uint32_t count, cudaStream_t stream);

namespace nvinfer1 {

int MishLayerPlugin::enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, 
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    const auto &inDim = inputDesc[0].dims;
    uint32_t batchSize = inDim.d[1] * inDim.d[2] * inDim.d[3];
    cudaMishLayer(inputs[0], outputs[0], inDim.d[0] * batchSize, stream);
    return 0;
}

PluginFieldCollection MishLayerPluginCreator::mFC{};

std::vector<PluginField> MishLayerPluginCreator::mPluginAttributes;

MishLayerPluginCreator::MishLayerPluginCreator()
{
    mPluginAttributes.clear();
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

IPluginV2DynamicExt* MishLayerPluginCreator::createPlugin(const char *name, const PluginFieldCollection *fc) noexcept
{
    MishLayerPlugin *obj = new MishLayerPlugin();
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2DynamicExt* MishLayerPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept
{
    MishLayerPlugin *obj = new MishLayerPlugin(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}
REGISTER_TENSORRT_PLUGIN(MishLayerPluginCreator);
}
