#pragma once
#include <string>
#include <vector>
#include <cassert>
#include <NvInfer.h>

namespace nvinfer1 {

static const char *MISH_PLUGIN_NAME = "Mish";
static const char *MISH_PLGIN_VERSION = "1.0";

class MishLayerPlugin final : public IPluginV2DynamicExt {
public:
    MishLayerPlugin() = default;

    MishLayerPlugin(void const* serialData, size_t serialLength) {}

    int getNbOutputs() const noexcept override { return 1; }

    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs,
        IExprBuilder& exprBuilder) noexcept override
    {
        return inputs[0];
    }

    int initialize() noexcept override { return 0; }

    void terminate() noexcept override {}

    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs,
        const PluginTensorDesc* outputs, int nbOutputs) const noexcept override
    {
        return 0;
    }

    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, 
        const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    size_t getSerializationSize() const noexcept override { return 0; }

    void serialize(void* buffer) const noexcept override {}

    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut,
        int nbInputs, int nbOutputs) noexcept override
    {
        if (pos == 0) {
            return inOut[0].type == DataType::kFLOAT && inOut[0].format == TensorFormat::kLINEAR;
        } else if (pos == 1) {
            return inOut[1].type == DataType::kFLOAT && inOut[1].format == TensorFormat::kLINEAR;
        }
        return false;
    }

    const char* getPluginType() const noexcept override { return MISH_PLUGIN_NAME; }

    const char* getPluginVersion() const noexcept override { return MISH_PLGIN_VERSION; }

    void destroy() noexcept override { delete this; }

    IPluginV2DynamicExt* clone() const noexcept override
    {
        auto plugin = new MishLayerPlugin();
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }

    void setPluginNamespace(const char* pluginNamespace) noexcept override { mNamespace = pluginNamespace; }

    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

    DataType getOutputDataType(int index, const DataType* inputTypes,
        int nbInputs) const noexcept override { return DataType::kFLOAT; }

    void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, 
        const DynamicPluginTensorDesc* out, int nbOutputs) noexcept override
    {
        assert(nbInputs == 1);
        assert(nbOutputs == 1);
        assert(in[0].desc.type == DataType::kFLOAT);
        assert(out[0].desc.type == DataType::kFLOAT);
    }
private:
    std::string mNamespace;
};

class MishLayerPluginCreator : public IPluginCreator
{
public:
    MishLayerPluginCreator();

    const char* getPluginName() const noexcept override { return MISH_PLUGIN_NAME; }

    const char* getPluginVersion() const noexcept override { return MISH_PLGIN_VERSION;}

    const PluginFieldCollection* getFieldNames() noexcept override { return &mFC; }

    IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc)  noexcept override;

    IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;

    void setPluginNamespace(const char* libNamespace) noexcept override { mNamespace = libNamespace; }

    const char* getPluginNamespace() const noexcept  override { return mNamespace.c_str(); }

private:
    std::string mNamespace;
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};
}
