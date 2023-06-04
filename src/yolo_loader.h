#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <pystring/pystring.h>
#include <NvInfer.h>

class AttrValue {
public:
    AttrValue() = default;
    AttrValue(const std::string &value) : mValue(value) {}
    void get(int &val) const { val = std::stoi(mValue); }
    void get(float &val) const { val = std::stof(mValue); }
    void get(std::string &val) const { val = mValue; }
    void get(std::vector<int> &val) const
    {
        val = toList<int>([](const std::string &s) { return std::stoi(s); });
    }

    void get(std::vector<float> &val) const
    {
        val = toList<float>([](const std::string &s) { return std::stof(s); });
    }

    template<typename T>
    void set(const T &val)
    {
        mValue = std::to_string(val);
    }

    void set(const std::string &val)
    {
        mValue = val;
    }

private:
    std::string mValue;
    template<typename T, typename Functor>
    std::vector<T> toList(const Functor &functor) const
    {
        std::vector<std::string> items;
        std::vector<T> res;
        pystring::split(mValue, items, ",");
        for (auto &item : items) {
            item = pystring::strip(item);
            if (!item.empty()) {
                res.push_back(functor(item));
            }
        }
        return res;
    }
};

class AttrManager {
public:
    template<typename T>
    void setAttr(const std::string &key, const T &val) { mAttrs[key].set(val); }

    template<typename T>
    bool getAttr(const std::string &key, T &val) const
    {
        if (mAttrs.count(key) == 0) {
            return false;
        }
        mAttrs.at(key).get(val);
        return true;
    }

    void setAttrs(const std::unordered_map<std::string, AttrValue> &attrs) { mAttrs = attrs; }
    const std::unordered_map<std::string, AttrValue> &getAttrs() const { return mAttrs; }

    template<typename T>
    T getAttr(const std::string &key) const
    {
        T val;
        if (mAttrs.count(key) > 0) {
            mAttrs.at(key).get(val);
        }
        return val;
    }

    bool hasAttr(const std::string &key) const { return mAttrs.count(key) > 0; }
protected:
    std::unordered_map<std::string, AttrValue> mAttrs;
};

class Layer : public AttrManager {
public:
    void print() const;
    std::string type;
    int index;
    std::unordered_map<std::string, std::vector<float>> mData;
};

class Yolo {
public:
    std::string name;
    int classes;
    std::vector<int> mask;
    std::vector<float> biases;
    int num;
    int n;
    int h;
    int w;
    float *output;
};

class Network : public AttrManager {
public:
    void print() const;
    std::vector<Yolo> toTrtNetwork(nvinfer1::INetworkDefinition &trt, const std::string &inputName, bool dynamic,
        const std::unordered_set<std::string> &dumps);
    const std::vector<Layer>& getLayers() const { return mLayers; }
    friend class YoloLoader;
private:
    std::vector<Layer> mLayers;
    std::unordered_map<std::string, std::vector<float>> mData;
};

class YoloLoader {
public:
    static std::shared_ptr<Network> loadNetwork(const std::string &cfgFile, const std::string &weightFile);
private:
    static bool initWeights(const std::shared_ptr<Network> &network, const std::string &weightFile);
    static Layer parseLayer(const std::vector<std::string> &lines, int &cur);
    static void skipCommentAndBlank(const std::vector<std::string> &lines, int &cur);
};
