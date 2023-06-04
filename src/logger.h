#pragma once

#include <NvInferRuntimeCommon.h>
#include <fmt/format.h>

using Serverity = nvinfer1::ILogger::Severity;

class Logger : public nvinfer1::ILogger {
public:
    explicit Logger(Serverity severity = Serverity::kINFO) : mServerity(severity) {}
    static Logger& getInstance()
    {
        static Logger logger;
        return logger;
    }

    void log(Serverity severity, const char *msg) noexcept override
    {
        if (static_cast<int>(severity) > static_cast<int>(mServerity)) {
            return;
        }
        if (severity == Serverity::kINTERNAL_ERROR) {
            fmt::println("[F] {}", msg);
        } else if (severity == Serverity::kERROR) {
            fmt::println("[E] {}", msg);
        } else if (severity == Serverity::kWARNING) {
            fmt::println("[W] {}", msg);
        } else if (severity == Serverity::kINFO) {
            fmt::println("[I] {}", msg);
        } else {
            fmt::println("[D] {}", msg);
        }
    }

    template<typename... Args>
    void log(Serverity severity, Args... args)
    {
        log(severity, fmt::format(args...).c_str());
    }

private:
    Serverity mServerity;
};

#define LOG_DEBUG(...) Logger::getInstance().log(Serverity::kVERBOSE, __VA_ARGS__)
#define LOG_INFO(...) Logger::getInstance().log(Serverity::kINFO, __VA_ARGS__)
#define LOG_WARN(...) Logger::getInstance().log(Serverity::kWARNING, __VA_ARGS__)
#define LOG_ERROR(...) Logger::getInstance().log(Serverity::kERROR, __VA_ARGS__)

