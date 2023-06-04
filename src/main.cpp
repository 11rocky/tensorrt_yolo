#include <vector>
#include <chrono>
#include <argparse/argparse.hpp>
#include <pystring/pystring.h>
#include "logger.h"
#include "yolo_engine.h"
#include "yolo_infer.h"

void buildEngine(const argparse::ArgumentParser &args)
{
    int maxBatch = args.get<int>("--batch");
    std::string modelFile = args.get("--cfg");
    std::string weightFile = args.get("--weights");
    std::string outFile = args.get("--output");
    std::string dataType = args.get("--type");
    std::string calibrationTable = args.get("--calibration_table");
    std::string calibrationPath = args.get("--calibration_path");
    std::vector<std::string> dumps = args.get<std::vector<std::string>>("--dump");
    static const std::unordered_map<std::string, nvinfer1::DataType> dataTypes = {
        {"fp32", nvinfer1::DataType::kFLOAT},
        {"fp16", nvinfer1::DataType::kHALF},
        {"int8", nvinfer1::DataType::kINT8},
    };
    if (dataTypes.count(dataType) == 0) {
        LOG_ERROR("invalid data type: {}", dataType);
        return;
    }
    if (dataType == "int8") {
        if (calibrationPath.empty()) {
            LOG_ERROR("data type [int8] need calibrator images");
            return;
        }
        if (calibrationTable.empty()) {
            calibrationTable = fmt::format("{}.ctable", outFile);
        }
    }
    std::string enginePath = fmt::format("{}_{}_b{}.engine", outFile, dataType, maxBatch);
    if (!YoloEngine::build(modelFile, weightFile, enginePath, calibrationPath, calibrationTable, maxBatch, dataTypes.at(dataType), dumps)) {
        LOG_ERROR("build failed");
    }
}

void inference(const argparse::ArgumentParser &args)
{
    std::string engineFile = args.get("--engine");
    std::vector<std::string> inputs = args.get<std::vector<std::string>>("--inputs");
    std::string outputDir = args.get("--output");
    std::string namesFile = args.get("--names");
    float thresh = args.get<float>("--thresh");
    float nms = args.get<float>("--nms");
    int repeat = args.get<int>("--repeat");

    YoloInfer yolo;
    if (!yolo.init(engineFile)) {
        return;
    }

    std::vector<cv::Mat> images;
    for (auto &i : inputs) {
        images.emplace_back(cv::imread(i));
    }
    
    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < repeat; i++) {
        if (!yolo.infer(images)) {
            return;
        }
    }
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    LOG_INFO("inference avg time: {}ms\n", duration.count() / (float)repeat);
    yolo.saveInferOutputs(outputDir);
    std::vector<std::vector<Detection>> result = yolo.postProcess(images, thresh, nms);
    if (result.empty()) {
        return;
    }

    std::vector<std::string> names;
    std::ifstream ifs(namesFile);
    if (ifs) {
        std::string name;
        while (std::getline(ifs, name)) {
            names.push_back(name);
        }
        ifs.close();
    }
    for (int b = 0; b < result.size(); b++) {
        auto &bgr = images[b];
        yolo.showDetection(result[b], bgr, thresh, names, true);
        cv::imshow(inputs[b], bgr);
    }
    cv::waitKey();
}

int main(int argc, char **argv)
{
    argparse::ArgumentParser program(pystring::os::path::basename(argv[0]), "1.0", argparse::default_arguments::help);

    argparse::ArgumentParser buildCmd("build", "1.0", argparse::default_arguments::help);
    buildCmd.add_description("build tensorrt engine");
    buildCmd.add_argument("-c", "--cfg").help("yolo cfg file").required();
    buildCmd.add_argument("-w", "--weights").help("yolo weights file").required();
    buildCmd.add_argument("-o", "--output").help("output file name").required();
    buildCmd.add_argument("-t", "--type").help("datatype for infer: fp32, fp16, int8").default_value("fp32");
    buildCmd.add_argument("-b", "--batch").help("max inference batch").default_value(1).scan<'i', int>();
    buildCmd.add_argument("--calibration_path").help("calibration images path for int8").default_value("");
    buildCmd.add_argument("--calibration_table").help("calibration table file").default_value("");
    buildCmd.add_argument("--dump").help("dump tensors").default_value(std::vector<std::string>())
        .nargs(argparse::nargs_pattern::at_least_one);
    program.add_subparser(buildCmd);

    argparse::ArgumentParser inferCmd("infer", "1.0", argparse::default_arguments::help);
    inferCmd.add_description("inference with inputs");
    inferCmd.add_argument("-e", "--engine").help("engine file").required();
    inferCmd.add_argument("-i", "--inputs").help("input files").required().nargs(argparse::nargs_pattern::at_least_one);
    inferCmd.add_argument("-o", "--output").help("output dir").default_value(".");
    inferCmd.add_argument("--names").help("class names file").default_value("coco.names");
    inferCmd.add_argument("--thresh").help("thresh value").default_value(0.5f).scan<'g', float>();
    inferCmd.add_argument("--nms").help("nms value").default_value(0.45f).scan<'g', float>();
    inferCmd.add_argument("--repeat").help("repeat times").default_value(1).scan<'i', int>();
    program.add_subparser(inferCmd);

    try {
        program.parse_args(argc, argv);
        if (program.is_subcommand_used("build")) {
            buildEngine(buildCmd);
        } else if (program.is_subcommand_used("infer")) {
            inference(inferCmd);
        }
    }
    catch (const std::runtime_error& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }
    return 0;
}
