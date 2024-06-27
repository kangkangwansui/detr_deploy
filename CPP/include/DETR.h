#ifndef DETR_H
#define DETR_H

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "logging.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#include <cmath>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

using samplesCommon::SampleUniquePtr;

struct InputParam
{
    int num;
    int classes;
    int modelLenth{800};
    int original_width{1330};
    int original_height{1330};

    bool fp16{false};
    bool int8{false};
    bool isBuild{false};
    bool isSaveEngine{false};
    bool isWidthOverHeight{false};

    int32_t dlaCore{-1}; 
    float confidence{0.5};

    std::string classes_txt;
    std::string onnx_file_path;
    std::string EngineOutputFile;
    std::string engineflie;
};

struct OutputParam
{
    int box_idx;
    
    float score;
    float center_x;
    float center_y;
    float width;
    float height;

    std::string classes;
};

class DETR{
public:
    DETR(InputParam input): mInput(input), mEngine(nullptr){
    }

    bool build();

    bool inferFromEngine(cv::Mat img,std::vector<OutputParam>& boxes_information);

    bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser);

private:
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    InputParam mInput;
};

#endif // DETR_H