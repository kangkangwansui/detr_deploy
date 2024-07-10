#include "include/zkutils.h"
#include "include/DETR.h"

bool DETR::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(mInput.onnx_file_path.c_str(),static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    config->setMaxWorkspaceSize (64_MiB);
    if (mInput.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mInput.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllDynamicRanges(network.get(), 127.0f, 127.0f);
    }

    samplesCommon::enableDLA(builder.get(), config.get(), mInput.dlaCore);

    return true;
}

bool DETR::build(){
    std::cout << "the onnx file path is : " << mInput.onnx_file_path << std::endl;
    std::cout << "begin to build the network !" <<std::endl;
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder){
        std::cout << " fail in builder " << std::endl;
        return false;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        std::cout << " fail in network " << std::endl;
        return false;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        std::cout << " fail in config " << std::endl;
        return false;
    }

    auto parser
        = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        std::cout << " fail in parser " << std::endl;
        return false;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        std::cout << " fail in constructed " << std::endl;
        return false;
    }

    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        std::cout << " fail in profileStream " << std::endl;
        return false;
    }
    config->setProfileStream(*profileStream);

    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        std::cout << " fail in plan " << std::endl;
        return false;
    }

    if(mInput.isSaveEngine){
        std::string engine_output_path = mInput.EngineOutputFile + '/' + getOnnxName(mInput.onnx_file_path) + ".engine";
        std::ofstream p(engine_output_path, std::ios::binary);
        if (!p){
            std::cout << "could not open engine_output_path create this file" << std::endl;
            std::ofstream new_file(engine_output_path.c_str(), std::ios::out | std::ios::binary);
            if (new_file.is_open()) {
                std::cout << engine_output_path << " has been created." << std::endl;
            } else {
            std::cerr << "Failed to create " << engine_output_path << "." << std::endl;
            }
        }  
        p.write(reinterpret_cast<const char*>(plan->data()), plan->size());
        std::cout<< "引擎已经保存在：" << mInput.EngineOutputFile << std::endl;
    }

    SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
    if (!runtime)
    {
        std::cout << " fail in runtime " << std::endl;
        return false;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        runtime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        std::cout << " fail in mEngine " << std::endl;
        return false;
    }

    return true;
}

bool DETR::inferFromEngine(cv::Mat img,std::vector<OutputParam>& boxes_information){
    std::string engineflie = mInput.engineflie;
    std::ifstream file(engineflie,std::ios::binary);
    if(!file.good()){
        std::cout << "引擎加载失败" << std::endl;
        return false;
    }
    size_t size = 0;
    file.seekg(0, file.end);    //将读指针从文件末尾开始移动0个字节
    size = file.tellg();        //获取读指针的位置，即文件末尾的字节数

    if (size == 0) {
        std::cout << "引擎文件是空的" << std::endl;
        return false;
    }

    file.seekg(0, file.beg);    //将读指针从文件开头开始移动0个字节
    char* TRTmodelStream = new char[size];
    file.read(TRTmodelStream, size);
    file.close();

    auto runtime = SampleUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    auto engine = SampleUniquePtr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(TRTmodelStream, size, nullptr));
    delete[] TRTmodelStream;
    if (engine == nullptr) {
        return false;
    }

    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    void *buffers[3];

    nvinfer1::Dims input_dim = engine->getBindingDimensions(0);
    std::cout << "input_dim.nbDims: " << input_dim.nbDims << std::endl;
    int input_size = 1;
    for(int j = 0; j < input_dim.nbDims; j++){
        input_size *= input_dim.d[j];
    }
    std::cout << "input_size: " << input_size << std::endl;
    cudaMalloc(&buffers[0], input_size * sizeof(float));

    nvinfer1::Dims output_dim_1 = engine->getBindingDimensions(1);
    int output_size_1 = 1;
    for(int i = 0;i < output_dim_1.nbDims; i++){
        output_size_1 *= output_dim_1.d[i];
    }
    cudaMalloc(&buffers[1], output_size_1 * sizeof(float));
    std::cout << "output_size_1: " << output_size_1 << std::endl;

    nvinfer1::Dims output_dim_2 = engine->getBindingDimensions(2);
    int output_size_2 = 1;
    for(int i = 0;i < output_dim_2.nbDims; i++){
        output_size_2 *= output_dim_2.d[i];
    }
    cudaMalloc(&buffers[2], output_size_2 * sizeof(float));
    std::cout << "output_size_2: " << output_size_2 << std::endl;

    float *output_CpuBuffer_1 = new float[output_size_1]();
    float *output_CpuBuffer_2 = new float[output_size_2]();  // 给输出分配cpu内存，以接收GPU计算的结果

    cudaStream_t stream;
    cudaStreamCreate(&stream);  //在GPU创建进程束

    auto startTime = std::chrono::steady_clock::now();

    float *input_blob = new float[mInput.modelLenth * mInput.modelLenth * 3];
    //图像预处理
    preprocess_img(img, mInput, input_blob);

    auto Time1 = std::chrono::steady_clock::now();

    // 将输入数据从CPU传输到GPU
    cudaMemcpy(buffers[0], input_blob, input_size * sizeof(float), cudaMemcpyHostToDevice); 

    auto Time2 = std::chrono::steady_clock::now(); 

    context->enqueueV2(buffers, stream, nullptr);

    auto Time3 = std::chrono::steady_clock::now();

    cudaMemcpy(output_CpuBuffer_1, buffers[1],output_size_1 * sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(output_CpuBuffer_2, buffers[2],output_size_2 * sizeof(float),cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(stream);//等待输出数据传输完毕

    auto Time4 = std::chrono::steady_clock::now();

    int output_softmax_size = output_size_1;
    float* output_softmax_1 = new float[output_softmax_size]();
    for(int i = 0; i < output_softmax_size; i++){
        output_softmax_1[i] = 0;
    }
    softmax(output_CpuBuffer_1,output_softmax_1,mInput);

    get_boxes_information(output_softmax_1,output_CpuBuffer_2,mInput,boxes_information);

    std::cout << "the boxes_information size is : " << boxes_information.size() << std::endl;

    print_output_information(boxes_information);

    auto endTime = std::chrono::steady_clock::now();

    //毫秒级
    double per_process_time = std::chrono::duration<double, std::milli>(Time1 - startTime).count();
    double cpy_c2g_time = std::chrono::duration<double, std::milli>(Time2 - Time1).count();
    double infer_time = std::chrono::duration<double, std::milli>(Time3 - Time2).count();
    double cpy_g2c_time = std::chrono::duration<double, std::milli>(Time4 - Time3).count();
    double post_process_time = std::chrono::duration<double, std::milli>(endTime - Time4).count();
    double totall_time = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    std::cout << "图像处理时间 ： " << per_process_time << "毫秒" << std::endl;
    std::cout << "cpu->gpu时间 ： " << cpy_c2g_time << "毫秒" << std::endl;
    std::cout << "推理时间 ： " << infer_time << "毫秒" << std::endl;
    std::cout << "gpu->cpu时间 ： " << cpy_g2c_time << "毫秒" << std::endl;
    std::cout << "后处理时间 ： " << post_process_time << "毫秒" << std::endl;
    std::cout << "总时间时间 ： " << totall_time << "毫秒" << std::endl;

    // 释放资源
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    cudaFree(buffers[2]);
    cudaStreamDestroy(stream);

    delete[] input_blob;
    delete[] output_CpuBuffer_1;
    delete[] output_CpuBuffer_2;
    delete[] output_softmax_1;

    return true;

}

void interface(std::string yamlPath, cv::Mat img){
    bool infer_direct = true;

    YAML::Node config = YAML::LoadFile(yamlPath);
    InputParam input;
    std::vector<OutputParam> boxes_information;

    input.fp16 = config["fp16"].as<bool>();
    input.int8 = config["int8"].as<bool>();
    input.isBuild = config["isBuild"].as<bool>();
    input.isSaveEngine = config["isSaveEngine"].as<bool>();
    input.onnx_file_path = config["onnx_file_path"].as<std::string>();
    input.EngineOutputFile = config["EngineOutputFile"].as<std::string>();
    input.classes_txt = config["classes_txt"].as<std::string>();
    input.engineflie = config["engineflie"].as<std::string>();
    input.classes = config["classes"].as<int>();
    input.num = config["num"].as<int>();
    input.dlaCore = config["dlaCore"].as<int32_t>();
    input.modelLenth = config["modelLenth"].as<int>();
    input.confidence = config["confidence"].as<float>();

    DETR detr(input);

    if(input.isBuild){
        auto flag_build = detr.build();
        std::cout << "the build flag single is : " << flag_build << std::endl;
    }
    else{
        auto flag_infer_from_engine = detr.inferFromEngine(img,boxes_information);
        std::cout << "the infer flag from engine is : " << flag_infer_from_engine << std::endl;
    }
}

// PYBIND11_MODULE(example, m){
//     m.doc() = "pybind11 DETR plugin"; // optional module docstring
//     m.def("example", &interface, "A function to detr");
// }

int main(){
    std::cout << "welcome to use DETR !" << std::endl;
    cv::Mat img = cv::imread("/home/zhoukang/GithubProject/detr_from_bubbliiiing/CPP/image/car.jpg");
    if(img.empty()){
        std::cout << "could not open or find the image" << std::endl;
        return -1;
    }

    std::string yamlPath = "/home/zhoukang/GithubProject/detr-deploy/CPP/config.yaml";
    interface(yamlPath,img);
    return 0;
}