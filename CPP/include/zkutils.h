#ifndef ZKUTILS_H
#define ZKUTILS_H

#include "DETR.h"


std::string getOnnxName(std::string onnx_file_path);

std::pair<int, int> get_new_img_size(int height, int width, InputParam& input);

cv::Mat resize_image(const cv::Mat& image, InputParam& input);

void preprocess_img(cv::Mat img, InputParam& input,float* input_blob);

void softmax(const float* input, float* output, InputParam& mInput);

std::vector<std::string> get_classes_vector(std::string classes_txt);

void get_boxes_information(float* pred_logits,float* pred_boxes,InputParam input,std::vector<OutputParam>& boxes_information);

void print_output_information(std::vector<OutputParam>& boxes_information);

#endif // ZKUTILS_H