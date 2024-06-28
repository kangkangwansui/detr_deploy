#include "include/zkutils.h"
#include "include/DETR.h"

std::string getOnnxName(std::string onnx_file_path){
    std::size_t last_slash_idx = onnx_file_path.rfind('/');
    if (last_slash_idx != std::string::npos) {
        std::string filename = onnx_file_path.substr(last_slash_idx + 1);
        std::string onnxName = filename.substr(0, filename.size() - 5);
        return onnxName;
    } else {
        std::cout << "No file name found" << std::endl;
        return "";
    }
}

// 计算新的图像尺寸
std::pair<int, int> get_new_img_size(int height, int width, InputParam& input){
    double scale = std::min(static_cast<double>(input.modelLenth) / width, static_cast<double>(input.modelLenth) / height);

    int nw = static_cast<int>(width * scale);
    int nh = static_cast<int>(height * scale);

    return std::make_pair(nh, nw);
}

// 缩放图像
cv::Mat resize_image(const cv::Mat& image, InputParam& input){
    cv::Size original_size = image.size();
    input.original_height = original_size.height;
    input.original_width = original_size.width;

    std::pair<int, int> new_size = get_new_img_size(input.original_height, input.original_width, input);
    int h = new_size.first;
    int w = new_size.second;

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(w, h), 0, 0, cv::INTER_CUBIC);

    cv::Mat new_image(cv::Size(input.modelLenth, input.modelLenth), image.type(), cv::Scalar(128, 128, 128));

    int offset_x = (input.modelLenth - w) / 2;
    int offset_y = (input.modelLenth - h) / 2;

    cv::Mat roi = new_image(cv::Rect(offset_x, offset_y, w, h));

    resized_image.copyTo(roi);

    return new_image;
}

void preprocess_img(cv::Mat img, InputParam& input,float* input_blob){
    cv::Mat rgbImage;
    if (img.channels() == 3) {
        rgbImage = img.clone(); 
    } else {
        // 将图像转换为RGB格式
        cv::cvtColor(img, rgbImage, cv::COLOR_BGR2RGB);
    }

    cv::Mat resizeImage = resize_image(rgbImage, input);

    //将输入图片转为数据格式
    const int channels = resizeImage.channels();
    const int width = resizeImage.cols;
    const int height = resizeImage.rows;

    std::cout << "Hight is : " << height << "," << " Width is : " << width << std::endl;
    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                input_blob[c * width * height + h * width + w] = resizeImage.at<cv::Vec3b>(h, w)[c] / 255.0f;
            }
        }
    }
}

void softmax(const float* input, float* output, InputParam& mInput){
    for (int i = 0; i < mInput.num; ++i) {
        float max_val = -INFINITY; // 找到当前框的最大值
        for (int c = 0; c < mInput.classes; ++c) {
            max_val = fmaxf(max_val, input[i * mInput.classes + c]);
        }

        float sum_exp = 0.0f; // 计算指数和
        for (int c = 0; c < mInput.classes; ++c) {
            output[i * mInput.classes + c] = expf(input[i * mInput.classes + c] - max_val);
            sum_exp += output[i * mInput.classes + c];
        }

        // 归一化
        for (int c = 0; c < mInput.classes; ++c) {
            output[i * mInput.classes + c] /= sum_exp;
        }
    }
}

std::vector<std::string> get_classes_vector(std::string classes_txt){
    std::vector<std::string> classes;
    std::ifstream file(classes_txt);
    if (!file.is_open()) {
        std::cerr << "Unable to open file!" << std::endl;
    }

    std::string line;
    // 按行读取文件
    while (std::getline(file, line)) {
        // 处理每一行
        classes.push_back(line);
    }
    file.close();
    return classes;
}

void get_boxes_information(float* pred_logits,float* pred_boxes,InputParam input,std::vector<OutputParam>& boxes_information){
    std::vector<std::string> classes_vector = get_classes_vector(input.classes_txt);
    OutputParam box_information;
    for (int i = 0; i < input.num; ++i){
        for (int c = 0; c < input.classes - 1; ++c){
            if(pred_logits[i*input.classes + c] > input.confidence){
                box_information.box_idx = i;
                box_information.classes = classes_vector[c];
                box_information.score = pred_logits[i*input.classes + c];
                box_information.center_x = pred_boxes[i * 4];
                box_information.center_y = pred_boxes[i * 4 + 1];
                box_information.width = pred_boxes[i * 4 + 2];
                box_information.height = pred_boxes[i * 4 + 3];
                boxes_information.push_back(box_information);
                break;
            }
        }
    }
}

void print_output_information(std::vector<OutputParam>& boxes_information){
    for (const auto& param : boxes_information) {
            // 访问每个OutputParam对象的成员变量
            std::cout << "Box Index: " << param.box_idx << std::endl;
            std::cout << "Classes: " << param.classes << std::endl;
            std::cout << "Score: " << param.score << std::endl;
            std::cout << "Center X: " << param.center_x * 255 << std::endl;
            std::cout << "Center Y: " << param.center_y * 255  << std::endl;
            std::cout << "Width: " << param.width * 255  << std::endl;
            std::cout << "Height: " << param.height * 255  << std::endl;
            // 输出换行，为了更好的可读性
            std::cout << std::endl;
        }
}
