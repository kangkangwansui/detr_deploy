## DETR目标检测模型在Pytorch当中的实现，以及部署
---

## 目录
1. python ： 基于pytorch实现的DETR目标检测模型，支持训练和测试，以及生成onnx模型。
2. CPP    :  将onnx模型转为序列化的engine模型保存，反序列化engine模型，部署并推理。

## 所需环境
python环境
torch== 1.7.1
torchvision == 0.8.2
numpy == 1.24.4
onnx == 1.15.0
onnxruntime == 1.18.0
opencv-python == 4.6.0.66
tqdm == 4.66.4
scipy == 1.10.1

c++环境：
cmake == 3.16.3
tensorrt == 8.2.1.8
opencv == 4.8.0
yaml-cpp

## 文件下载
训练需要的数据集，训练模型的初始权重、150 epoches 训练后保存的权重文件、导出的onnx文件以及构建的engine文件，可在百度网盘中下载。  
链接：https://pan.baidu.com/s/16T2SXd6uNO2sFCzwc2WhNw 
提取码：6syh

python代码的实现可参考CSDN博客：睿智的目标检测65——Pytorch搭建DETR目标检测平台，代码来源：https://github.com/bubbliiiing/detr-pytorch

## 训练自己的数据集
1. 数据集的准备，按照VOC2007的格式准备自己的数据集，记住数据集存储的路径  

2. 生成数据集路径：cd python,修改voc_annotation.py文件下的classes_path和VOCdevkit_path路径；运行python voc_annotation.py，生成2007_train.txt，2007_val.txt文件。

3. 修改train.py文件的classes_path和model_path参数，其他参数根据自身需求修改；运行python train.py 训练结束后会在log文件下生成权重文件。

## 导出onnx模型
1. 修改export_onnx.py的resume和onnx_path参数，运行python export_onnx.py ,会在onnx文件夹下生成detr.onnx模型

## 根据onnx模型构建engine并推理
1. 解析onnx文件，构建网络，序列化模型：我们的项目是基于c++，利用tensorrt解析onnx模型构建网络，并序列化模型然后写入detr.engine文件并保存。具体操作为：cd ..退回根目录下，然后cd CPP进入c++语言下的目录，然后在config.yaml配置参数，具体参数如下：首先将isBuild设置为true(代表解析onnx文件，序列化模型)，isSaveEngine设置为true(保存engine文件)，onnx_file_path和classes_txt设置的路径与python生成的一致，EngineOutputFile指定engine文件保存的路径。最后cmake -B build(创建build文件和项目文件);cmake --build build(编译项目);build/DETR(运行项目，生成detr.engine文件)

2. 反序列化模型,并推理：config.yaml配置文件内的isBuild设置为false(直接加载engine文件，推理)；engineflie设置为engine文件生成的路径。最后build/DETR(运行项目，预测图片生成结果)



