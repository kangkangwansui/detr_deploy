import os 
import sys

import argparse
import numpy as np 

import torch
import onnx
import torch.nn as nn 

from nets.detr import DETR

def export_onnx(args):
    model = DETR(args.backbone, 'sine', 256, args.num_classes, 100, pretrained=args.pretrained)
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location='cpu'))
    
    input_tensor = torch.randn(1, 3, 800, 800)

    model.eval()

    if not os.path.exists(args.onnx_path):
        os.makedirs(args.onnx_path)
        print(f"文件夹 '{args.onnx_path}' 已创建。")

    onnx_save_path = os.path.join(args.onnx_path, 'detr.onnx')


    torch.onnx.export(model, 
                    input_tensor, 
                    onnx_save_path, 
                    verbose=False, 
                    opset_version=12,
                    training = torch.onnx.TrainingMode.EVAL,
                    do_constant_folding = True,
                    input_names=['input'], 
                    output_names=['output_1','output_2'],
                    dynamic_axes    = None)
    
    # Checks
    model_onnx = onnx.load(onnx_save_path)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    
    if args.onnxSimplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, onnx_save_path)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='resnet50', type=str, help='name of backbone')
    parser.add_argument('--num_classes', default=20, type=int, help='number of classes')
    parser.add_argument('--pretrained', default=False, type=bool, help='Whether to use a pretrained model')
    parser.add_argument('--resume', default='/home/zhoukang/GithubProject/detr_from_bubbliiiing/logs/best_epoch_weights.pth', type=str, help='The path to load the model')
    parser.add_argument('--onnx_path', default="/home/zhoukang/GithubProject/detr_from_bubbliiiing/onnx", type=str, help='The path to save the onnx model')
    parser.add_argument('--onnxSimplify', default=True, type=bool, help='')

    args = parser.parse_args()

    export_onnx(args)