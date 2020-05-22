"""eval_yolov3.py
This script is for evaluating mAP (accuracy) of YOLOv3 models.
"""
import time
import torch
import os
import sys
import json
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from progressbar import progressbar

from utils.yolov3 import *
from utils.yolov3_classes import yolov3_cls_to_ssd
import tensorrt as trt
import pycuda.driver as cuda







def parse_args():
    """Parse input arguments."""
    desc = 'Evaluate mAP of SSD model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model', type=str, default='yolov3-416',
                        choices=['yolov3-288', 'yolov3-416', 'yolov3-608',
                                 'yolov3-tiny-288', 'yolov3-tiny-416'])
    args = parser.parse_args()
    return args



def main():
    args = parse_args()
    yolo_dim = int(args.model.split('-')[-1])  # 416 or 608
    yolov3_trt = TrtYOLOv3(args.model, (yolo_dim, yolo_dim))
    #TRTbin = 'yolov3_onnx/yolov3-416.trt' 
    #with open('yolov3_onnx/yolov3-416.trt', 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.INFO)) as runtime:
        #yolov3_trt= runtime.deserialize_cuda_engine(f.read())
    #network_size=(416,416)
    #test_data = torch.randn(1,3,416,416,dtype=torch.float).cuda()
    test_data = torch.rand(size=(100, 1, 3, 416, 416)).cuda()
    

    print("Start test speed")
    
    
    

    start = time.time()
    for i in range(test_data.size()[0]):
        detections = yolov3_trt(test_data[0])
        end = time.time()
        
        print("Fp32 TensorRT Speed:", 1 / (end - start) * test_data.size()[0], "Hz")



if __name__ == '__main__':
    main()
