
# PyTorch-Onnx-Tensorrt
Test yolov3-trt on jetson nano

## Requirements
1. Python 3
2. OpenCV
3. PyTorch
4. Onnx 1.4.1
5. Tensorrt
6. Mkdir yolov3_onnx，得到的yolov3-416.trt放在yolov3_onnx测试



## Downloading YoloV3 Configs and Weights
```
mkdir cfg
cd cfg 
原链接失效百度网盘地址
链接：https://pan.baidu.com/s/1CtWGfJQ2PWl6kLoMsH7Yhg 
提取码：gm4v

mkdir weights
cd weights
wget https://pjreddie.com/media/files/yolov3.weights
```

## Editing Config File
Inorder to Run the model in Pytorch or creating Onnx / Tensorrt File for different Input image Sizes ( 416, 608, 960 etc), you need to edit the Batch Size and Input image size in the config file - net info section.
```
batch=1
width=416
height=416
```


## Generating the Onnx File

```
python3 create_onnx.py --reso 416
```

## Generating the Tensorrt File

```
python3 onnx_to_tensorrt.py --model yolov3-416
 
```
Creating the Tensorrt engine takes some time. So have some patience.
## Test the YOLOv3 TensorRT engine with the "dog.jpg" image.(jetson nano run 2.55 FPS,jetson xavier nx run 11.59 FPS)

```
python3 trt_yolov3.py --model yolov3-416
                        --image --filename ${HOME}/Pictures/dog.jpg
```

## Run the "trt_yolov3.py" demo program.(jetson nano run 3.18 FPS,jetson xavier nx run 13.01FPS)

```
python3 trt_yolov3.py --usb --vid 0 --width 1280 --height 720   (or 640x480)

```
## evaluating mAP of the optimized YOLOv3 engine (jetson nano coco map@IOU=0.5 → 61.6%)

```
python3 eval_yolov3.py --model yolov3-416 
```
## 相关链接
* https://github.com/jkjung-avt/tensorrt_demos#yolov3
* https://github.com/Rapternmn/PyTorch-Onnx-Tensorrt
