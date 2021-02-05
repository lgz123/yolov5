<a href="https://apps.apple.com/app/id1452689527" target="_blank">
<img src="https://user-images.githubusercontent.com/26833433/98699617-a1595a00-2377-11eb-8145-fc674eb9b1a7.jpg" width="1000"></a>
&nbsp

#### 1. 模型效果
yolov5按大小分为四个模型yolov5s、yolov5m、yolov5l、yolov5x，这四个模型的表现见下图：

This repository represents Ultralytics open-source research into future object detection methods, and incorporates lessons learned and best practices evolved over thousands of hours of training and evolution on anonymized client datasets. **All code and models are under active development, and are subject to modification or deletion without notice.** Use at your own risk.

<img src="https://user-images.githubusercontent.com/26833433/103594689-455e0e00-4eae-11eb-9cdf-7d753e2ceeeb.png" width="1000">** GPU Speed measures end-to-end time per image averaged over 5000 COCO val2017 images using a V100 GPU with batch size 32, and includes image preprocessing, PyTorch FP16 inference, postprocessing and NMS. EfficientDet data from [google/automl](https://github.com/google/automl) at batch size 8.

- **January 5, 2021**: [v4.0 release](https://github.com/ultralytics/yolov5/releases/tag/v4.0): nn.SiLU() activations, [Weights & Biases](https://wandb.ai/) logging, [PyTorch Hub](https://pytorch.org/hub/ultralytics_yolov5/) integration.
- **August 13, 2020**: [v3.0 release](https://github.com/ultralytics/yolov5/releases/tag/v3.0): nn.Hardswish() activations, data autodownload, native AMP.
- **July 23, 2020**: [v2.0 release](https://github.com/ultralytics/yolov5/releases/tag/v2.0): improved model definition, training and mAP.
- **June 22, 2020**: [PANet](https://arxiv.org/abs/1803.01534) updates: new heads, reduced parameters, improved speed and mAP [364fcfd](https://github.com/ultralytics/yolov5/commit/364fcfd7dba53f46edd4f04c037a039c0a287972).
- **June 19, 2020**: [FP16](https://pytorch.org/docs/stable/nn.html#torch.nn.Module.half) as new default for smaller checkpoints and faster inference [d4c6674](https://github.com/ultralytics/yolov5/commit/d4c6674c98e19df4c40e33a777610a18d1961145).

- 2020年10月29日：[v3.1 release](https://github.com/ultralytics/yolov5/releases/tag/v3.1)
- 2020年8月13日: [v3.0 release](https://github.com/wudashuo/yolov5/releases/tag/v3.0)
- 2020年7月23日: [v2.0 release](https://github.com/wudashuo/yolov5/releases/tag/v2.0)
- 2020年6月26日: [v1.0 release](https://github.com/wudashuo/yolov5/releases/tag/v1.0)

v2.0相对于v1.0是大版本更新，效果提升显著。v3.0使用nn.Hardwish()激活，图片推理速度下降10%，训练时显存占用增加10%（官方说法，我自己实测将近30%），训练时长不变。但是模型mAP会上升，对越小的模型收益越大。  
**注意**：v2.0和v3.0权重通用，但不兼容v1.0，不建议使用v1.0，建议使用最新版本代码。

| Model | size | AP<sup>val</sup> | AP<sup>test</sup> | AP<sub>50</sub> | Speed<sub>V100</sub> | FPS<sub>V100</sub> || params | GFLOPS |
|---------- |------ |------ |------ |------ | -------- | ------| ------ |------  |  :------: |
| [YOLOv5s](https://github.com/ultralytics/yolov5/releases)    |640 |36.8     |36.8     |55.6     |**2.2ms** |**455** ||7.3M   |17.0
| [YOLOv5m](https://github.com/ultralytics/yolov5/releases)    |640 |44.5     |44.5     |63.1     |2.9ms     |345     ||21.4M  |51.3
| [YOLOv5l](https://github.com/ultralytics/yolov5/releases)    |640 |48.1     |48.1     |66.4     |3.8ms     |264     ||47.0M  |115.4
| [YOLOv5x](https://github.com/ultralytics/yolov5/releases)    |640 |**50.1** |**50.1** |**68.7** |6.0ms     |167     ||87.7M  |218.8
| | | | | | | || |
| [YOLOv5x](https://github.com/ultralytics/yolov5/releases) + TTA |832 |**51.9** |**51.9** |**69.6** |24.9ms |40      ||87.7M  |1005.3

<!--- 
| [YOLOv5l6](https://github.com/ultralytics/yolov5/releases)   |640 |49.0     |49.0     |67.4     |4.1ms     |244     ||77.2M  |117.7
| [YOLOv5l6](https://github.com/ultralytics/yolov5/releases)   |1280 |53.0     |53.0     |70.8     |12.3ms     |81     ||77.2M  |117.7
--->

** AP<sup>test</sup> denotes COCO [test-dev2017](http://cocodataset.org/#upload) server results, all other AP results denote val2017 accuracy.  
** All AP numbers are for single-model single-scale without ensemble or TTA. **Reproduce mAP** by `python test.py --data coco.yaml --img 640 --conf 0.001 --iou 0.65`  
** Speed<sub>GPU</sub> averaged over 5000 COCO val2017 images using a GCP [n1-standard-16](https://cloud.google.com/compute/docs/machine-types#n1_standard_machine_types) V100 instance, and includes image preprocessing, FP16 inference, postprocessing and NMS. NMS is 1-2ms/img.  **Reproduce speed** by `python test.py --data coco.yaml --img 640 --conf 0.25 --iou 0.45`  
** All checkpoints are trained to 300 epochs with default settings and hyperparameters (no autoaugmentation). 
** Test Time Augmentation ([TTA](https://github.com/ultralytics/yolov5/issues/303)) runs at 3 image sizes. **Reproduce TTA** by `python test.py --data coco.yaml --img 832 --iou 0.65 --augment` 


## Requirements

Python 3.8 or later with all [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) dependencies installed, including `torch>=1.7`. To install run:
```bash
$ pip install -r requirements.txt
```


## Tutorials

* [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)&nbsp; 🚀 RECOMMENDED
* [Weights & Biases Logging](https://github.com/ultralytics/yolov5/issues/1289)&nbsp; 🌟 NEW
* [Multi-GPU Training](https://github.com/ultralytics/yolov5/issues/475)
* [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36)&nbsp; ⭐ NEW
* [ONNX and TorchScript Export](https://github.com/ultralytics/yolov5/issues/251)
* [Test-Time Augmentation (TTA)](https://github.com/ultralytics/yolov5/issues/303)
* [Model Ensembling](https://github.com/ultralytics/yolov5/issues/318)
* [Model Pruning/Sparsity](https://github.com/ultralytics/yolov5/issues/304)
* [Hyperparameter Evolution](https://github.com/ultralytics/yolov5/issues/607)
* [Transfer Learning with Frozen Layers](https://github.com/ultralytics/yolov5/issues/1314)&nbsp; ⭐ NEW
* [TensorRT Deployment](https://github.com/wang-xinyu/tensorrtx)


## Environments

YOLOv5 may be run in any of the following up-to-date verified environments (with all dependencies including [CUDA](https://developer.nvidia.com/cuda)/[CUDNN](https://developer.nvidia.com/cudnn), [Python](https://www.python.org/) and [PyTorch](https://pytorch.org/) preinstalled):

- **Google Colab and Kaggle** notebooks with free GPU: <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a> <a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
- **Google Cloud** Deep Learning VM. See [GCP Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/GCP-Quickstart)
- **Amazon** Deep Learning AMI. See [AWS Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/AWS-Quickstart)
- **Docker Image**. See [Docker Quickstart Guide](https://github.com/ultralytics/yolov5/wiki/Docker-Quickstart) <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>


## Inference

detect.py runs inference on a variety of sources, downloading models automatically from the [latest YOLOv5 release](https://github.com/ultralytics/yolov5/releases) and saving results to `runs/detect`.
```bash
$ pip install pip -U
$ pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

To run inference on example images in `data/images`:
```bash
$ python detect.py --source data/images --weights yolov5s.pt --conf 0.25

Namespace(agnostic_nms=False, augment=False, classes=None, conf_thres=0.25, device='', img_size=640, iou_thres=0.45, save_conf=False, save_dir='runs/detect', save_txt=False, source='data/images/', update=False, view_img=False, weights=['yolov5s.pt'])
Using torch 1.7.0+cu101 CUDA:0 (Tesla V100-SXM2-16GB, 16130MB)

Downloading https://github.com/ultralytics/yolov5/releases/download/v3.1/yolov5s.pt to yolov5s.pt... 100%|██████████████| 14.5M/14.5M [00:00<00:00, 21.3MB/s]

Fusing layers... 
Model Summary: 232 layers, 7459581 parameters, 0 gradients
image 1/2 data/images/bus.jpg: 640x480 4 persons, 1 buss, 1 skateboards, Done. (0.012s)
image 2/2 data/images/zidane.jpg: 384x640 2 persons, 2 ties, Done. (0.012s)
Results saved to runs/detect/exp
Done. (0.113s)
```
<img src="https://user-images.githubusercontent.com/26833433/97107365-685a8d80-16c7-11eb-8c2e-83aac701d8b9.jpeg" width="500">  

### PyTorch Hub

To run **batched inference** with YOLOv5 and [PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36):
```python
import torch
from PIL import Image

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


# Inference
result = model(imgs)
```


## Training

Run commands below to reproduce results on [COCO](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh) dataset (dataset auto-downloads on first use). Training times for YOLOv5s/m/l/x are 2/4/6/8 days on a single V100 (multi-GPU times faster). Use the largest `--batch-size` your GPU allows (batch sizes shown for 16 GB devices).
```bash
$ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                         yolov5m.yaml                           40
                                         yolov5l.yaml                       	24
                                         yolov5x.yaml                       	16
```
四个模型yolov5s/m/l/x使用COCO数据集在单个V100显卡上的训练时间为2/4/6/8天。
<img src="https://user-images.githubusercontent.com/26833433/90222759-949d8800-ddc1-11ea-9fa1-1c97eed2b963.png" width="900">
#### 2. 自定义训练
##### 2.1 准备标签
yolo格式的标签为txt格式的文件，文件名跟对应的图片名一样，除了后缀改为了.txt。
具体格式如下：
- 每个目标一行，整个图片没有目标的话不需要有txt文件
- 每行的格式为`class_num x_center y_center width height`
- 其中`class_num`取值为`0`至`total_class - 1`，框的四个值`x_center` `y_center` `width` `height`是相对于图片分辨率大小正则化的`0-1`之间的数，左上角为`(0,0)`，右下角为`(1,1)`
<img src="https://user-images.githubusercontent.com/26833433/91506361-c7965000-e886-11ea-8291-c72b98c25eec.jpg" width="900">
最终的标签文件应该是这样的：
<img src="https://user-images.githubusercontent.com/26833433/78174482-307bb800-740e-11ea-8b09-840693671042.png" width="900">

##### 2.2 数据规范
不同于DarkNet版yolo，图片和标签要分开存放。yolov5的代码会根据图片找标签，具体形式的把图片路径`/images/*.jpg`替换为`/labels/*.txt`，所以要新建两个文件夹，一个名为`images`存放图片，一个名为`labels`存放标签txt文件，如分训练集、验证集和测试集的话，还要再新建各自的文件夹，如图：
<img src="https://user-images.githubusercontent.com/26833433/83666389-bab4d980-a581-11ea-898b-b25471d37b83.jpg" width="900">

##### 2.3 准备yaml文件
自定义训练需要修改两个.yaml文件，一个是模型文件，一个是数据文件。
- 模型文件:可以根据你选择训练的模型，直接修改`./models`里的`yolov5s.yaml` / `yolov5m.yaml` / `yolov5l.yaml` / `yolov5x.yaml`文件，只需要将`nc: 80`中的80修改为你数据集的类别数。其他为模型结构不需要改。
- 数据文件:根据`./data`文件夹里的coco数据文件，制作自己的数据文件，在数据文件中定义训练集、验证集、测试集路径；定义总类别数；定义类别名称
    ```yaml
    # train and val data as 1) directory: path/images/, 2) file: path/images.txt, or 3) list: [path1/images/, path2/images/]
    train: ../coco128/images/train2017/
    val: ../coco128/images/val2017/
    test:../coco128/images/test2017/

    # number of classes
    nc: 80

    # class names
    names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 
            'teddy bear', 'hair drier', 'toothbrush']
    ```
##### 2.4 进行训练
训练直接运行`train.py`即可，后面根据需要加上指令参数，`--weights`指定权重，`--cfg`指定模型文件，`--data`指定数据文件，`--batch-size`指定batch大小，`--epochs`指定epoch，`--device`指定设备。一个简单的训练语句：
```bash
# 使用yolov5s模型训练coco128数据集5个epochs，batch size设为16
$ python train.py --batch 16 --epochs 5 --data ./data/coco128.yaml --cfg ./models/yolov5s.yaml --weights ''
```
#### 3. 训练指令说明
有参：
- `--weights` (**☆**)指定权重，如果不加此参数会默认使用COCO预训的`yolov5s.pt`，`--weights ''`则会随机初始化权重
- `--cfg` (**☆**)指定模型文件
- `--data` (**☆**)指定数据文件
- `--hyp`指定超参数文件
- `--epochs` (**☆**)指定epoch数，默认300
- `--batch-size` (**☆**)指定batch大小，默认`16`，官方推荐越大越好，用你GPU能承受最大的`batch size`，可简写为`--batch`
- `--img-size` 指定训练图片大小，默认`640`，可简写为`--img`
- `--name` 指定结果文件名，默认`result.txt`        
- `--device` (**☆**)指定训练设备，如`--device 0,1,2,3`
- `--local_rank` 分布式训练参数，不要自己修改！
- `--logdir` 指定训练过程存储路径，默认`./runs`
- `--workers` 指定dataloader的workers数量，默认`8`

无参： 
- `--rect`矩形训练
- `--resume` 继续训练，默认从最后一次训练继续
- `--nosave` 训练中途不存储模型，只存最后一个checkpoint
- `--notest` 训练中途不在验证集上测试，训练完毕再测试
- `--noautoanchor` 关闭自动锚点检测
- `--evolve`超参数演变
- `--bucket`使用gsutil bucket
- `--cache-images` 使用缓存图片训练，速度更快
- `--image-weights` 训练中对图片加权重
- `--multi-scale` 训练图片大小+/-50%变换
- `--single-cls` 单类训练
- `--adam` 使用torch.optim.Adam()优化器
- `--sync-bn` 使用SyncBatchNorm，只在分布式训练可用



## 检测
推理支持多种模式，图片、视频、文件夹、rtsp视频流和流媒体都支持。
#### 1. 快速检测命令
直接执行`detect.py`，指定一下要推理的目录即可，如果没有指定权重，会自动下载默认COCO预训练权重模型。手动下载：[Google Drive](https://drive.google.com/open?id=1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J)、[国内网盘待上传](待上传)。 
推理结果默认会保存到 `./inference/output`中。  
注意：每次推理会清空output文件夹，注意留存推理结果。
```bash
# 快速推理，--source 指定检测源，以下任意一种类型都支持：
$ python detect.py --source 0  # 本机默认摄像头
                            file.jpg  # 图片 
                            file.mp4  # 视频
                            path/  # 文件夹下所有媒体
                            path/*.jpg  # 文件夹下某类型媒体
                            rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa  # rtsp视频流
                            http://112.50.243.8/PLTV/88888888/224/3221225900/1.m3u8  # http视频流
```
#### 2. 自定义检测
使用权重`./weights/yolov5s.pt`去推理`./inference/images`文件夹下的所有媒体，并且推理置信度设为0.5:

```bash
$ python detect.py --source ./inference/images/ --weights ./weights/yolov5s.pt --conf 0.5
```

#### 3. 检测指令说明

自己根据需要加各种指令。

有参：
- `--source` (**必须**)指定检测来源
- `--weights` 指定权重，不指定的话会使用yolov5sCOCO预训练权重
- `--save-dir` 指定输出文件夹，默认./inference/output
- `--img-size` 指定推理图片分辨率，默认640，也可使用`--img`
- `--conf-thres` 指定置信度阈值，默认0.4，也可使用`--conf`
- `--iou-thres` 指定NMS(非极大值抑制)的IOU阈值，默认0.5
- `--device` 指定设备，如`--device 0` `--device 0,1,2,3` `--device cpu`
- `--classes` 只检测特定的类，如`--classes 0 2 4 6 8`

无参：
- `--view-img` 图片形式显示结果
- `--save-txt` 输出标签结果(yolo格式)为txt
- `--save-conf` 在输出标签结果txt中同样写入每个目标的置信度
- `--agnostic-nms` 使用agnostic NMS
- `--augment` 增强识别，[详情](https://github.com/ultralytics/yolov5/issues/303)
- `--update` 更新所有模型  


## 测试
#### 1.测试命令
首先明确，推理是直接检测图片，而测试是需要图片有相应的真实标签的，相当于检测图片后再把推理标签和真实标签做mAP计算。  
使用`./weights/yolov5x.pt`权重检测`./data/coco.yaml`里定义的测试集，测试集图片分辨率resize成672。
```bash
$ python test.py --weights ./weights/yolov5x.pt --data ./data/coco.yaml --img 672
```
#### 2.各指令说明
有参：
- `--weights` 测试所用权重，默认yolov5sCOCO预训练权重模型
- `--data` 测试所用的.yaml文件，默认使用`./data/coco128.yaml`
- `--batch-size` 测试用的batch大小，默认32，这个大小对结果无影响
- `--img-size` 测试集分辨率大小，默认640，测试建议使用更高分辨率
- `--conf-thres`目标置信度阈值，默认0.001
- `--iou-thres`NMS的IOU阈值，默认0.65
- `--task` 指定任务模式，train, val, 或者test,测试的话用`--task test`
- `--device` 指定设备，如`--device 0` `--device 0,1,2,3` `--device cpu`

无参：
- `--save-json`保存结果为json
- `--single-cls` 视为只有一类
- `--augment` 增强识别
~~- `--merge` 使用Merge NMS~~
- `--verbose` 输出各个类别的mAP
- `--save-txt` 输出标签结果(yolo格式)为txt


## 联系方式
如有代码bug请去[yolov5官方Issue](https://github.com/ultralytics/yolov5/issues)下提。

个人联系方式：<wudashuo@gmail.com>

## LICENSE
遵循yolov5官方[LICENSE](https://github.com/ultralytics/yolov5/blob/master/LICENSE)

