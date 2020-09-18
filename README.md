# yolov5汉化版
## 简介
本仓库Fork自Ultralytics公司出品的yolov5，原仓库地址为：[ultralytics/yolov5](https://github.com/ultralytics/yolov5) ，**所有版权均属于原仓库作者所有**，请参见原仓库[License](https://github.com/ultralytics/yolov5/blob/master/LICENSE)。本人汉化自用，也方便各位国人使用。

####1. 模型效果
yolov5按大小分为四个模型yolov5s、yolov5m、yolov5l、yolov5x，这四个模型的表现见下图：

<img src="https://user-images.githubusercontent.com/26833433/90187293-6773ba00-dd6e-11ea-8f90-cd94afc0427f.png" width="1000">  

上图为基于5000张COCO val2017图像进行推理时，每张图像的平均端到端时间，batch size = 32, GPU：Tesla V100，这个时间包括图像预处理，FP16推理，后处理和NMS（非极大值抑制）。 EfficientDet的数据是从 [google/automl](https://github.com/google/automl) 仓库得到的（batch size = 8）。

####2. yolov5版本：

- 2020年8月13日: [v3.0 release](https://github.com/wudashuo/yolov5/releases/tag/v3.0)
- 2020年7月23日: [v2.0 release](https://github.com/wudashuo/yolov5/releases/tag/v2.0)
- 2020年6月26日: [v1.0 release](https://github.com/wudashuo/yolov5/releases/tag/v1.0)

v2.0相对于v1.0是大版本更新，效果提升显著。v3.0使用nn.Hardwish()激活，图片推理速度下降10%，训练时显存占用增加10%（官方说法，我自己实测将近30%），训练时长不变。但是模型mAP会上升，对越小的模型收益越大。  
**注意**：v2.0和v3.0权重通用，但不兼容v1.0，不建议使用v1.0，建议使用最新版本代码。


## 依赖
yolov5官方说Python版本需要≥3.8，但是我自用3.7也可以，但仍然推荐≥3.8。其他依赖都写在了[requirements.txt](https://github.com/wudashuo/yolov5/blob/master/requirements.txt) 里面。一键安装的话，打开命令行，cd到yolov5的文件夹里，输入：
```bash
$ pip install -r requirements.txt
```
pip安装慢的，请配置镜像源，下面是清华的镜像源。
```bash
$ pip install pip -U
$ pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```
想配其他镜像源直接把网址替换即可，如阿里云：https://mirrors.aliyun.com/pypi/simple/


## 训练
#### 1. 快速训练/复现训练
下载 [COCO数据集](https://github.com/ultralytics/yolov5/blob/master/data/scripts/get_coco.sh)，然后执行下面命令。根据你的显卡情况，使用最大的 `--batch-size` ，(下列命令中的batch size是16G显存的显卡推荐值).
```bash
$ python train.py --data coco.yaml --cfg yolov5s.yaml --weights '' --batch-size 64
                                         yolov5m.yaml                           40
                                         yolov5l.yaml                       	24
                                         yolov5x.yaml                       	16
```
四个模型yolov5s/m/l/x使用COCO数据集在单个V100显卡上的训练时间为2/4/6/8天。
<img src="https://user-images.githubusercontent.com/26833433/90222759-949d8800-ddc1-11ea-9fa1-1c97eed2b963.png" width="900">
#### 2. 自定义训练
**TODO**

## 推理（检测）
推理支持多种模式，图片、视频、文件夹、rtsp视频流和流媒体都支持。
#### 1. 快速检测：
直接执行`detect.py`，指定一下要推理的目录即可，如果没有指定权重，会自动下载默认COCO预训练权重模型。手动下载：[Google Drive](https://drive.google.com/open?id=1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J)、[国内网盘待上传](待上传)。 
推理结果默认会保存到 `./inference/output`中。  
注意：每次推理会清空output文件夹，注意留存推理结果。
```bash
# 快速推理，使用yolov5官方默认coco预训的权重进行检测，以下任意一种都支持：
$ python detect.py --source 0  # 本机默认摄像头
                            file.jpg  # 图片 
                            file.mp4  # 视频
                            path/  # 文件夹下所有媒体
                            path/*.jpg  # 文件夹下某类型媒体
                            rtsp://170.93.143.139/rtplive/470011e600ef003a004ee33696235daa  # rtsp视频流
                            http://112.50.243.8/PLTV/88888888/224/3221225900/1.m3u8  # http视频流
```
#### 2. 自定义检测
使用权重`./weights/yolov5s.pt`去推理`./inference/images`文件夹下的所有媒体，并且推理置信度设为0.5(默认0.4):

```bash
$ python detect.py --source ./inference/images/ --weights ./weights/yolov5s.pt --conf 0.5
```

#### 3. 各个参数说明

自己根据需要加各种参数。

有参：
- `--source` 指定检测来源(**必须**)
- `--weights` 指定权重，不指定的话会使用yolov5sCOCO预训练权重
- `--output` 指定输出文件夹，默认./inference/output
- `--img-size` 指定推理图片分辨率，默认640
- `--conf-thres` 指定置信度阈值，默认0.4
- `--iou-thres` 指定NMS(非极大值抑制)的IOU阈值，默认0.5
- `--device` 指定设备，如`--device 0` `--device 0,1,2,3` `--device cpu`
- `--classes` 只检测特定的类，如`--classes 0 2 4 6 8`

无参：
- `--view-img` 图片形式显示结果
- `--save-txt` 输出标签结果(yolo格式)为txt
- `--agnostic-nms` 只检测前景
- `--augment` 增强识别
- `--update` 更新所有模型  


## 测试
#### 1.测试命令
首先明确，推理是直接检测图片，而测试是需要图片有相应的真实标签的，相当于检测图片后再把推理标签和真实标签做mAP计算。  
使用`./weights/yolov5x.pt`权重检测`./data/coco.yaml`里定义的测试集，测试集图片分辨率resize成672。
```bash
$ python test.py --weights ./weights/yolov5x.pt --data ./data/coco.yaml --img 672
```
#### 2.各参数说明
有参：
- `--weights` 测试所用权重，默认yolov5sCOCO预训练权重模型
- `--data` 测试所用的.yaml文件，默认使用`./data/coco128.yaml`
- `--batch-size` 测试用的batch大小，默认32，这个大小对结果无影响
- `--img-size` 测试集分辨率大小，默认640，建议使用更高分辨率
- `--conf-thres`目标置信度阈值，默认0.001
- `--iou-thres`NMS的IOU阈值，默认0.65
- `--save-json`保存结果为json
- `--task` 任务模式，train, val, 或者test,测试的话用`--task test`
- `--device` 设备，
- `--single-cls` 视为只有一类
- `--augment` 增强识别
- `--merge` 使用Merge NMS
- `--verbose` 输出各个类别的mAP
- `--save-txt` 输出标签结果(yolo格式)为txt


**TODO**


## 联系方式
有任何问题请在[Issues](https://github.com/wudashuo/yolov5/issues)里提，方便大家都查看。
如有代码bug请去[yolov5官方Issue](https://github.com/ultralytics/yolov5/issues)下提。

个人联系方式：<wudashuo@gmail.com>

## LICENSE
遵循yolov5官方[LICENSE](https://github.com/ultralytics/yolov5/blob/master/LICENSE)

