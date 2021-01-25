# under-water-detect


## pretrained model
https://open-mmlab.oss-cn-beijing.aliyuncs.com/v2.0/htc/htc_r50_fpn_20e_coco/htc_r50_fpn_20e_coco_20200319-fe28c577.pth


## Dataset

### convert label format from xml to json
```shell script
python ./tools/data_process/xml2coco.py
```
### generate pseudo json label of test image 
```

```

## Config

| 配置 | 设置 |
| :-----:| :----: 
| 模型 | CascadeRCNN + ResNeXt101 + FPN | 
| anchor_ratio | (0.5, 1, 2) | 
| 多尺度训练|(4096, 600), (4096, 1400)| 
| 多尺度预测|(4096, 600), (4096, 1000), (4096, 1400)|
| soft-NMS| (iou_thr=0.5, min_score=0.0001)|
| epoch| 1 x schedule(12 epoch)|
| steps| [8, 12]|
| fp16| 开启|
| pretrained|Hybrid Task Cascade(HTC)|


### lr computer
$ lr = 0.00125 \times \text{num_gpus} \times \text{img_per_gpu} $

### Albumentations

## config
```shell script
$ chmod +x tools/dist_train.sh

$ vim ./configs/htc_r50_fpn_1x.py
$ vim ./configs/htc_x101_64x4d_fpn_1x.py 
```
## setup mmdet
```shell script
python setup.py develop
```

## Train
```shell script
python ./tools
```


## Inference

## Trick


## Reference


* <https://github.com/Wakinguup/Underwater_detection>

