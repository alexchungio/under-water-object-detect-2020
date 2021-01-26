# under-water-detect


## pretrained model
https://open-mmlab.oss-cn-beijing.aliyuncs.com/v2.0/htc/htc_r50_fpn_20e_coco/htc_r50_fpn_20e_coco_20200319-fe28c577.pth


## Dataset

### convert label format from xml to json
```shell script
python ./tools/data_process/xml2coco.py
```
### generate pseudo json label of test image 
```shell script
python ./tools/data_process/generate_test_anns.py
```

## EDA

[data analysis](./docs/data_analysis.ipynb)


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

$ vim cascade_r50_fpn_1x.py
$ vim cascade_x101_64x4d_fpn_1x.py 
```
## setup mmdet
```shell script
python setup.py develop
```

## Train
```shell script
python ./tools/train.py ./configs/cascade_r50_fpn_1x.py --no-validate --gpus=1

```


## Inference

## Trick


## Log
### loss NAN
```shell script
2021-01-25 19:25:22,220 - mmdet - INFO - workflow: [('train', 1)], max: 12 epochs
2021-01-25 19:25:44,459 - mmdet - INFO - Epoch [1][50/5455]	lr: 4.983e-04, eta: 8:02:25, time: 0.443, data_time: 0.047, memory: 3654, loss_rpn_cls: 0.1925, loss_rpn_bbox: 0.0304, s0.loss_cls: 0.5629, s0.acc: 87.7969, s0.loss_bbox: 0.1178, s1.loss_cls: 0.2588, s1.acc: 90.2930, s1.loss_bbox: 0.0968, s2.loss_cls: 0.1165, s2.acc: 94.1406, s2.loss_bbox: 0.0351, loss: 1.4106, grad_norm: 10.6679
2021-01-25 19:26:04,422 - mmdet - INFO - Epoch [1][100/5455]	lr: 5.817e-04, eta: 7:38:30, time: 0.399, data_time: 0.003, memory: 3654, loss_rpn_cls: 0.0565, loss_rpn_bbox: 0.0194, s0.loss_cls: 0.2479, s0.acc: 92.6836, s0.loss_bbox: 0.1215, s1.loss_cls: 0.1272, s1.acc: 91.4200, s1.loss_bbox: 0.1315, s2.loss_cls: 0.0633, s2.acc: 92.3482, s2.loss_bbox: 0.0634, loss: 0.8306, grad_norm: 6.3832
2021-01-25 19:26:21,410 - mmdet - INFO - Epoch [1][150/5455]	lr: 6.650e-04, eta: 7:08:42, time: 0.340, data_time: 0.003, memory: 3654, loss_rpn_cls: 0.0738, loss_rpn_bbox: 0.0234, s0.loss_cls: 0.2896, s0.acc: 91.3672, s0.loss_bbox: 0.1508, s1.loss_cls: 0.1442, s1.acc: 91.0758, s1.loss_bbox: 0.1669, s2.loss_cls: 0.0735, s2.acc: 89.6029, s2.loss_bbox: 0.0791, loss: 1.0013, grad_norm: 8.8747
2021-01-25 19:26:37,296 - mmdet - INFO - Epoch [1][200/5455]	lr: 7.483e-04, eta: 6:47:40, time: 0.318, data_time: 0.003, memory: 3654, loss_rpn_cls: 0.0615, loss_rpn_bbox: 0.0221, s0.loss_cls: 0.2475, s0.acc: 93.0781, s0.loss_bbox: 0.1243, s1.loss_cls: 0.1164, s1.acc: 93.5012, s1.loss_bbox: 0.1407, s2.loss_cls: 0.0559, s2.acc: 93.1238, s2.loss_bbox: 0.0658, loss: 0.8342, grad_norm: 7.5786
2021-01-25 19:26:53,736 - mmdet - INFO - Epoch [1][250/5455]	lr: 8.317e-04, eta: 6:37:21, time: 0.329, data_time: 0.003, memory: 3672, loss_rpn_cls: 0.0812, loss_rpn_bbox: 0.0254, s0.loss_cls: 0.2639, s0.acc: 92.2188, s0.loss_bbox: 0.1250, s1.loss_cls: 0.1282, s1.acc: 92.6880, s1.loss_bbox: 0.1348, s2.loss_cls: 0.0610, s2.acc: 92.6380, s2.loss_bbox: 0.0637, loss: 0.8832, grad_norm: 7.2209
2021-01-25 19:27:09,358 - mmdet - INFO - Epoch [1][300/5455]	lr: 9.150e-04, eta: 6:27:25, time: 0.312, data_time: 0.003, memory: 3672, loss_rpn_cls: 0.6478, loss_rpn_bbox: 0.0342, s0.loss_cls: nan, s0.acc: 14.7109, s0.loss_bbox: nan, s1.loss_cls: nan, s1.acc: 14.4101, s1.loss_bbox: nan, s2.loss_cls: nan, s2.acc: 14.3531, s2.loss_bbox: nan, loss: nan, grad_norm: nan
2021-01-25 19:27:24,706 - mmdet - INFO - Epoch [1][350/5455]	lr: 9.983e-04, eta: 6:19:25, time: 0.307, data_time: 0.003, memory: 3763, loss_rpn_cls: 0.7337, loss_rpn_bbox: 0.0419, s0.loss_cls: nan, s0.acc: 0.1523, s0.loss_bbox: nan, s1.loss_cls: nan, s1.acc: 2.6650, s1.loss_bbox: nan, s2.loss_cls: nan, s2.acc: 2.4035, s2.loss_bbox: nan, loss: nan, grad_norm: nan
2021-01-25 19:27:41,646 - mmdet - INFO - Epoch [1][400/5455]	lr: 1.082e-03, eta: 6:17:39, time: 0.339, data_time: 0.003, memory: 3763, loss_rpn_cls: 0.7058, loss_rpn_bbox: 0.0391, s0.loss_cls: nan, s0.acc: 0.0742, s0.loss_bbox: nan, s1.loss_cls: nan, s1.acc: 0.4444, s1.loss_bbox: nan, s2.loss_cls: nan, s2.acc: 3.6349, s2.loss_bbox: nan, loss: nan, grad_norm: nan
2021-01-25 19:27:57,810 - mmdet - INFO - Epoch [1][450/5455]	lr: 1.165e-03, eta: 6:14:21, time: 0.323, data_time: 0.003, memory: 3763, loss_rpn_cls: 0.6784, loss_rpn_bbox: 0.0438, s0.loss_cls: nan, s0.acc: 0.1875, s0.loss_bbox: nan, s1.loss_cls: nan, s1.acc: 1.7714, s1.loss_bbox: nan, s2.loss_cls: nan, s2.acc: 2.1500, s2.loss_bbox: nan, loss: nan, grad_norm: nan
2021-01-25 19:28:13,354 - mmdet - INFO - Epoch [1][500/5455]	lr: 1.248e-03, eta: 6:10:19, time: 0.311, data_time: 0.003, memory: 3763, loss_rpn_cls: 0.6525, loss_rpn_bbox: 0.0467, s0.loss_cls: nan, s0.acc: 0.0703, s0.loss_bbox: nan, s1.loss_cls: nan, s1.acc: 1.0330, s1.loss_bbox: nan, s2.loss_cls: nan, s2.acc: 0.6964, s2.loss_bbox: nan, loss: nan, grad_norm: nan
2021-01-25 19:28:30,701 - mmdet - INFO - Epoch [1][550/5455]	lr: 1.250e-03, eta: 6:10:30, time: 0.347, data_time: 0.003, memory: 3763, loss_rpn_cls: 0.6215, loss_rpn_bbox: 0.0336, s0.loss_cls: nan, s0.acc: 0.0625, s0.loss_bbox: nan, s1.loss_cls: nan, s1.acc: 1.1000, s1.loss_bbox: nan, s2.loss_cls: nan, s2.acc: 0.3000, s2.loss_bbox: nan, loss: nan, grad_norm: nan
2021-01-25 19:28:46,898 - mmdet - INFO - Epoch [1][600/5455]	lr: 1.250e-03, eta: 6:08:33, time: 0.324, data_time: 0.003, memory: 3763, loss_rpn_cls: 0.5968, loss_rpn_bbox: 0.0351, s0.loss_cls: nan, s0.acc: 2.0742, s0.loss_bbox: nan, s1.loss_cls: nan, s1.acc: 0.3676, s1.loss_bbox: nan, s2.loss_cls: nan, s2.acc: 1.7367, s2.loss_bbox: nan, loss: nan, grad_norm: nan
2021-01-25 19:29:04,411 - mmdet - INFO - Epoch [1][650/5455]	lr: 1.250e-03, eta: 6:09:02, time: 0.350, data_time: 0.003, memory: 3763, loss_rpn_cls: 0.5764, loss_rpn_bbox: 0.0393, s0.loss_cls: nan, s0.acc: 2.0664, s0.loss_bbox: nan, s1.loss_cls: nan, s1.acc: 0.6667, s1.loss_bbox: nan, s2.loss_cls: nan, s2.acc: 2.3714, s2.loss_bbox: nan, loss: nan, grad_norm: nan
2021-01-25 19:29:22,212 - mmdet - INFO - Epoch [1][700/5455]	lr: 1.250e-03, eta: 6:09:52, time: 0.356, data_time: 0.003, memory: 3763, loss_rpn_cls: 0.5571, loss_rpn_bbox: 0.0401, s0.loss_cls: nan, s0.acc: 0.0586, s0.loss_bbox: nan, s1.loss_cls: nan, s1.acc: 3.4667, s1.loss_bbox: nan, s2.loss_cls: nan, s2.acc: 1.3889, s2.loss_bbox: nan, loss: nan, grad_norm: nan
2021-01-25 19:29:41,731 - mmdet - INFO - Epoch [1][750/5455]	lr: 1.250e-03, eta: 6:13:00, time: 0.390, data_time: 0.003, memory: 3811, loss_rpn_cls: 0.5406, loss_rpn_bbox: 0.0409, s0.loss_cls: nan, s0.acc: 0.0742, s0.loss_bbox: nan, s1.loss_cls: nan, s1.acc: 1.5111, s1.loss_bbox: nan, s2.loss_cls: nan, s2.acc: 1.0667, s2.loss_bbox: nan, loss: nan, grad_norm: nan
2021-01-25 19:29:59,933 - mmdet - INFO - Epoch [1][800/5455]	lr: 1.250e-03, eta: 6:13:56, time: 0.364, data_time: 0.003, memory: 3858, loss_rpn_cls: 0.5265, loss_rpn_bbox: 0.0438, s0.loss_cls: nan, s0.acc: 1.5273, s0.loss_bbox: nan, s1.loss_cls: nan, s1.acc: 3.2214, s1.loss_bbox: nan, s2.loss_cls: nan, s2.acc: 1.3485, s2.loss_bbox: nan, loss: nan, grad_norm: nan
2021-01-25 19:30:18,538 - mmdet - INFO - Epoch [1][850/5455]	lr: 1.250e-03, eta: 6:15:14, time: 0.372, data_time: 0.003, memory: 3858, loss_rpn_cls: 0.5190, loss_rpn_bbox: 0.0485, s0.loss_cls: nan, s0.acc: 0.0430, s0.loss_bbox: nan, s1.loss_cls: nan, s1.acc: 2.2333, s1.loss_bbox: nan, s2.loss_cls: nan, s2.acc: 2.6553, s2.loss_bbox: nan, loss: nan, grad_norm: nan
2021-01-25 19:30:35,671 - mmdet - INFO - Epoch [1][900/5455]	lr: 1.250e-03, eta: 6:14:36, time: 0.343, data_time: 0.003, memory: 3858, loss_rpn_cls: 0.5023, loss_rpn_bbox: 0.0420, s0.loss_cls: nan, s0.acc: 0.0430, s0.loss_bbox: nan, s1.loss_cls: nan, s1.acc: 1.5000, s1.loss_bbox: nan, s2.loss_cls: nan, s2.acc: 0.1905, s2.loss_bbox: nan, loss: nan, grad_norm: nan
2021-01-25 19:30:53,774 - mmdet - INFO - Epoch [1][950/5455]	lr: 1.250e-03, eta: 6:15:06, time: 0.362, data_time: 0.003, memory: 3858, loss_rpn_cls: 0.4698, loss_rpn_bbox: 0.0266, s0.loss_cls: nan, s0.acc: 2.0508, s0.loss_bbox: nan, s1.loss_cls: nan, s1.acc: 2.2714, s1.loss_bbox: nan, s2.loss_cls: nan, s2.acc: 1.5000, s2.loss_bbox: nan, loss: nan, grad_norm: nan
2021-01-25 19:31:10,917 - mmdet - INFO - Exp name: cascade_r50_fpn_1x.py

```
* acknowledge
[Compatibility with MMDetection 1.x](https://github.com/open-mmlab/mmdetection/blob/master/docs/compatibility.md)
[Loss becomes NAN after a few iterations](https://github.com/open-mmlab/mmdetection/issues/2739)
* check if the dataset annotations are correct
* reduce the learning rate
* extend the warmup iterations
* add gradient clipping

## Reference


* <https://github.com/Wakinguup/Underwater_detection>
* <https://blog.csdn.net/u014479551/article/details/107762513#commentBox>

