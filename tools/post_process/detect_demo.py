#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : detect_demo.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/1/28 上午10:55
# @ Software   : PyCharm
#-------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
import mmcv

from mmdet.datasets import build_dataset
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmcv import Config
from mmdet.apis import set_random_seed


from PIL import Image, ImageFont, ImageDraw
import matplotlib.font_manager as fm


def draw_detect_bbox(image, bboxs, labels, scores, class_name, color):
    """

    :param image:
    :param bbox:
    :param label:
    :param color_dict:
    :return:
    """

    image = mmcv.bgr2rgb(image)
    image = Image.fromarray(image)

    img_w, img_h = image.size[0], image.size[1]

    bboxs = np.array(bboxs, dtype=np.int32).reshape(-1, 4)
    print('image shape ({},{})'.format(img_w, img_h))
    # set font
    font = ImageFont.truetype(font=fm.findfont(fm.FontProperties()),
                              size=np.floor(1.5e-2 * img_w ).astype(np.int32), encoding="unic")

    # draw box
    draw = ImageDraw.Draw(image)
    for box, label, score in zip(bboxs, labels, scores):

        cat_score = '{}|{:.4f}'.format(class_name[label], score)
        # get label size
        label_size = draw.textsize(cat_score, font)
        # get label start point
        text_origin = np.array([box[0], box[1] - label_size[1]])
        # draw bbox rectangle and label rectangle
        draw.rectangle([box[0], box[1], box[2], box[3]], outline=color[label], width=2)
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=color[label])
        draw.text(text_origin, cat_score, fill=(255, 255, 255), font=font)

    plt.style.use({'figure.figsize': (16, 12)})
    plt.imshow(image)
    plt.show()
    image.save('../../docs/demo/detect.jpg')


def inference(model, image, score_threshold=0.3, visual=False):
    """

    :param model:
    :param image:
    :param score_threshold:
    :param visual:
    :return:
    """
    class_name = model.CLASSES
    color = ('red', 'green', 'blue', 'purple')
    bbox_result = inference_detector(model, image)

    bboxes = np.vstack(bbox_result) # (, (x1, y1, x2, y2, confidence))
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)

    # filter bboxes and labels
    if score_threshold > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_threshold
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    results = []
    for bbox, label in zip(bboxes, labels):
        results.append({'bbox': bbox[:4],
                        'category': class_name[label],
                        'score': bbox[4]})

    if visual:
        # box_image = mmcv.imshow_det_bboxes(image, bboxes, labels, score_thr=score_threshold, class_names=class_name,
        #                                    show=False)
        # show_result_pyplot(model, image, bbox_result)
        # plt.figure(figsize=((16, 12)))
        # plt.imshow(mmcv.bgr2rgb(box_image))
        # plt.show(block=True)
        draw_detect_bbox(image, bboxs=bboxes[:, :4], labels=labels, scores=bboxes[:, 4], class_name=class_name,
                         color=color)

    return results


def main():
    cfg = Config.fromfile('../../configs/cascade_r50_fpn_1x.py')
    dataset = [build_dataset(cfg.data.test)]
    checkpoint = osp.join('../../outputs/htc_rcnn_r50_fpn_1x', 'latest.pth')
    model = init_detector(config=cfg, checkpoint=checkpoint, device='cuda:0')
    # Add an attribute for visualization convenience
    model.cfg = cfg
    model.CLASSES = dataset[0].CLASSES

    image = mmcv.imread(osp.join('../../docs/demo/', 'demo_1.jpg'))

    results = inference(model, image, visual=True)
    print(results)

if __name__ == "__main__":

    main()