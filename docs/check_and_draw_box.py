#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : check_and_draw_box.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/1/26 下午2:43
# @ Software   : PyCharm
#-------------------------------------------------------
import os
import json
import os.path as osp
import numpy as np
from PIL import Image, ImageFont, ImageDraw

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tqdm import tqdm


def draw_box_with_pil(image, bbox, label, color_dict):
    """

    :param image:
    :param bbox:
    :param label:
    :param color_dict:
    :return:
    """

    img_w = image.size[0]
    img_h = image.size[1]

    bbox = np.array(bbox, dtype=np.int32).reshape(-1, 4)
    # print('image shape ({},{})'.format(img_w, img_h))
    # set font
    font = ImageFont.truetype(font=fm.findfont(fm.FontProperties()),
                              size=np.floor(1.5e-2 * img_w ).astype(np.int32), encoding="unic")

    # draw box
    draw = ImageDraw.Draw(image)
    for box, tag in zip(bbox, label):
        # get label size
        label_size = draw.textsize(tag, font)
        # get label start point
        text_origin = np.array([box[0], box[1] - label_size[1]])
        # draw bbox rectangle and label rectangle
        draw.rectangle([box[0], box[1], box[2], box[3]], outline=color_dict[tag], width=2)
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=color_dict[tag])
        draw.text(text_origin, str(tag), fill=(255, 255, 255), font=font)

    return image


def check_bbox_boundary(images_info, annotations_info, img_dir, box_img_dir, label_tag, color_dict):
    """

    :return:
    """

    for img in tqdm(images_info):
        img_name = img['file_name']
        img_id = img['id']
        img_w, img_h = img['width'], img['height']
        # get image bbox
        bboxs = []
        labels = []
        for anns in annotations_info:
            if anns['image_id'] == img['id']:
                x1, y1, w, h = anns['bbox']

                w, h = w -1, h - 1
                if anns['area'] < 0 or w < 0 or h < 0:
                    print(anns['area'], w, h)
                    continue
                # x1, y1, x2, y2 = x1, y1, x1 + w, y1 + h
                # restrict bbox to image area
                x1 = max(x1, 0)
                y1 = max(y1, 0)
                x2 = min(x1 + w, img_w)
                y2 = min(y1 + h, img_h)
                bboxs.append([x1, y1, x2, y2])
                labels.append(anns['category_id'])

        bboxs = np.array(bboxs, dtype=np.int32).reshape(-1, 4)
        # assert (bboxs[:, 2] >= 1).all(), "Warning, {}  bbox tag error in width aspect {}".format(img_name, bboxs)
        # assert (bboxs[:, 3] >= 1).all(), "Warning, {}  bbox tag error in height aspect {}".format(img_name, bboxs)

        # bboxs[:, 2:] = bboxs[:,:2] + bboxs[:, 2:]
        assert (bboxs[:, 0] >= 0).all() and (bboxs[:, 2] <= img_w).all(), \
            "Warning, {} bbox size out of range in width aspect {} {}".format(img_name, bboxs, img_w)
        assert (bboxs[:, 1] >= 0).all() and ( bboxs[:, 3] <= img_h).all(), \
            "Warning, {} bbox size out of range in height aspect {} {}".format(img_name, bboxs, img_h)

        # draw box on image
        label = [label_tag[label] for label in labels]

        image = Image.open(osp.join(img_dir, img_name))
        box_img = draw_box_with_pil(image, bboxs, label, color_dict)

        box_img.save(osp.join(box_img_dir, img_name))


def main():

    json_path = '/media/alex/80CA308ECA308288/alex_dataset/URPC-2020/train/annotation/voc_all.json'
    img_dir = '/media/alex/80CA308ECA308288/alex_dataset/URPC-2020/train/image'
    box_img_dir = '/media/alex/80CA308ECA308288/alex_dataset/URPC-2020/train/box_image'

    # load annotation
    with open(json_path) as f:
        all_data = json.load(f)
    images_info = all_data['images']
    annotations_info = []
    for ann in all_data['annotations']:
        ann.pop('id')  # remove annotation id
        ann.pop('iscrowd')
        annotations_info.append(ann)
    category_dict = {x['name']: x['id'] for x in all_data['categories']}

    label_tag = {id:name for name, id in category_dict.items()}
    color_dict = {'echinus': 'red', 'starfish': 'green', 'holothurian': 'blue', 'scallop': 'purple'}
    os.makedirs(box_img_dir, exist_ok=True)
    check_bbox_boundary(images_info, annotations_info, img_dir=img_dir, box_img_dir=box_img_dir, label_tag=label_tag,
                        color_dict=color_dict)
    print('Done')


if __name__ == "__main__":
    main()
