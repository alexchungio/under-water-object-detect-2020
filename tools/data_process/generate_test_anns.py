#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : generate_test_anns.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/1/20 下午8:57
# @ Software   : PyCharm
#-------------------------------------------------------

import json
import os
import os.path as osp
from glob import glob
from tqdm import tqdm
from PIL import Image
from mmdet.core import underwater_classes

# cat_label = {name: i + 1  for i, name in enumerate(underwater_classes())}  # for mmdet_v1
cat_label = {name: i for i, name in enumerate(underwater_classes())}  # for mmdet_v2


def save(images, annotations, path):
    ann = {}
    ann['type'] = 'instances'
    ann['images'] = images
    ann['annotations'] = annotations

    categories = []
    for k, v in cat_label.items():
        categories.append({"name": k, "id": v})
    ann['categories'] = categories
    json.dump(ann, open(path, 'w'))


def test_dataset(im_dir, path):
    im_list = glob(os.path.join(im_dir, '*.jpg'))
    idx = 1
    image_id = 1
    images = []
    annotations = []
    for im_path in tqdm(im_list):
        image_id += 1
        im = Image.open(im_path)
        w, h = im.size
        image = {'file_name': os.path.basename(im_path), 'width': w, 'height': h, 'id': image_id}
        images.append(image)
        labels = [[1, 2, 3, 4]]
        for label in labels:
            bbox = [label[0], label[1], label[2] - label[0], label[3] - label[1]]
            seg = []
            ann = {'segmentation': [seg], 'area': bbox[2] * bbox[3], 'iscrowd': 0, 'image_id': image_id,
                   'bbox': bbox, 'category_id': 1, 'id': idx, 'ignore': 0}
            idx += 1
            annotations.append(ann)
    save(images, annotations, path)


if __name__ == '__main__':
    root_path = '/media/alex/80CA308ECA308288/alex_dataset/URPC-2020/'
    test_dir = osp.join(root_path, 'test-A-image')
    test_anns_path = osp.join(root_path, 'train', 'annotation', 'testA.json')
    print("generate test json label file.")
    test_dataset(test_dir, test_anns_path)