#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : xml2coco.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/1/20 下午8:21
# @ Software   : PyCharm
#-------------------------------------------------------


import os
import os.path as osp
import xml.etree.ElementTree as ET

import mmcv

from mmdet.core import underwater_classes

from glob import glob
from tqdm import tqdm
from PIL import Image

cat_label = {name: i for i, name in enumerate(underwater_classes())}  # for mmdet_v1

def get_segmentation(points):

    return [points[0], points[1], points[2] + points[0], points[1],
             points[2] + points[0], points[3] + points[1], points[0], points[3] + points[1]]


def parse_xml(xml_path, img_id, anno_id):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotation = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name == 'waterweeds':
            continue
        category_id = cat_label[name]
        bnd_box = obj.find('bndbox')
        x_min = int(bnd_box.find('xmin').text)
        y_min = int(bnd_box.find('ymin').text)
        x_max = int(bnd_box.find('xmax').text)
        y_max = int(bnd_box.find('ymax').text)
        # w = x_max - x_min + 1  # mmdet v1
        # h = y_max - y_min + 1
        w = x_max - x_min + 1  # mmdet v2
        h = y_max - y_min + 1
        area = w * h
        segmentation = get_segmentation([x_min, y_min, w, h])
        annotation.append({
                        "segmentation": segmentation,
                        "area": area,
                        "iscrowd": 0,
                        "image_id": img_id,
                        "bbox": [x_min, y_min, w, h],
                        "category_id": category_id,
                        "id": anno_id,
                        "ignore": 0})
        anno_id += 1
    return annotation, anno_id


def cvt_annotations(img_path, xml_path, out_file):
    images = []
    annotations = []

    # xml_paths = glob(xml_path + '/*.xml')
    img_id = 1
    anno_id = 1
    for img_path in tqdm(glob(img_path + '/*.jpg')):
        w, h = Image.open(img_path).size
        img_name = osp.basename(img_path)
        img = {"file_name": img_name, "height": int(h), "width": int(w), "id": img_id}
        images.append(img)

        xml_file_name = img_name.split('.')[0] + '.xml'
        xml_file_path = osp.join(xml_path, xml_file_name)
        annos, anno_id = parse_xml(xml_file_path, img_id, anno_id)
        annotations.extend(annos)
        img_id += 1

    categories = []
    for k,v in cat_label.items():
        categories.append({"name": k, "id": v})
    final_result = {"images": images, "annotations": annotations, "categories": categories}
    mmcv.dump(final_result, out_file)
    return annotations


def main():

    data_path = '/media/alex/80CA308ECA308288/alex_dataset/URPC-2020/train'
    xml_path =  osp.join(data_path, 'box')
    img_path =  osp.join(data_path, 'image')
    json_path = osp.join(data_path, 'annotation', 'voc_all.json')

    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    print('processing {} ...'.format("xml format annotations"))
    cvt_annotations(img_path, xml_path, json_path)
    print('Done!')


if __name__ == '__main__':
    main()