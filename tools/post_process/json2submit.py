#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : json2submit.py
# @ Description:
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/1/27 下午3:43
# @ Software   : PyCharm
#-------------------------------------------------------

import json
import os
import argparse
from mmcv import Config

cfg = Config.fromfile(os.path.join('configs', 'cascade_r50_fpn_1x.py'))

underwater_classes = ['holothurian', 'echinus', 'scallop', 'starfish']
def parse_args():
    parser = argparse.ArgumentParser(description='json2submit_nms')
    parser.add_argument('--test-json', help='test result json', type=str)
    parser.add_argument('--submit-file', help='submit_file_name', type=str)
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    test_json_raw = json.load(open(cfg.data.test.ann_file))
    test_json = json.load(open(args.test_json, "r"))

    os.makedirs(os.path.dirname(args.submit_file), exist_ok=True)
    img = test_json_raw['images']
    images = []
    csv_file = open(args.submit_file, 'w')
    csv_file.write("name,image_id,confidence,xmin,ymin,xmax,ymax\n")
    imgid2anno = {}
    imgid2name = {}
    for imageinfo in test_json_raw['images']:
        imgid = imageinfo['id']
        imgid2name[imgid] = imageinfo['file_name']
    for anno in test_json:
        img_id = anno['image_id']
        if img_id not in imgid2anno:
            imgid2anno[img_id] = []
        imgid2anno[img_id].append(anno)
    for imgid, annos in imgid2anno.items():
        for anno in annos:
            xmin, ymin, w, h = anno['bbox']
            xmax = xmin + w
            ymax = ymin + h
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            confidence = anno['score']
            class_id = int(anno['category_id'])
            class_name = underwater_classes[class_id-1]
            image_name = imgid2name[imgid]
            image_id = image_name.split('.')[0] + '.xml'
            csv_file.write(class_name + ',' + image_id + ',' + str(confidence) + ',' + str(xmin) + ',' + str(ymin) + ',' + str(xmax) + ',' + str(ymax) + '\n')
    csv_file.close()
    print('save summit file to {}'.format(args.submit_file))
