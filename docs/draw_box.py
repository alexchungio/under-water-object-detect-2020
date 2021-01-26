#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : draw_box.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2021/1/26 下午1:42
# @ Software   : PyCharm
#-------------------------------------------------------

import json
import cv2 as cv
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import ImageColor


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
    print('image shape ({},{})'.format(img_w, img_h))
    # set font
    font = ImageFont.truetype(font=fm.findfont(fm.FontProperties()),
                              size=np.floor(2.5e-2 * img_w ).astype(np.int32), encoding="unic")

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

    plt.style.use({'figure.figsize': (20, 10)})
    plt.imshow(image)
    plt.show()


def draw_box_with_opencv(image, bbox, label, color_dict):
    """

    :param image:
    :param bbox:
    :param label:
    :param color_dict:
    :return:
    """

    img_w = image.size[0]
    img_h = image.size[1]
    image = np.array(image)

    bbox = np.array(bbox, dtype=np.int32).reshape(-1, 4)
    print('image shape ({},{})'.format(img_w, img_h))
    # set font
    font = cv.FONT_HERSHEY_SIMPLEX

    for box, tag in zip(bbox, label):
        # get label size
        label_size = cv.getTextSize(tag, font, 1.6, 2)
        # get label start point
        text_origin = np.array([box[0], box[1] - label_size[0][1]])

        cv.rectangle(image, (box[0], box[1]), (box[2], box[3]),
                      color=color_dict[tag], thickness=2)
        cv.rectangle(image, tuple(text_origin), tuple(text_origin + label_size[0]),
                      color=color_dict[tag], thickness=-1)  # thickness=-1 represent set fill format

        cv.putText(image, tag, (box[0], box[1]), font, 1.5, (255, 255, 255), 2)

    plt.style.use({'figure.figsize': (20, 10)})
    plt.imshow(image)
    plt.show()


def main():
    img_path = './demo/demo.jpg'


    pil_color_dict = {'echinus': 'red', 'starfish': 'green', 'holothurian': 'blue', 'scallop': 'yellow'}
    opencv_color_dict = {'echinus': ImageColor.getcolor('red', 'RGB')[::-1],
                         'starfish': ImageColor.getcolor('green', 'RGB')[::-1],
                         'holothurian': ImageColor.getcolor('blue', 'RGB')[::-1],
                         'scallop': ImageColor.getcolor('yellow', 'RGB')[::-1]}
    # bbox = [[890, 593, 1463, 990]]
    bbox = [[890, 593, 1463, 990]]
    label = ['starfish']
    img = Image.open(img_path)
    draw_box_with_pil(img, bbox, label, pil_color_dict)
    draw_box_with_opencv(img, bbox, label, opencv_color_dict)


if __name__ == "__main__":
    main()

