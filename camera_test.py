#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-11-23 下午11:10
# @Author  : tangming
# @File    : camera_test.py
import cv2
from PIL import Image, ImageTk
import numpy as np

capture2 = cv2.VideoCapture(1)
while True:
    ret2, frame2 = capture2.read()  # 从摄像头读取照片
    print frame2.shape
    # frame2 = cv2.transpose(frame2)  # (640,480,3)
    # # print frame2.shape
    # frame2 = cv2.flip(frame2, 0)
    # frame2 = cv2.flip(frame2, 1)#翻转 0:上下颠倒 大于0水平颠倒
    # cv2image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGBA)
    # img2 = Image.fromarray(cv2image2)
    # image_file2 = ImageTk.PhotoImage(img2)
    # frame2 = cv2.transpose(frame2)
    # print frame2.shape
    # frame2 = cv2.flip(frame2, 0)
    # print frame2.shape
    # frame2 = cv2.flip(frame2, 1)#翻转 0:上下颠倒 大于0水平颠倒
    #cv2.imwrite("xiangzuo.png", frame2)
    cv2.imwrite("11zhongjian.png", frame2)
    cv2.imshow("hello", frame2)
