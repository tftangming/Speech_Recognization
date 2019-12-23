#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-6-30 上午11:17
# @Author  : tangming
# @File    : speech_data_aug.py

import os

raw_wav_path = "/home/tangming/Final_speech_recognization/test_data_myself/"
aug_wav_path = "/home/tangming/Final_speech_recognization/aug_test_data_myself/"
if not os.path.exists(aug_wav_path):
    os.makedirs(aug_wav_path)
dirs = os.listdir(raw_wav_path)  # 获取各个类别的目录列表
for sub_dir in dirs:
    #if sub_dir == "down" or sub_dir == "up" or sub_dir == "left" or sub_dir == "right" or sub_dir == "stop":
    #    continue
    print sub_dir
    wav_label_dir = raw_wav_path + sub_dir + "/"
    aug_wav_label_dir = aug_wav_path + sub_dir + "/"
    if not os.path.exists(aug_wav_label_dir):
        os.makedirs(aug_wav_label_dir)

    files = os.listdir(wav_label_dir)
    for j in files:
        file_path = wav_label_dir + j
        output_path = aug_wav_label_dir + j[:-4] + "_aug"
        output_filepath2 = output_path + "tempo1.2" + ".wav"
        output_filepath3 = output_path + "tempo0.8" + ".wav"
        os.system("sox -v 0.90 " + file_path + " " + output_filepath2 + " tempo 1.2")
        os.system("sox -v 0.90 " + file_path + " " + output_filepath3 + " tempo 0.8")
        for k in ["-200", "-100", "100", "200"]:
            output_filepath1 = output_path + k + ".wav"
            #if not os.path.exists(output_filepath1):
            #    os.mknod(output_filepath1)
            #if not os.path.exists(output_filepath2):
            #    os.mknod(output_filepath2)
            os.system("sox -v 0.88 " + file_path + " " + output_filepath1 + " pitch " + k)
            # sox -v 0.90 fcx_QRov3Oca.wav fcx_change2.wav tempo 1.2
