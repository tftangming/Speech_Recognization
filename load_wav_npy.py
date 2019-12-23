#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-6-5 下午2:57
# @Author  : tangming
# @File    : load_wav_npy.py
import os
from os import listdir
from os.path import isfile, join
import numpy as np

if __name__ == '__main__':
    labsIndName = []
    wavs = []  # wav文件集
    labels = []  # labels里面存的值都是对应标签的下标，下标对应的名字在labsInd中
    val_wavs = []  # wav文件集
    val_labels = []  # labels里面存的值都是对应标签的下标，下标对应的名字在labsInd中

    spectrogram_path = "/home/tangming/Final_speech_recognization/aug_spec_dataset/"
    dirs = os.listdir(spectrogram_path)  # 获取各个类别的目录列表
    for sub_dir in dirs:
        print "load", sub_dir
        labsIndName.append(sub_dir) # 当前分类进入到标签的名字集
        valNum = 0
        spectrogram_dir = spectrogram_path + sub_dir
        files = os.listdir(spectrogram_dir)
        for j in files:
            waveData = np.load(spectrogram_dir+"/"+j)
            waveData = np.transpose(waveData)
            waveData = np.pad(waveData,((0, 100-waveData.shape[0]), (0, 0)), mode='constant', constant_values=0)
            if valNum < 200:
                val_wavs.append(waveData)
                val_labels.append(labsIndName.index(sub_dir))
                valNum += 1
            else:
                wavs.append(waveData)
                labels.append(labsIndName.index(sub_dir))

    num_class = len(labsIndName)
    print "total", num_class, "class"
    wavs = np.array(wavs)
    labels = np.array(labels)
    val_wavs = np.array(val_wavs)
    val_labels = np.array(val_labels)
    print wavs.shape, labels.shape, val_wavs.shape, val_labels.shape
    print np.max(wavs[0])


