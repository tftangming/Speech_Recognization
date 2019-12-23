#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-6-30 下午10:09
# @Author  : tangming
# @File    : add_noise.py

import os
import wave
import librosa
import numpy as np

def add_noise(data):
    wn = np.random.normal(0, 1, len(data))
    data_noise = np.where(data != 0.0, data.astype('float64') + 0.02 * wn, 0.0).astype(np.float32)
    return data_noise

raw_wav_path = "/home/tangming/Final_speech_recognization/dataset/"
noise_wav_path = "/home/tangming/Final_speech_recognization/noise_dataset/"
if not os.path.exists(noise_wav_path):
    os.makedirs(noise_wav_path)
dirs = os.listdir(raw_wav_path)  # 获取各个类别的目录列表
for sub_dir in dirs:
    wav_label_dir = raw_wav_path + sub_dir + "/"
    noise_wav_label_dir = noise_wav_path + sub_dir + "/"
    if not os.path.exists(noise_wav_label_dir):
        os.makedirs(noise_wav_label_dir)
    files = os.listdir(wav_label_dir)
    for i in range(len(files)):
        print i
        data, fs = librosa.core.load(wav_label_dir + files[i])
        path_noise = noise_wav_label_dir + files[i][:-4] + '-noise.wav'
        data_noise = add_noise(data)
        librosa.output.write_wav(path_noise, data_noise, fs)

print('run over！')





