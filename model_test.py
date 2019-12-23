#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-7-13 上午11:56
# @Author  : tangming
# @File    : model_test.py
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import keras
from keras.models import load_model
# load test data
from generate_wav_spectrum import *

# 加载数据集 和 标签[并返回标签集的处理结果]
def create_datasets():
    labsIndName = []
    wavs = []  # wav文件集
    labels = []  # labels里面存的值都是对应标签的下标，下标对应的名字在labsInd中

    spectrogram_path = "/home/tangming/Final_speech_recognization/aug_spec_test_data_myself/"
    dirs = os.listdir(spectrogram_path)  # 获取各个类别的目录列表
    for sub_dir in dirs:
        print "load", sub_dir
        labsIndName.append(sub_dir) # 当前分类进入到标签的名字集
        spectrogram_dir = spectrogram_path + sub_dir
        files = os.listdir(spectrogram_dir)
        for j in files:
            waveData = np.load(spectrogram_dir+"/"+j)
            waveData = np.transpose(waveData)
            waveData = np.pad(waveData,(((100-waveData.shape[0])/2, 100-waveData.shape[0]-(100-waveData.shape[0])/2),
                                        (0, 0)), mode='constant', constant_values=0)
            wavs.append(waveData)
            labels.append(labsIndName.index(sub_dir))

    num_class = len(labsIndName)
    print "total", num_class, "class"
    wavs = np.array(wavs)
    labels = np.array(labels)
    return wavs, labels, num_class, labsIndName

# 获取测试文件并计算他的频谱特征
def create_testdata(audiopath):
    labsIndName = ['rotate', 'left', 'feed', 'stop', 'down', 'right', 'start', 'reset', 'up', 'back']
    test_wavs = []  # wav文件集

    wavs = cal_spectrogram(audiopath)
    wavs = np.transpose(wavs)
    wavs= np.pad(wavs,(((100-wavs.shape[0])/2, 100-wavs.shape[0]-(100-wavs.shape[0])/2),
                                        (0, 0)), mode='constant', constant_values=0)

    test_wavs.append(wavs)
    test_wavs = np.array(test_wavs)
    return test_wavs, labsIndName


if __name__ == '__main__':
    # wavs, labels, num_class, labsIndName = create_datasets()
    # print labsIndName  # ['rotate', 'left', 'feed', 'stop', 'down', 'right', 'start', 'reset', 'up', 'back']
    # wavs = wavs / float(np.max(wavs))
    # wavs = np.expand_dims(wavs, axis=-1)
    # temp = labels
    # # 标签转换为独热码
    # labels = keras.utils.to_categorical(labels, num_class)  # num_class = 10
    audiopath = "/home/tangming/Final_speech_recognization/test/up_Nfp6uyxH.wav"
    test_wavs,labsIndName = create_testdata(audiopath)
    test_wavs = test_wavs / float(np.max(test_wavs))
    test_wavs = np.expand_dims(test_wavs, axis=-1)
    #print wavs.shape, labels.shape
    #加载模型
    model = load_model('/home/tangming/Final_speech_recognization/realdata_model/更新数据后的0.015_test/cnn4no_pool_single_gru1fc0shared-43-0.9925.hdf5')
    predict = model.predict(test_wavs)
    predict = np.argmax(predict, axis=1)
    #print predict
    label = audiopath.split("/")[-1].split("_")[0]
    print label
    print labsIndName[predict[0]] == label
    #print predict == temp

    # feed和right区分不开;



