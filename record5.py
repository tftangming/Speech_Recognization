#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-8-3 下午7:56
# @Author  : tangming
# @File    : record5.py

import numpy as np
import pyaudio
import struct
import wave

import keras.backend as K
K.clear_session()
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import librosa.display
import librosa.feature
from keras import models
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
model = models.load_model('cnn4no_pool_single_gru1fc0shared-31-0.9930.hdf5')

from generate_wav_spectrum import *

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

def predict(audiopath):
    test_wavs, labsIndName = create_testdata(audiopath)
    test_wavs = test_wavs / float(np.max(test_wavs))
    test_wavs = np.expand_dims(test_wavs, axis=-1)
    # print wavs.shape, labels.shape
    # 加载模型
    predict = model.predict(test_wavs)
    if np.max(predict) < 0.8:
        print 'Can not Recognize!'
    else:
        #  print predict
        predict = np.argmax(predict, axis=1)
        print "识别结果为：", labsIndName[predict[0]]
    return

def is_silent(data, THRESHOLD):
    return max(data) < THRESHOLD


def record(path):

    BLOCKSIZE = 128
    RATE = 16000
    WIDTH = 2
    CHANNELS = 1
    LEN = 1 * RATE
    RECORD_SECONDS = 1

    #设置录音文件的声音信息
    wf = wave.open(path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(WIDTH)
    wf.setframerate(RATE)

    #创建声卡采集到的数据流
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(WIDTH),
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True)

    #  需要说话声音大一点(超过2000)才能跳出下面这个循环
    while True:
        input_string = stream.read(BLOCKSIZE)
        input_value = struct.unpack('h' * BLOCKSIZE, input_string)
        silent = is_silent(input_value, 2000)
        if not silent:
            break

    print "Start recording"
    for i in range(0, int(RATE / BLOCKSIZE)):
        output_value = np.array(input_value)
        #  将数据整理压缩一下放入到指定输出文件中
        output_value = output_value.astype(int)
        #  后面看是否需要删除掉这一句
        output_value = np.clip(output_value, -2 ** 15, 2 ** 15 - 1)

        ouput_string = struct.pack('h' * BLOCKSIZE, *output_value)
        wf.writeframes(ouput_string)

        #  继续从数据流这读取一块数据
        input_string = stream.read(BLOCKSIZE)
        input_value = struct.unpack('h' * BLOCKSIZE, input_string)

    print "Finish recording"
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf.close()
if __name__ == '__main__':
    record('tm_test.wav')
    #predict('weile_code_test/test.wav')

