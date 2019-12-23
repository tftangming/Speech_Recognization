#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-8-3 下午8:36
# @Author  : tangming
# @File    : record_gui_initial.py

import numpy as np
import pyaudio
import struct
import wave
import threading
import sys
import Tkinter as Tk
import random
import string
import time
import keras.backend as K
import tensorflow as tf
K.clear_session()
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from keras import models
from generate_wav_spectrum import *

graph = tf.get_default_graph()
model = models.load_model('../cnn4no_pool_single_gru1fc0shared-31-0.9930.hdf5')


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
    with graph.as_default():
        predicts = model.predict(test_wavs)
    if np.max(predicts) < 0.8:
        print 'Can not Recognize!'
        s1.set('Can not Recognize!')
        windows.update()
    else:
        #  print predict
        predicts = np.argmax(predicts, axis=1)
        print "识别结果为：", labsIndName[predicts[0]]
        s1.set("识别结果为：" + labsIndName[predicts[0]])
        windows.update()
    return


def is_silent(data, THRESHOLD):
    return max(data) < THRESHOLD


def record():
    BLOCKSIZE = 128
    RATE = 16000
    WIDTH = 2
    CHANNELS = 1
    LEN = 1 * RATE
    RECORD_SECONDS = 1

    while allowRecording:

        # 设置录音文件的声音信息
        ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
        path = "weile_code_test/" + ran_str + ".wav"
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
        predict(path)
        stream.stop_stream()
        stream.close()
        p.terminate()
        wf.close()
        time.sleep(1)


# 开始按钮
def start():
    #Tk.Label(windows, textvariable=label_var, font=('Arial', 15)).pack()
    global allowRecording
    allowRecording = True
    threading.Thread(target=record).start()

def stop():
    global allowRecording
    allowRecording = False

if __name__ == '__main__':
    windows = Tk.Tk()
    s1 = Tk.StringVar()
    L0 = Tk.Label(windows, text='Recognition', font=(None, 30))
    B1 = Tk.Button(windows, text='Start', command=start, font=(None, 20))
    L1 = Tk.Label(windows, textvariable=s1, font=(None, 25))
    B3 = Tk.Button(windows, text='Quit', command=quit, font=(None, 20))

    L0.pack()
    B1.pack()
    B3.pack()
    L1.pack(fill=Tk.X)

    windows.mainloop()
    #windows.destroy()

    #存在问题：　回调函数是一个死循环，怎么强制退出?
