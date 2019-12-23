#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-8-5 上午11:09
# @Author  : tangming
# @File    : record_gui_final.py


import numpy as np
import pyaudio
import struct
import wave
import sys
import Tkinter as Tk
import tkinter
import threading
import random
import string
import time
import os
from keras import models
from generate_wav_spectrum import *
from tkinter import messagebox, filedialog
import tensorflow as tf
BLOCKSIZE = 128
RATE = 16000
WIDTH = 2
CHANNELS = 1
LEN = 1 * RATE
ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
path = "weile_code_test/" + ran_str + ".wav"
graph = tf.get_default_graph()
model = models.load_model('../cnn4no_pool_single_gru1fc0shared-31-0.9930.hdf5')
labsIndName = ['rotate', 'left', 'feed', 'stop', 'down', 'right', 'start', 'reset', 'up', 'back']
allowRecording = False


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
        predict = model.predict(test_wavs)
    if np.max(predict) < 0.2:
        print 'Can not Recognize!'
        label_var.set('无法识别')
    else:
        #  print predict
        predict = np.argmax(predict, axis=1)
        if predict[0]==0:
            print "识别结果为：left"
            label_var.set("识别结果为：left")
        else:
            print "识别结果为：", labsIndName[predict[0]]
            label_var.set("识别结果为：" + labsIndName[predict[0]])
    return


def is_silent(data, THRESHOLD):
    return max(data) < THRESHOLD


def record():

    #while allowRecording:
    wf = wave.open(path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(WIDTH)
    wf.setframerate(RATE)

    # 创建声卡采集到的数据流
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(WIDTH),
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True)
    while allowRecording:
    #  需要说话声音大一点(超过2000)才能跳出下面这个循环
        while allowRecording:
            input_string = stream.read(BLOCKSIZE)
            input_value = struct.unpack('h' * BLOCKSIZE, input_string)
            silent = is_silent(input_value, 20000)
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
        wf.close()
        os.remove(path)
        wf = wave.open(path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(WIDTH)
        wf.setframerate(RATE)
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf.close()


windows = Tk.Tk()
windows.title("语音识别")
windows.geometry("400x300")
windows.resizable(False, False)

label_var = Tk.StringVar()
# 开始按钮
def start():
    global allowRecording
    allowRecording = True
    threading.Thread(target=record).start()


# 结束按钮
def stop():
    global allowRecording
    allowRecording = False



L = Tk.Label(windows, text=' ', font=('Arial', 10)).pack()
L0 = Tk.Label(windows, text='机械臂指令语音识别系统', font=('Arial', 22)).pack()
B1 = Tk.Button(windows, text='启动监听', command=start, font=('Arial', 15))
B3 = Tk.Button(windows, text='停止监听', font=('Arial', 15), command=stop)
B1.place(x=25,y=250)
B3.place(x=270,y=250)

label_var.set("识别结果")
tkinter.Label(windows, textvariable=label_var, bg='white', font=('Arial', 15), width=20, height=4).place(x=85, y=100)


# 关闭程序时检查是否正在录制
def closeWindow():
    if allowRecording:
        tkinter.messagebox.showerror("正在监听", "请先停止监听")
        return
    windows.destroy()


# 用于定义用户使用窗口管理器明确关闭窗口时发生的情况
windows.protocol("WM_DELETE_WINDOW", closeWindow)
windows.mainloop()