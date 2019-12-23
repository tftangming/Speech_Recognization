#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-8-5 下午4:01
# @Author  : tangming
# @File    : voice_camera_gui.py


import cv2
from PIL import Image, ImageTk
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
path = "/home/tangming/Final_speech_recognization/weile_code_test/" + ran_str + ".wav"
graph = tf.get_default_graph()
model = models.load_model('/home/tangming/Final_speech_recognization/cnn4no_pool_single_gru1fc0shared-31-0.9930.hdf5')
labsIndName = ['rotate', 'left', 'feed', 'stop', 'down', 'right', 'start', 'reset', 'up', 'back']
allowRecording = False
allowCapture = False


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
    if np.max(predict) < 0.8:
        print 'Can not Recognize!'
        label_var.set('Can not Recognize!')
    else:
        #  print predict
        predict = np.argmax(predict, axis=1)
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
windows.title("我的窗口")
windows.geometry("1000x650")
windows.resizable(False, False)

label_var = Tk.StringVar()


# 开始按钮
def start_record():
    global allowRecording
    allowRecording = True
    threading.Thread(target=record).start()


# 结束按钮
def stop_record():
    global allowRecording
    allowRecording = False


# 第一部分
L = Tk.Label(windows, text=' ', font=('Arial', 10)).pack()
L0 = Tk.Label(windows, text='机械臂指令语音控制系统', font=('Arial', 25)).pack()
B1 = Tk.Button(windows, text='启动监听', command=start_record, font=('Arial', 15))
B3 = Tk.Button(windows, text='停止监听', font=('Arial', 15), command=stop_record)
B1.place(x=80, y=600)
B3.place(x=260, y=600)


# 打开摄像头
def start_capture():
    global allowCapture
    allowCapture = True
    def cc():
        capture = cv2.VideoCapture(0)
        while allowCapture:
            ret, frame = capture.read()#从摄像头读取照片
            #frame = cv2.flip(frame, 1)#翻转 0:上下颠倒 大于0水平颠倒
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            image_file = ImageTk.PhotoImage(img)
            canvas.create_image(0, 0, anchor="nw", image=image_file)
            obr = image_file  # 这条指令必须有，否则画面会闪烁
    t=threading.Thread(target=cc)
    t.start()

def stop_capture():
    global allowCapture
    allowCapture = False

# 第二部分
canvas = Tk.Canvas(windows, bg="gray", height=400, width=500)#绘制画布
canvas.place(x=480, y=150)

bt_start = Tk.Button(windows, text="打开摄像头", font=('Arial', 15), command=start_capture)
bt_start.place(x=580, y=600)
bt_stop = Tk.Button(windows, text="关闭摄像头", font=('Arial', 15), command=stop_capture)
bt_stop.place(x=760, y=600)



label_var.set("识别结果")
tkinter.Label(windows, textvariable=label_var, bg='white', font=('Arial', 15), width=30, height=8).place(x=60, y=270)

# 关闭程序时检查是否正在录制
def closeWindow():
    if allowRecording:
        tkinter.messagebox.showerror("正在录音", "请先停止录音")
        return
    if allowCapture:
        tkinter.messagebox.showerror("正在录像", "请先停止录像")
        return
    windows.destroy()


# 用于定义用户使用窗口管理器明确关闭窗口时发生的情况
windows.protocol("WM_DELETE_WINDOW", closeWindow)
windows.mainloop()