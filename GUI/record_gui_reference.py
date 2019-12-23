#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-8-5 上午12:04
# @Author  : tangming
# @File    : record_gui_reference.py

# 一个实现录音机功能的界面设计（利用了线程）

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

BLOCKSIZE = 128
RATE = 16000
WIDTH = 2
CHANNELS = 1
LEN = 1 * RATE
ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
path = "weile_code_test/" + ran_str + ".wav"
allowRecording = False

def record():
    #tkinter.Label(root, text="lalalala").pack()
    global path
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(WIDTH),
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    output=True)
    wf = wave.open(path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(WIDTH)
    wf.setframerate(RATE)
    while allowRecording:
        data = stream.read(16000)
        wf.writeframes(data)
        print allowRecording
    wf.close()
    stream.stop_stream()
    stream.close()
    p.terminate()


root = Tk.Tk()
root.title("录音机")
root.geometry("280x80+400+300")
root.resizable(False, False)


# 开始按钮
def start():
    global allowRecording
    allowRecording = True
    lbStatus["text"] = "正在录音"
    threading.Thread(target=record).start()


btnStart = tkinter.Button(root, text="开始录音", command=start)
btnStart.place(x=30, y=20, width=100, height=20)


# 结束按钮
def stop():
    global allowRecording
    allowRecording = False
    lbStatus["text"] = "准备就绪"


btnStop = tkinter.Button(root, text="停止录音", command=stop)
btnStop.place(x=140, y=20, width=100, height=20)

lbStatus = tkinter.Label(root,text="准备就绪", anchor="w", fg="green")
lbStatus.place(x=30, y=50, width=200, height=20)


# 关闭程序时检查是否正在录制
def closeWindow():
    if allowRecording:
        tkinter.messagebox.showerror("正在录制", "请先停止录制")
        return
    root.destroy()

# 用于定义用户使用窗口管理器明确关闭窗口时发生的情况
root.protocol("WM_DELETE_WINDOW", closeWindow)
root.mainloop()



