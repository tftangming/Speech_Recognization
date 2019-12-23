#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-11-29 下午12:34
# @Author  : tangming
# @File    : draw_signal_plt.py

"""Python绘制语谱图"""
"""Python绘制时域波形"""

# 导入相应的包
import numpy, wave
import matplotlib.pyplot as plt
import numpy as np
import os

filename = "output.wav"  # 得到文件夹下的所有文件名
f = wave.open(filename, 'rb')  # 调用wave模块中的open函数，打开语音文件。
params = f.getparams()  # 得到语音参数
nchannels, sampwidth, framerate, nframes = params[
                                           :4]  # nchannels:音频通道数，sampwidth:每个音频样本的字节数，framerate:采样率，nframes:音频采样点数
strData = f.readframes(nframes)  # 读取音频，字符串格式
wavaData = np.fromstring(strData, dtype=np.int16)  # 得到的数据是字符串，将字符串转为int型
wavaData = wavaData * 1.0 / max(abs(wavaData))  # wave幅值归一化
wavaData = np.reshape(wavaData, [nframes, nchannels]).T  # .T 表示转置
f.close()
time = np.arange(0, nframes) * (1.0 / framerate)
time = np.reshape(time, [nframes, 1]).T
plt.plot(time[0, :nframes], wavaData[0, :nframes], c="b")
plt.xlabel("time(seconds)")
plt.ylabel("amplitude")
plt.title("Original wave")
plt.savefig('output.jpg'.format(filename[:-4]))  # 保存绘制的图形
plt.show()

