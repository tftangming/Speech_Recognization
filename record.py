#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-6-15 上午10:10
# @Author  : tangming
# @File    : record.py

'''
以SAMPLING_RATE为采样频率，
每次读入一块有NUM_SAMPLES个采样点的数据块，
当读入的采样数据中有COUNT_NUM个值大于LEVEL的取样的时候，
将采样数据保存进WAV文件，
一旦开始保存数据，所保存的数据长度最短为SAVE_LENGTH个数据块。

从声卡读入的数据和从WAV文件读入的类似，都是二进制数据，
由于我们用paInt16格式(16bit的short类型)保存采样值，
因此将它自己转换为dtype为np.short的数组。
'''

#  这个文件并没有用到

from pyaudio import PyAudio, paInt16
import numpy as np
import wave

# 将data中的数据保存到名为filename的WAV文件中
def save_wave_file(filename, data):
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)          # 单通道
    wf.setsampwidth(2)          # 量化位数
    wf.setframerate(SAMPLING_RATE)  # 设置采样频率
    wf.writeframes(b"".join(data))  # 写入语音帧
    wf.close()


NUM_SAMPLES = 2000      # pyAudio内部缓存块的大小
SAMPLING_RATE = 16000    # 取样频率
LEVEL = 1500           # 声音保存的阈值，小于这个阈值不录
COUNT_NUM = 20 # 缓存快类如果有20个大于阈值的取样则记录声音
SAVE_LENGTH = 8 # 声音记录的最小长度：SAVE_LENGTH * NUM_SAMPLES 个取样

# 开启声音输入
pa = PyAudio()
stream = pa.open(format=paInt16, channels=1, rate=SAMPLING_RATE, input=True,
                frames_per_buffer=NUM_SAMPLES)

save_count = 0  # 用来计数
save_buffer = []  #

while True:
    # 读入NUM_SAMPLES个取样
    string_audio_data = stream.read(NUM_SAMPLES)
    # 将读入的数据转换为数组
    audio_data = np.fromstring(string_audio_data, dtype=np.short)
    # 计算大于LEVEL的取样的个数
    large_sample_count = np.sum(audio_data > LEVEL)
    print(np.max(audio_data))
    # 如果个数大于COUNT_NUM，则至少保存SAVE_LENGTH个块
    if large_sample_count > COUNT_NUM:
        save_count = SAVE_LENGTH
    else:
        save_count -= 1

    if save_count < 0:
        save_count = 0

    if save_count > 0:
        # 将要保存的数据存放到save_buffer中
        save_buffer.append(string_audio_data)
    else:
        # 将save_buffer中的数据写入WAV文件，WAV文件的文件名是保存的时刻
        if len(save_buffer) > 0:
            filename = "recorde" + ".wav"
            save_wave_file(filename, save_buffer)
            print(filename, "saved")
            break