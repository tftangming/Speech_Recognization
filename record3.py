#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-6-15 下午2:57
# @Author  : tangming
# @File    : record3.py

import numpy as np
import pyaudio
import struct
import wave
import os
import string
import random

nrow = 200
ncol = 200

BLOCKSIZE = 128

RATE = 16000
WIDTH = 2
CHANNELS = 1
LEN = 1 * RATE

#设置录音时间
RECORD_SECONDS = 1

def is_silent(data, THRESHOLD):
    return max(data) < THRESHOLD


def record(file_path, RECORD_SECONDS=0):
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(WIDTH),
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=BLOCKSIZE)

    print("* Start recording")
    frames = []
    for i in range(0, int(RATE / BLOCKSIZE * RECORD_SECONDS)):
        data = stream.read(BLOCKSIZE)
        frames.append(data)

    print("* Finish recording *")
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(file_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(WIDTH)
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

if __name__ == '__main__':
    # 创建文件夹和音频文件
    label_name = "test"
    ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    file_path = label_name + "/" + "up_" + ran_str + '.wav'
    print file_path
    if not os.path.exists(label_name):
        os.makedirs(label_name)
    if not os.path.exists(file_path):
        os.system(r"touch {}".format(file_path))

    # 采集音频信号
    record(file_path, RECORD_SECONDS=1)

