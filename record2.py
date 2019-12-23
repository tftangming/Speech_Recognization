#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-6-15 下午2:31
# @Author  : tangming
# @File    : record2.py

#  这只是一个基础的测试

import pyaudio
import wave

#定义一个缓冲区内存储的帧数和采样数据的大小类型
CHUNK = 1024
FORMAT = pyaudio.paInt16
#设置声道数目
CHANNELS = 1
#设置采样率,即1s中的采样点数目
RATE = 16000
#设置录音时间
RECORD_SECONDS = 6
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,#生成输入流
                frames_per_buffer=CHUNK)

print("* Start recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* Finish recording *")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(2)
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()