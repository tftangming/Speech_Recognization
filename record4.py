#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-8-2 下午3:54
# @Author  : tangming
# @File    : record4.py

import numpy as np
import pyaudio
import struct
import wave




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

    #  需要说话声音大一点(超过500)才能跳出下面这个循环
    while True:
        input_string = stream.read(BLOCKSIZE)
        input_value = struct.unpack('h' * BLOCKSIZE, input_string)
        silent = is_silent(input_value, 2000)
        print "haha"
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
    record('test.wav')
