#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-8-5 下午12:54
# @Author  : tangming
# @File    : predict_ros_gui.py

# -*- coding: UTF-8 -*-

import numpy as np
import pyaudio
import struct
import wave

import keras.backend as K

K.clear_session()

import librosa.display
import librosa.feature
from keras import models
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

# import os

nrow = 200
ncol = 200

BLOCKSIZE = 128

RATE = 22050
WIDTH = 2
CHANNELS = 1
LEN = 1 * RATE

from Tkinter import *

import os
import time

model = models.load_model('cnn4no_pool_single_gru1fc0shared-31-0.9930.hdf5')


def is_silent(data, THRESHOLD):
    "Returns 'True' if below the threshold"
    return max(data) < THRESHOLD


def extract_mfcc(file, fmax, nMel):
    y, sr = librosa.load(file)

    plt.figure(figsize=(3, 3), dpi=100)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=nMel, fmax=fmax)
    librosa.display.specshow(librosa.logamplitude(S, ref_power=np.max), fmax=fmax)

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig('tmp/tmp/myImg.png', bbox_inches='tight', pad_inches=-0.1)

    plt.close()
    return


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)

    if not isExists:
        os.makedirs(path)
        print '创建目录成功'
        return True
    else:
        print '已存在目录'
        return False


def predict():
    extract_mfcc('myNumber.wav', 8000, 256)
    test_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0,
        zoom_range=0,
        horizontal_flip=False)
    test_generator = test_datagen.flow_from_directory(
        'tmp',
        target_size=(nrow, ncol),
        batch_size=1,
        class_mode='sparse')

    Xts, _ = test_generator.next()

    yts = model.predict(Xts)
    if np.max(yts) < 0.5:
        print ('Can not Recognize!')

    res = np.argmax(yts)
    print (res)

    if res == 0:
        text_1.set('向上')
        root.update()
        os.system(' gnome-terminal --geometry 20x10+1+1 -x bash -c "rosrun kinova_arm_moveit_demo up" ')
        # os.system('rosrun kinova_arm_moveit_demo up')
        return

    if res == 1:
        text_1.set('向下')
        root.update()
        os.system(' gnome-terminal --geometry 20x10+1+1 -x bash -c "rosrun kinova_arm_moveit_demo down" ')
        # os.system('rosrun kinova_arm_moveit_demo down')
        return

    if res == 2:
        text_1.set('向左')
        root.update()
        os.system(' gnome-terminal --geometry 20x10+1+1 -x bash -c "rosrun kinova_arm_moveit_demo right" ')
        # os.system('rosrun kinova_arm_moveit_demo left')
        return

    if res == 3:
        text_1.set('向右')
        root.update()
        os.system(' gnome-terminal --geometry 20x10+1+1 -x bash -c "rosrun kinova_arm_moveit_demo left" ')
        # os.system('rosrun kinova_arm_moveit_demo right')
        return

    if res == 4:
        text_1.set('左上')
        root.update()
        os.system(' gnome-terminal --geometry 20x10+1+1 -x bash -c "rosrun kinova_arm_moveit_demo xie_4" ')
        # os.system('rosrun kinova_arm_moveit_demo xie_1')
        return

    if res == 5:
        text_1.set('左下')
        root.update()
        os.system(' gnome-terminal --geometry 20x10+1+1 -x bash -c "rosrun kinova_arm_moveit_demo xie_2" ')
        # os.system('rosrun kinova_arm_moveit_demo xie_2')
        return

    if res == 6:
        text_1.set('右上')
        root.update()
        os.system(' gnome-terminal --geometry 20x10+1+1 -x bash -c "rosrun kinova_arm_moveit_demo xie_1" ')
        # os.system('rosrun kinova_arm_moveit_demo xie_3')
        return

    if res == 7:
        text_1.set('右下')
        root.update()
        os.system(' gnome-terminal --geometry 20x10+1+1 -x bash -c "rosrun kinova_arm_moveit_demo xie_3" ')
        # os.system('rosrun kinova_arm_moveit_demo xie_4')
        return

    if res == 8:
        text_1.set('前进')
        root.update()
        os.system(' gnome-terminal --geometry 20x10+1+1 -x bash -c "rosrun kinova_arm_moveit_demo stretch" ')
        # os.system('rosrun kinova_arm_moveit_demo stretch')
        return

    if res == 9:
        text_1.set('复位')
        root.update()
        os.system(' gnome-terminal --geometry 20x10+1+1 -x bash -c "rosrun kinova_arm_moveit_demo fuwei" ')
        # os.system('rosrun kinova_arm_moveit_demo fuwei')
        return


def start_simulation():
    os.system(
        ' gnome-terminal --geometry 20x10+1+1 -x bash -c "roslaunch m1n6s300_moveit_config m1n6s300_virtual_robot_demo.launch" ')
    time.sleep(10)
    os.system(' gnome-terminal --geometry 20x10+1+1 -x bash -c "rosrun kinova_arm_moveit_demo fuwei" ')

    flag = 0
    while flag < 4:
        flag += 1

        output_wf = wave.open('myNumber.wav', 'w')
        output_wf.setframerate(RATE)
        output_wf.setsampwidth(WIDTH)
        output_wf.setnchannels(CHANNELS)

        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(WIDTH),
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        output=True)

        start = False
        while True:
            input_string = stream.read(BLOCKSIZE)
            input_value = struct.unpack('h' * BLOCKSIZE, input_string)
            silent = is_silent(input_value, 1000)
            if not silent:
                start = True

            if start:
                print("Start")

                nBLOCK = int(LEN / BLOCKSIZE)
                numSilence = 0
                for n in range(0, nBLOCK):

                    if is_silent(input_value, 100):
                        numSilence += 1

                    output_value = np.array(input_value)

                    if numSilence > RATE / 8000 * 5:
                        break

                    output_value = output_value.astype(int)
                    output_value = np.clip(output_value, -2 ** 15, 2 ** 15 - 1)

                    ouput_string = struct.pack('h' * BLOCKSIZE, *output_value)
                    output_wf.writeframes(ouput_string)

                    input_string = stream.read(BLOCKSIZE)
                    input_value = struct.unpack('h' * BLOCKSIZE, input_string)

                print('Done')
                start = False

                predict()

                stream.stop_stream()
                stream.close()
                p.terminate()
                output_wf.close()
                break


def stop_simulation():
    os.system('killall -9 rviz')


root = Tk()
root.title("GUI")
root.geometry('350x400')
root.resizable(width=False, height=True)

text_1 = StringVar()
text_1.set('识别结果')
Label(root, text='', width=20, height=1).pack()
Label(root, textvariable=text_1, bg='white', font=('Arial', 20), width=20, height=5).pack()
Label(root, text='', width=20, height=1).pack()

frm = Frame(root, height=5, width=20)
Button(frm, text='语音入库', font=('Arial', 20)).pack(side=LEFT)
Button(frm, text='语音识别', font=('Arial', 20)).pack(side=RIGHT)
# Button(frm, text=' ', font=('Arial', 20),compound='center').pack()
Label(frm, text='    ', font=('Arial', 20), compound='center').pack()
frm.pack()

Label(root, text='', width=20, height=1).pack()
# frm_1=Frame(root,height = 1,width = 20)

# Label(frm_1, text='启动机械臂                ',width=20, height=1).pack(side=LEFT)
# Label(frm_1, text='        ',width=20, height=1).pack(side=LEFT)
# frm_1.pack()

frm_2 = Frame(root, height=2, width=20)
Button(frm_2, text='启动仿真', font=('Arial', 20), command=start_simulation).pack(side=LEFT)
Button(frm_2, text='结束仿真', font=('Arial', 20), command=stop_simulation).pack(side=RIGHT)
Label(frm_2, text='    ', font=('Arial', 20), compound='center').pack()
frm_2.pack()


def rtnkey(event=None):
    global name
    name = e.get()
    mkdir('myRecording/' + name)


frm_3 = Frame(root, height=1, width=20)
Label(frm_3, text='请输入实验人名字:', width=20, height=3).pack(side=LEFT)

var = StringVar()
e = Entry(frm_3, validate='key', textvariable=var)
e.bind('<Return>', rtnkey)
e.pack(side=LEFT)
Label(frm_3, text='   ', width=20, height=3).pack(side=LEFT)
frm_3.pack()

root.mainloop()


















