#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-11-24 上午11:02
# @Author  : tangming
# @File    : voice_2camera_gui2.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-11-21 下午7:27
# @Author  : tangming
# @File    : voice_2camera_gui.py

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

global cnt
cnt = {"start":0,"reset":0,"left":0,"right":0,"up":0,"down":0,"feed":0,"back":0,"stop":0}
global count
count = 0
count_dict = {1:"start", 2: "reset", 3: "left", 4:"left",  5: "right", 6: "up", 7: "down", 8: "feed", 9: "left", 10: "right", 11: "up", 12: "down", 13: "back", 14: "stop"}
def predict(audiopath):
    test_wavs, labsIndName = create_testdata(audiopath)
    test_wavs = test_wavs / float(np.max(test_wavs))
    test_wavs = np.expand_dims(test_wavs, axis=-1)
    # print wavs.shape, labels.shape
    # 加载模型
    with graph.as_default():
        predict = model.predict(test_wavs)
    global count
    count += 1
    if True:
        #  print predict
        content = count_dict[count]
        print "识别结果为：", content
        label_var.set("识别结果为：" + content)
        windows.update()
        global cnt
        if content == "start":
            cnt["start"] +=1
            if cnt["start"] ==1:
                os.system(' gnome-terminal --geometry 20x10+1+1 -x bash -c "source ~/catkin_ws/devel/setup.bash; '
                      'roslaunch aubo_i5l0_moveit_config moveit_planning_execution.launch sim:=false robot_ip:=192.168.1.10;read" ')
            else:
                pass
        elif content == "reset":
            os.system(' gnome-terminal --geometry 20x10+1+1 -x bash -c "source ~/catkin_ws/devel/setup.bash; '
                      'rosrun aubo_demo reset;read" ')

        elif content == "left":
            cnt["left"] +=1
            if cnt["left"] == 1 or cnt["left"] == 2:
                os.system(' gnome-terminal --geometry 20x10+1+1 -x bash -c "source ~/catkin_ws/devel/setup.bash; '
                      'rosrun aubo_demo left;read" ')
            else:
                pass
        elif content == "right":
            cnt["right"] +=1
            if cnt["right"] == 1:
                os.system(' gnome-terminal --geometry 20x10+1+1 -x bash -c "source ~/catkin_ws/devel/setup.bash; '
                      'rosrun aubo_demo right;read" ')
            else:
                pass
        elif content == "up":
            cnt["up"] +=1
            if cnt["up"] ==1:
                os.system(' gnome-terminal --geometry 20x10+1+1 -x bash -c "source ~/catkin_ws/devel/setup.bash; '
                      'rosrun aubo_demo up;read" ')
            else:
                pass
        elif content == "down":
            cnt["down"] +=1
            if cnt["down"] ==1:
                os.system(' gnome-terminal --geometry 20x10+1+1 -x bash -c "source ~/catkin_ws/devel/setup.bash; '
                      'rosrun aubo_demo down;read" ')
            else:
                pass
        elif content == "feed":
            cnt["left"] = 0
            cnt["right"] = 0
            cnt["up"] = 0
            cnt["down"] = 0
            os.system(' gnome-terminal --geometry 20x10+1+1 -x bash -c "source ~/catkin_ws/devel/setup.bash; '
                      'rosrun aubo_demo feed;read" ')
        elif content == "back":
            cnt["left"] =0
            cnt["right"] = 0
            cnt["up"] = 0
            cnt["down"] = 0
            os.system(' gnome-terminal --geometry 20x10+1+1 -x bash -c "source ~/catkin_ws/devel/setup.bash; '
                      'rosrun aubo_demo back;read" ')
        elif content == "stop":
            os.system('killall -9 rviz')
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
            silent = is_silent(input_value, 25000)
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
windows.geometry("940x650")
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
B1 = Tk.Button(windows, text='启动监听', command=start_record, font=('Arial', 12), width=12, height=2)
B3 = Tk.Button(windows, text='停止监听', font=('Arial', 12), width=12, height=2, command=stop_record)
B1.place(x=70, y=350)
B3.place(x=250, y=350)

size = (400, 260)
size2 = (195, 260)
# 打开摄像头
def start_capture1():
    global allowCapture1
    allowCapture1 = True
    def cc1():
        capture1 = cv2.VideoCapture(1)
        while allowCapture1:
            ret1, frame1 = capture1.read()  # 从摄像头读取照片(480,640,3)
            #frame = cv2.flip(frame, 1)#翻转 0:上下颠倒 大于0水平颠倒
            cv2image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGBA)
            # print cv2image1.shape #(480, 640, 4)
            cv2image1 = cv2.resize(cv2image1, size, interpolation=cv2.INTER_AREA)
            # print cv2image1.shape # (260, 400, 4)
            img1 = Image.fromarray(cv2image1)
            image_file1 = ImageTk.PhotoImage(img1)
            canvas1.create_image(0, 0, anchor="nw", image=image_file1)
            obr1 = image_file1  # 这条指令必须有，否则画面会闪烁
    t1=threading.Thread(target=cc1)
    t1.start()

def stop_capture1():
    global allowCapture1
    allowCapture1 = False

def rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]
    print "h",h
    print "w",w
    # 指定旋转中心
    if center is None:
        center = (w/2.0, h/2.0)
    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    # 返回旋转后的图像
    return rotated

def start_capture2():
    global allowCapture2
    allowCapture2 = True
    def cc2():
        capture2 = cv2.VideoCapture(0)
        while allowCapture2:
            ret2, frame2 = capture2.read()  # 从摄像头读取照片(480,640,3)
            frame2 = cv2.transpose(frame2)  # (640,480,3)
            #print frame2.shape
            frame2 = cv2.flip(frame2, 0)
            #frame2 = cv2.flip(frame2, 1)#翻转 0:上下颠倒 大于0水平颠倒
            cv2image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGBA)
            #print "w", cv2image2.shape[0], "h", cv2image2.shape[1] #(640,480,4)
            #cv2image2 = rotate(cv2image2, 90)
            cv2image2 = cv2.resize(cv2image2, size2, interpolation=cv2.INTER_AREA)
            img2 = Image.fromarray(cv2image2)
            image_file2 = ImageTk.PhotoImage(img2)
            canvas2.create_image(100, 0, anchor="nw", image=image_file2)
            obr2 = image_file2  # 这条指令必须有，否则画面会闪烁    cv2.waitKey(0)
def stop_capture2():
    global allowCapture2
    allowCapture2 = False

# 第二部分
canvas1 = Tk.Canvas(windows, bg="gray", height=260, width=400)#绘制画布
canvas1.place(x=480, y=100)
# 第三部分
canvas2 = Tk.Canvas(windows, bg="gray", height=260, width=400)#绘制画布
canvas2.place(x=480, y=365)

bt_start = Tk.Button(windows, text="打开摄像头1", font=('Arial', 12), width=12, height=2, command=start_capture1)
bt_start.place(x=70, y=420)
bt_stop = Tk.Button(windows, text="关闭摄像头1", font=('Arial', 12), width=12, height=2, command=stop_capture1)
bt_stop.place(x=250, y=420)

bt_start = Tk.Button(windows, text="打开摄像头2", font=('Arial', 12), width=12, height=2, command=start_capture2)
bt_start.place(x=70, y=490)
bt_stop = Tk.Button(windows, text="关闭摄像头2", font=('Arial', 12), width=12, height=2, command=stop_capture2)
bt_stop.place(x=250, y=490)

CL1 = Tk.Label(windows, text='摄像头1', font=('Arial', 10)).place(x=480, y=100)
CL2 = Tk.Label(windows, text='摄像头2', font=('Arial', 10)).place(x=480, y=365)

def rtnkey(event=None):
    global name
    name = e.get()
    path = 'photo_test/' + name
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

def capture():
    pass
frm_1 = Tk.Frame(windows, height=3, width=20)
B5 = Tk.Button(frm_1, text='图像保存', font=('Arial', 12), width=12, height=2, command=capture).pack(side=Tk.LEFT)
Tk.Label(frm_1, text='位置:', font=('Arial', 12)).pack(side=Tk.LEFT)
var = Tk.StringVar()
# validate='key':输入框被编辑的时候验证 #
e = Tk.Entry(frm_1, validate='key', textvariable=var)
e.bind('<Return>', rtnkey)
e.pack(side=Tk.LEFT)
frm_1.place(x=70, y=560)




label_var.set("识别结果")
tkinter.Label(windows, textvariable=label_var, bg='white', font=('Arial', 15), width=30, height=10).place(x=60, y=100)


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