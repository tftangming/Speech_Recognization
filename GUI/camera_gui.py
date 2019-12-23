#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-8-5 下午3:27
# @Author  : tangming
# @File    : camera_gui.py

import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk#图像控件
import threading

# 创建窗口
window = tk.Tk()
window.title("摄像头")
sw = window.winfo_screenwidth()#获取屏幕宽
sh = window.winfo_screenheight()#获取屏幕高
wx = 600
wh = 700
window.geometry("%dx%d+%d+%d" % (wx, wh, (sw-wx)/2, (sh-wh)/2-100))#窗口至指定位置
canvas = tk.Canvas(window, bg="gray", height=wh, width=wx)#绘制画布
canvas.pack()

# 打开摄像头获取图片
def video_demo():
    global allowCapture
    allowCapture = True
    def cc():
        capture = cv2.VideoCapture(0)
        while allowCapture:
            ret, frame = capture.read()#从摄像头读取照片
            frame = cv2.flip(frame, 1)#翻转 0:上下颠倒 大于0水平颠倒
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            image_file = ImageTk.PhotoImage(img)
            canvas.create_image(0, 0, anchor="nw", image=image_file)
            obr = image_file  # 这条指令必须有，否则画面会闪烁
    t=threading.Thread(target=cc)
    t.start()

def stop():
    global allowCapture
    allowCapture = False

bt_start = tk.Button(window, text="打开摄像头", height=2, width=15, command=video_demo)
bt_start.place(x=130, y=600)
bt_stop = tk.Button(window, text="关闭摄像头", height=2, width=15, command=stop)
bt_stop.place(x=330, y=600)
window.mainloop()