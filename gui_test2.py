#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-8-3 下午6:20
# @Author  : tangming
# @File    : gui_test2.py

import os
import time

from Tkinter import *
from tkinter import scrolledtext

GUI = Tk()            # 创建父容器GUI
GUI.title("Serial Tool")  # 父容器标题
GUI.geometry("460x380")  # 设置父容器窗口初始大小，如果没有这个设置，窗口会随着组件大小的变化而变化
# 第一部分
Information = LabelFrame(GUI, text="操作信息", padx=10, pady=10)  # 创建子容器，水平，垂直方向上的边距均为10
Information.place(x=20, y=20)
Information_Window = scrolledtext.ScrolledText(Information, width=20, height=5, padx=10, pady=10)
Information_Window.grid()
# 第二部分
Receive = LabelFrame(GUI, text="接收区", padx=10, pady=10 )  # 水平，垂直方向上的边距均为 10
Receive.place(x=240, y=150)
Receive_Window = scrolledtext.ScrolledText(Receive, width=20, height=12, padx=10, pady=10)
Receive_Window.grid()
# 第三部分
Send = LabelFrame(GUI, text="发送指令", padx=10, pady=5)
Send.place(x=240, y=20)

DataSend = StringVar()  # 定义DataSend为保存文本框内容的字符串
EntrySend = StringVar()

Send_Window = Entry(Send, textvariable=EntrySend, width=23)
Send_Window.grid()


def WriteData():  # 按钮按下时触发的动作函数
    global DataSend
    DataSend = EntrySend.get()  # 读取当前文本框的内容保存到字符串变量DataSend
    Information_Window.insert("end", '发送指令为：' + str(DataSend) + '\n')  # 在操作信息窗口显示发送的指令并换行，end为在窗口末尾处显示
    Information_Window.see("end")  # 此处为显示操作信息窗口进度条末尾内容，以上两行可实现窗口内容满时，进度条自动下滚并在最下方显示新的内容
    #SerialPort.write(bytes(DataSend, encoding='utf8'))  # 串口发送文本框内容


Button(Send, text="发送", command=WriteData).grid(pady=5, sticky=E)

GUI.mainloop()