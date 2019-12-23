#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-8-2 下午11:52
# @Author  : tangming
# @File    : create_gui1.py

import numpy as np
import pyaudio
import struct
import wave
import os
import string
import random

BLOCKSIZE = 128
RATE = 16000
WIDTH = 2
CHANNELS = 1
LEN = 1 * RATE
RECORD_SECONDS = 1

from record4 import record, is_silent

from Tkinter import *




def luyin():
    ran_str = ''.join(random.sample(string.ascii_letters + string.digits, 8))
    path = 'weile_code_test/' + name + "/" + ran_str + ".wav"
    record(path)

def rtnkey(event=None):
    global name
    name = e.get()
    path = 'weile_code_test/' + name
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

# 创建主窗口
root = Tk()

# 设置窗口标题
root.title("GUI")
# 设置大小和位置
root.geometry('400x400')
# 设置窗口宽度不变，高度可调节
root.resizable(width=True, height=True)

# Label:标签控件,可以显示文本; width,height:如果Label显示的是文本,那么单位是文本单元
# 以下这三行代码使得这三个标签自上而下正中央排列
Label(root, text='', width=20, height=1).pack()
Label(root, text='识别结果', bg='white', font=('Arial', 20), width=20, height=3).pack()
Label(root, text='', width=20, height=1).pack()

# frame还是会根据放在frame上的控件排布而变化
frm = Frame(root, width=20, height=3)
Button(frm, text='开始录音', font=('Arial', 20), command=luyin).pack(side=LEFT)
Button(frm, text='语音识别', font=('Arial', 20)).pack(side=RIGHT)
Label(frm, text='  ', font=('Arial', 20), compound='center').pack()
frm.pack()

Label(root, text='', width=20, height=1).pack()
Label(root, text='启动机械臂', font=('Arial', 15), width=20, height=3).pack(side=LEFT)
Label(root, text='   ', width=10, height=3).pack(side=LEFT)

frm_1 = Frame(root, height=3, width=20)
# Label(root, text='', width=20, height=1).pack()
Label(root, text='请输入识别人名字:', font=('Arial', 15), width=20, height=3).pack(side=LEFT)
var = StringVar()
# validate='key':输入框被编辑的时候验证 #
e = Entry(root, validate='key', textvariable=var)
e.bind('<Return>', rtnkey)
e.pack(side=LEFT)
frm_1.pack()


root.mainloop()