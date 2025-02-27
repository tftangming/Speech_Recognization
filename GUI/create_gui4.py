#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-8-3 下午4:45
# @Author  : tangming
# @File    : create_gui4.py


from Tkinter import *

import os
import time


def start_simulation():
    os.system(' gnome-terminal -x bash -c "roslaunch m1n6s300_moveit_config m1n6s300_virtual_robot_demo.launch;read" ')
    time.sleep(10)
    os.system(' gnome-terminal -x bash -c "rosrun kinova_arm_moveit_demo fuwei;read" ')


def stop_simulation():
    os.system('killall -9 rviz')


root = Tk()
root.title("GUI")
root.geometry('350x400')
root.resizable(width=False, height=True)
# 第一部分
Label(root, text='', width=20, height=1).pack()
Label(root, text='识别结果', bg='white', font=('Arial', 20), width=20, height=5).pack()
Label(root, text='', width=20, height=1).pack()
# 第二部分
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
# 第三部分
frm_2 = Frame(root, height=2, width=20)
Button(frm_2, text='启动仿真', font=('Arial', 20), command=start_simulation).pack(side=LEFT)
Button(frm_2, text='结束仿真', font=('Arial', 20), command=stop_simulation).pack(side=RIGHT)
Label(frm_2, text='    ', font=('Arial', 20), compound='center').pack()
frm_2.pack()


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


frm_3 = Frame(root, height=1, width=20)
Label(frm_3, text='请输入实验人名字:', width=20, height=3).pack(side=LEFT)

var = StringVar()
e = Entry(frm_3, validate='key', textvariable=var)
e.bind('<Return>', rtnkey)
e.pack(side=LEFT)
Label(frm_3, text='   ', width=20, height=3).pack(side=LEFT)
frm_3.pack()

root.mainloop()