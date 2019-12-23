#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-8-3 下午3:03
# @Author  : tangming
# @File    : create_gui3.py

from Tkinter import *

root_2=Tk()
root_2.title("语音入库")
root_2.geometry('250x330')
root_2.resizable(width=False, height=True)
Label(root_2, text='', width=20, height=1).pack()

frm_5 = Frame(root_2)

frm_5_L=Frame(frm_5)
Button(frm_5_L, text='训练样本采集1', font=('Arial', 10)).pack()
Label(frm_5_L, text='').pack()
Button(frm_5_L, text='训练样本采集2', font=('Arial', 10)).pack()
Label(frm_5_L, text='').pack()
Button(frm_5_L, text='训练样本采集3', font=('Arial', 10)).pack()
Label(frm_5_L, text='').pack()
Button(frm_5_L, text='训练样本采集4', font=('Arial', 10)).pack()
Label(frm_5_L, text='').pack()
Button(frm_5_L, text='训练样本采集5', font=('Arial', 10)).pack()
frm_5_L.pack(side=LEFT)

frm_5_R=Frame(frm_5)
Button(frm_5_R, text='播放样本1', font=('Arial', 10)).pack()
Label(frm_5_R, text='').pack()
Button(frm_5_R, text='播放样本2', font=('Arial', 10)).pack()
Label(frm_5_R, text='').pack()
Button(frm_5_R, text='播放样本3', font=('Arial', 10)).pack()
Label(frm_5_R, text='').pack()
Button(frm_5_R, text='播放样本4', font=('Arial', 10)).pack()
Label(frm_5_R, text='').pack()
Button(frm_5_R, text='播放样本5', font=('Arial', 10)).pack()
frm_5_R.pack(side=RIGHT)

# 下面这条语句用来将上面两个框架分隔开
Label(frm_5, text='    ', compound='center').pack()

frm_5.pack()

Label(root_2, text='',width=20, height=1).pack()
frm_6 = Frame(root_2)
Button(frm_6, text='保存', font=('Arial', 10), width=26).pack()
frm_6.pack()


root_2.mainloop()