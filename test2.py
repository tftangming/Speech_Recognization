#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-7-23 下午4:12
# @Author  : tangming
# @File    : test2.py
# import sys
# try:
#     while True:
#         line1 = sys.stdin.readline().strip()
#         if line1 == '':
#             break
#         line2 = sys.stdin.readline().strip()
#         a = int(line1)
#         l = list(map(int, line2.split()))
#         b = [int(n) for n in line2.split()]
#         print(a)
#         print(l)
#         print(b)
# except:
#     pass
#
# import heapq
# heapq.nsmallest()

from Tkinter import *
import time
import datetime
win = Tk()

# 设置窗口标题
win.title("GUI")
# 设置大小和位置
win.geometry('400x400')
# 设置窗口宽度不变，高度可调节
win.resizable(width=True, height=True)



def say():
    new_win = Toplevel(win)
    new_win.title("监听")
    new_win.geometry("400x200")

    l1 = Label(new_win, text="empty", bg="yellow")
    l1.pack()

    while True:
        i = 0
        l1.config(text=str(i))


B1 = Button(win, text='Start', command=say, font=(None, 20))
B1.pack()
B3 = Button(win, text='Quit',command=quit, font=(None, 20))
B3.pack()

win.mainloop()