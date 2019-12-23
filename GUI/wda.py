#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-8-4 下午8:23
# @Author  : tangming
# @File    : wda.py

# 引入模块
import tkinter as tk
import time


# 设计一个时钟类
class clock(tk.Frame):
    wait_time = 1000

    def __init__(self, parent=None, **kw):
        tk.Frame.__init__(self, parent, kw)
        self.time_str = tk.StringVar()
        self.time_str.set("empty")
        tk.Label(self, textvariable=self.time_str).pack()

    def _update(self):
        self.time_str.set(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        self.timer = self.after(self.wait_time, self._update)
    def start(self):
        self._update()
        print "haha"
        self.pack(side=tk.BOTTOM)


def main():
    root = tk.Tk()
    root.title("测试用例")
    root.geometry("400x400")
    mw = clock(root)
    mywatch = tk.Button(root, text='时钟', command=mw.start)
    b2 = tk.Button(root,text="退出", command=quit)
    mywatch.place(x=175, y=200)
    b2.pack(side=tk.BOTTOM)
    root.mainloop()


main()