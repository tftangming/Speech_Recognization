#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-8-3 下午3:01
# @Author  : tangming
# @File    : create_gui2.py

from Tkinter import *


def hh():
    root_1 = Tk()
    root_1.title("语音入库")
    root_1.geometry('250x400')
    root_1.resizable(width=False, height=True)
    # 第一部分
    Label(root_1, text='', width=20, height=1).pack()

    frm_2 = Frame(root_1, width=20, height=3)
    Label(frm_2, text='请选择训练的命令:', width=20, height=3, font=('Arial', 15)).pack(side=LEFT)
    frm_2.pack()

    # 第二部分
    frm_3 = Frame(root_1, width=20, height=6)
    frm_3_L = Frame(frm_3)
    LANGS_1 = [
        ('停止', 1),
        ('向左', 2),
        ('向右', 3),
        ('向上', 4),
        ('向下', 5)]

    # IntVar 是tkinter的一个类，可以管理单选按钮
    v_1 = IntVar()

    for lang, num in LANGS_1:
        b = Radiobutton(frm_3_L, text=lang, variable=v_1, value=num, font=('Arial', 15))
        b.pack(anchor=W)

    frm_3_L.pack(side=LEFT)

    frm_3_R = Frame(frm_3)
    LANGS_2 = [
        ('左上', 6),
        ('左下', 7),
        ('右上', 8),
        ('右下', 9),
        ('向前', 10)]
    v_2 = IntVar()

    for lang, num in LANGS_2:
        b = Radiobutton(frm_3_R, text=lang, variable=v_1, value=num, font=('Arial', 15))
        b.pack(anchor=W)

    frm_3_R.pack(side=RIGHT)

    frm_3.pack()

    # 第三部分
    Label(root_1, text='', width=20, height=1).pack()

    frm_4 = Frame(root_1, width=20, height=3)
    Button(frm_4, text='样本采集', font=('Arial', 10), width=11, height=1).pack()
    # Label(frm_4, text=' ').pack(side=LEFT)
    Button(frm_4, text='生成Mel谱图', font=('Arial', 10), width=11, height=1).pack()

    # Label(frm_4, text=' ',compound='center').pack()
    Button(frm_4, text='开始训练', font=('Arial', 10), width=11, height=1).pack()
    frm_4.pack()

    root_1.mainloop()


if __name__ == '__main__':
    hh()
