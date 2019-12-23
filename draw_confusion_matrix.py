#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-11-8 下午7:19
# @Author  : tangming
# @File    : draw_confusion_matrix.py

# -*-coding:utf-8-*-
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
import numpy as np

#labels表示你不同类别的代号，比如这里的demo中有13个类别
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
labsIndName = ['up', 'down', 'left', 'right', 'feed', 'back', 'start', 'stop','reset']

'''
具体解释一下re_label.txt和pr_label.txt这两个文件，比如你有100个样本
去做预测，这100个样本中一共有10类，那么首先这100个样本的真实label你一定
是知道的，一共有10个类别，用[0,9]表示，则re_label.txt文件中应该有100
个数字，第n个数字代表的是第n个样本的真实label（100个样本自然就有100个
数字）。
同理，pr_label.txt里面也应该有1--个数字，第n个数字代表的是第n个样本经过
你训练好的网络预测出来的预测label。
这样，re_label.txt和pr_label.txt这两个文件分别代表了你样本的真实label和预测label，然后读到y_true和y_pred这两个变量中计算后面的混淆矩阵。当然，不一定非要使用这种txt格式的文件读入的方式，只要你最后将你的真实
label和预测label分别保存到y_true和y_pred这两个变量中即可。
'''

y_true = [0]*20+[1]*20+[2]*20+[3]*20+[4]*20+[5]*20+[6]*20+[7]*20+[8]*20
y_pred = [0]*20+[1]*20+[2]*20+[5]+[3]*18+[5]+[4]*20+[5]*20+[6]*20+[7]*17+[6]*2+[0]+[8]*20
y_true = [labsIndName[i] for i in y_true]
y_pred = [labsIndName[i] for i in y_pred]
tick_marks = np.array(range(len(labels))) + 0.5
print y_true
def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)
    cb.set_ticks([4,8,12,16,20])
    cb.get_cmap()
    cb.set_cmap("Reds_r")
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labsIndName, rotation=90)
    plt.yticks(xlocations, labsIndName)
    plt.tick_params(labelsize=15)
    plt.ylabel(u'真实语音指令')
    plt.xlabel(u'预测语音指令')


cm = confusion_matrix(y_true, y_pred, labsIndName)
np.set_printoptions(precision=1)
print cm
plt.figure(figsize=(12, 8), dpi=120)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm[y_val][x_val]
    if c > 0.01:
        plt.text(x_val, y_val, "%d" % (c,), color='red', fontsize=12, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm, title=u"混淆矩阵")
# show confusion matrix
plt.savefig('./confusion_matrix.jpg', format='jpg')
plt.show()


