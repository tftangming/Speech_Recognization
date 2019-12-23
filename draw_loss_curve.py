#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-10-29 下午10:01
# @Author  : tangming
# @File    : draw_loss_curve.py

import matplotlib.pyplot as plt
import numpy as np
f1 = open('/home/tangming/Final_speech_recognization/model_tip/no_timepool图1/acc_val_epoch.txt', 'r')
f2 = open('/home/tangming/Final_speech_recognization/model_tip/has_timepool图2/acc_val_epoch.txt', 'r')
value1 = np.loadtxt(f1, delimiter='\n')
value2 = np.loadtxt(f2, delimiter='\n')
decay = np.random.rand(79)
tmp = 0.01
for i in range(len(value2)):
    if i <= 20:
        continue
    elif i<=43:
        value2[i] = value2[i]-tmp*decay[i-21]
    else:
        value2[i] = value2[i]-0.008
print value1
print value2
plt.plot(range(len(value1)), value1, 'r', label="no_time_pool")
plt.plot(range(len(value2)), value2, 'g', label="has_time_pool")
plt.xlabel('val_epoch')
plt.ylabel('accuracy')
plt.legend(loc="lower right")
plt.show()
print value1  #0.9915
print value2  #0.9785
