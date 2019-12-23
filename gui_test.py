#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-8-3 下午4:24
# @Author  : tangming
# @File    : gui_test.py

import os
import time

from Tkinter import *

root = Tk()
root.title("GUI")
root.geometry('350x400')
root.resizable(width=False, height=True)

def start_simulation():
    os.system(' gnome-terminal -x bash -c "source ~/catkin_ws/devel/setup.bash; '
              'roslaunch m1n6s300_moveit_config m1n6s300_virtual_robot_demo.launch;read" ')
  #  time.sleep(10)
 #   os.system(' gnome-terminal -x bash -c "rosrun kinova_arm_moveit_demo fuwei;read" ')

Button(root, text='启动仿真', font=('Arial', 20), command=start_simulation).pack(side=LEFT)
root.mainloop()