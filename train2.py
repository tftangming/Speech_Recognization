#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-6-5 下午9:46
# @Author  : tangming
# @File    : train2.py

import os
from os import listdir
from os.path import isfile, join
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Conv2D, BatchNormalization, MaxPooling2D, Concatenate, Flatten
from keras.layers import GRU, LSTM
from keras.losses import categorical_crossentropy
from keras import regularizers
from random import shuffle
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping

labsIndName = []
#加载数据集 和 标签[并返回标签集的处理结果]
def create_datasets():
    wavs = []  # wav文件集
    labels = []  # labels里面存的值都是对应标签的下标，下标对应的名字在labsInd中
    val_wavs = []  # wav文件集
    val_labels = []  # labels里面存的值都是对应标签的下标，下标对应的名字在labsInd中

    spectrogram_path = "/home/tangming/Final_speech_recognization/aug_spec_dataset/"
    dirs = os.listdir(spectrogram_path)  # 获取各个类别的目录列表
    for sub_dir in dirs:
        print "load", sub_dir
        labsIndName.append(sub_dir) # 当前分类进入到标签的名字集
        valNum = 0
        spectrogram_dir = spectrogram_path + sub_dir
        files = os.listdir(spectrogram_dir)
        for j in files:
            waveData = np.load(spectrogram_dir+"/"+j)
            waveData = np.transpose(waveData)
            waveData = np.pad(waveData,(((100-waveData.shape[0])/2, 100-waveData.shape[0]-(100-waveData.shape[0])/2), (0, 0)), mode='constant', constant_values=0)
            if valNum < 200:
                val_wavs.append(waveData)
                val_labels.append(labsIndName.index(sub_dir))
                valNum += 1
            else:
                wavs.append(waveData)
                labels.append(labsIndName.index(sub_dir))

    num_class = len(labsIndName)
    print "total", num_class, "class"
    wavs = np.array(wavs)
    labels = np.array(labels)
    val_wavs = np.array(val_wavs)
    val_labels = np.array(val_labels)
    return (wavs, labels), (val_wavs, val_labels), num_class, labsIndName

if __name__ == '__main__':
    (wavs, labels), (val_wavs, val_labels), num_class, labsIndName = create_datasets()
    wavs = wavs/float(np.max(wavs))
    val_wavs = val_wavs / float(np.max(val_wavs))
    #wavs = np.expand_dims(wavs, axis=-1)
    #val_wavs = np.expand_dims(val_wavs, axis=-1)
    # 标签转换为独热码
    labels = keras.utils.to_categorical(labels, num_class) #num_class = 10
    val_labels = keras.utils.to_categorical(val_labels, num_class)
    print wavs.shape, labels.shape
    print val_wavs.shape, val_labels.shape
    # 构建模型
    print "start to train the model..."
    # 模型输入层
    model_input = Input(shape=(100, 200), name="model_input")
    #GRU-layer 1
    gru1 = GRU(500, return_sequences=True)(model_input)
    batch_normal1 = BatchNormalization()(gru1)
    #GRU-layer 2
    gru2 = GRU(500, return_sequences=False)(gru1)
    batch_normal2 = BatchNormalization()(gru2)
    # output-layer
    model_output = Dense(10, activation="softmax")(batch_normal2)
    model = Model(inputs=model_input, outputs=model_output, name="gru2fc0")
    #model.summary()

    #模型编译
    model.compile(loss=categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=2e-2),
                  metrics=['accuracy'])


    # 模型训练,checkpoint
    filepath = "./realdata_model/gru2fc0softmax1-{epoch:02d}-{val_acc:.2f}.hdf5"
    # 中途训练效果提升, 则将文件保存, 每提升一次, 保存一次
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                 mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto')
    hist = model.fit(wavs, labels, batch_size=32, epochs=200, validation_data=(val_wavs, val_labels),
                     callbacks=[reduce_lr, checkpoint, early_stopping])
    # best_val_acc:0.60850