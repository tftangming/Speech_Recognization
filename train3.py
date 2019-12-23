#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-6-6 上午8:49
# @Author  : tangming
# @File    : train3.py

import os
from os import listdir
from os.path import isfile, join
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, GRU, Dropout, Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten
from keras.layers import add, Concatenate, Lambda
from keras.layers.merge import concatenate
from keras.losses import categorical_crossentropy
from keras import regularizers
from random import shuffle
from keras import backend as K
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
            waveData = np.pad(waveData,((0, 100-waveData.shape[0]), (0, 0)), mode='constant', constant_values=0)
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

def expand_dim_backend(x):
    x1 = K.expand_dims(x, -1)
    return x1

def split_feature_map(x, num_channel):
    x_batch = []
    for i in range(num_channel):
        x1 = x[:, :, :, i]
        x_batch.append(x1)
    return x_batch

def multi_gru_bn(x, n_units):
    outputs = []
    for i in range(32):
        shared_gru = GRU(n_units, return_sequences=False)
        gru_out = shared_gru(x[i])
        print i
        print shared_gru.input
        print shared_gru.output
        bn_out = BatchNormalization()(gru_out)
        outputs.append(bn_out)
    return outputs



if __name__ == '__main__':
    (wavs, labels), (val_wavs, val_labels), num_class, labsIndName = create_datasets()
    wavs = wavs/float(np.max(wavs))
    val_wavs = val_wavs / float(np.max(val_wavs))
    wavs = np.expand_dims(wavs, axis=-1)
    val_wavs = np.expand_dims(val_wavs, axis=-1)
    # 标签转换为独热码
    labels = keras.utils.to_categorical(labels, num_class) #num_class = 10
    val_labels = keras.utils.to_categorical(val_labels, num_class)
    print wavs.shape, labels.shape
    print val_wavs.shape, val_labels.shape
    # 构建模型
    print "start to train the model..."
    # 模型输入层
    model_input = Input(shape=(100, 200, 1), name="model_input")
    # CONV-RELU-POOL 1
    conv1 = Conv2D(16, (5, 5), activation="relu")(model_input)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(conv1)
    batch_normal1 = BatchNormalization(axis=-1)(pool1)
    # CONV-RELU-POOL 2
    conv2 = Conv2D(32, (3, 3), activation="relu")(batch_normal1)
    pool2 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(conv2)
    batch_normal2 = BatchNormalization(axis=-1)(pool2)
    # CONV-RELU-POOL 3
    conv3 = Conv2D(32, (3, 3), activation="relu")(batch_normal2)
    pool3 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(conv3)
    batch_normal2 = BatchNormalization(axis=-1)(pool3)
    #test_out = Lambda(split_feature_map, arguments = {'num_channel': 32})(batch_normal2)
    #print len(test_out)
    #model = Model(inputs=model_input, outputs=test_out, name="cnn3fc1")

    # multi-GRU-layers concatenate
    num_channels = 32
    x_batch = []
    for i in range(32):
        x_output = Lambda(lambda x: x[:, :, :, i])(batch_normal2)
        x_batch.append(x_output)

    outputs = []
    #shared_gru = GRU(15, return_sequences=False)
    for i in range(32):
        output = GRU(15, return_sequences=False)(x_batch[i])
        output_bn = BatchNormalization(axis=-1)(output)
        outputs.append(output_bn)

    # concatenate-layer
    gru_output = Concatenate(axis=-1)(outputs)

    # ful-connected layer
    #flatten1 = Flatten()(gru_output)
    dense1 = Dense(128, activation="relu")(gru_output)
    batch_normal_out = BatchNormalization(axis=-1)(dense1)
    dropout1 = Dropout(0.3)(batch_normal_out)
    # output-layer
    model_output = Dense(10,activation="softmax")(dropout1)
    model = Model(inputs=model_input, outputs=model_output, name="cnn3gru1fc1")
    model.summary()

    # 模型编译
    model.compile(loss=categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=1e-3),
                  metrics=['accuracy'])
    # 模型训练,checkpoint
    filepath = "./realdata_model/cnn4no_pool_single_gru1fc0shared-{epoch:02d}-{val_acc:.2f}.hdf5"
    # 中途训练效果提升, 则将文件保存, 每提升一次, 保存一次
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                 mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto')
    hist = model.fit(wavs, labels, batch_size=32, epochs=100, validation_data=(val_wavs, val_labels),
                     callbacks=[reduce_lr, checkpoint, early_stopping])
    # best_val_acc:0.89200