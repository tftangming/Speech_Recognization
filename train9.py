#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-6-10 下午2:21
# @Author  : tangming
# @File    : train9.py


import os
from os import listdir
from os.path import isfile, join
import numpy as np
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, GRU, Dropout, Input, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Reshape
from keras.layers import add, Concatenate, Lambda
from keras.layers.merge import concatenate
from keras.losses import categorical_crossentropy
from keras import regularizers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from random import shuffle
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
import time

labsIndName = []
#加载数据集 和 标签[并返回标签集的处理结果]
def create_datasets():
    wavs = []  # wav文件集
    labels = []  # labels里面存的值都是对应标签的下标，下标对应的名字在labsInd中
    val_wavs = []  # wav文件集
    val_labels = []  # labels里面存的值都是对应标签的下标，下标对应的名字在labsInd中

    spectrogram_path = "/home/tangming/Automatic_Speech_Recognization/dataset2/spectrogram_dataset/"
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
            waveData = np.pad(waveData,((0, 80-waveData.shape[0]), (0, 0)), mode='constant', constant_values=0)
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
    shared_gru = GRU(n_units, return_sequences=False)
    for i in range(len(x)):
        gru_out = shared_gru(x[i])
        #print i
        #print shared_gru.get_input_at(i)
        #print shared_gru.get_output_at(i)
        bn_out = BatchNormalization()(gru_out)
        outputs.append(bn_out)
    return outputs


class LossHistory(keras.callbacks.Callback):
    # 函数开始时创建盛放loss与acc的容器
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    # 按照batch来进行追加数据
    def on_batch_end(self, batch, logs={}):
        # 每一个batch完成后向容器里面追加loss，acc
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))
        # 每五秒按照当前容器里的值来绘图
        # if int(time.time()) % 5 == 0:
        #     self.draw_p(self.losses['batch'], 'loss', 'train_batch')
        #     self.draw_p(self.accuracy['batch'], 'acc', 'train_batch')
        #     self.draw_p(self.val_loss['batch'], 'loss', 'val_batch')
        #     self.draw_p(self.val_acc['batch'], 'acc', 'val_batch')

    def on_epoch_end(self, batch, logs={}):
        # 每一个epoch完成后向容器里面追加loss，acc
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))
        # 每五秒按照当前容器里的值来绘图
        # if int(time.time()) % 5 == 0:
        #     self.draw_p(self.losses['epoch'], 'loss', 'train_epoch')
        #     self.draw_p(self.accuracy['epoch'], 'acc', 'train_epoch')
        #     self.draw_p(self.val_loss['epoch'], 'loss', 'val_epoch')
        #     self.draw_p(self.val_acc['epoch'], 'acc', 'val_epoch')

    # 绘图，这里把每一种曲线都单独绘图，若想把各种曲线绘制在一张图上的话可修改此方法
    def draw_p(self, lists, label, type):
        plt.figure()
        plt.plot(range(len(lists)), lists, 'r', label=label)
        plt.ylabel(label)
        plt.xlabel(type)
        plt.legend(loc="upper right")
        file_path = './model_tip/' + type + '_' + label + '.jpg'
        plt.savefig(file_path)

    # 由于这里的绘图设置的是5s绘制一次，当训练结束后得到的图可能不是一个完整的训练过程（最后一次绘图结束，有训练了0-5秒的时间）
    # 所以这里的方法会在整个训练结束以后调用
    def end_draw(self):
        #self.draw_p(self.losses['batch'], 'loss', 'train_batch')
        #self.draw_p(self.accuracy['batch'], 'acc', 'train_batch')
        #self.draw_p(self.val_loss['batch'], 'loss', 'val_batch')
        #self.draw_p(self.val_acc['batch'], 'acc', 'val_batch')
        self.draw_p(self.losses['epoch'], 'loss', 'train_epoch')
        self.draw_p(self.accuracy['epoch'], 'acc', 'train_epoch')
        self.draw_p(self.val_loss['epoch'], 'loss', 'val_epoch')
        self.draw_p(self.val_acc['epoch'], 'acc', 'val_epoch')

def create_model():
    print "start to train the model..."
    # 模型输入层
    model_input = Input(shape=(80, 200, 1), name="model_input")
    # CONV-RELU-POOL 1
    conv1 = Conv2D(16, (5, 5), activation="relu", kernel_regularizer=regularizers.l2(0.01))(model_input)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 2), padding='same')(conv1)
    batch_normal1 = BatchNormalization(axis=-1)(pool1)
    # CONV-RELU-POOL 2
    conv2 = Conv2D(32, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(0.01))(batch_normal1)
    pool2 = MaxPooling2D(pool_size=(3, 3), strides=(1, 2), padding='same')(conv2)
    batch_normal2 = BatchNormalization(axis=-1)(pool2)
    # CONV-RELU-POOL 3
    conv3 = Conv2D(32, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(0.01))(batch_normal2)
    pool3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 2), padding='same')(conv3)
    batch_normal3 = BatchNormalization(axis=-1)(pool3)
    # CONV-RELU-POOL 3
    conv4 = Conv2D(32, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(0.01))(batch_normal3)
    pool4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 2), padding='same')(conv4)
    batch_normal4 = BatchNormalization(axis=-1)(pool4)
    # check the output shape of the conv layer
    # model = Model(inputs=model_input, outputs=batch_normal4, name="cnn3gru1fc1")
    # model.summary()
    # the result of the check
    # filter_width = 70
    # filter_height = 11
    # filter_channel = 32
    # change the location of the channel into the rnn layer
    gru_input = Reshape((70, 352))(batch_normal4)

    # single-GRU-layers
    gru_output = GRU(128, return_sequences=False)(gru_input)
    batch_normal_5 = BatchNormalization(axis=-1)(gru_output)
    dropout1 = Dropout(0.4)(batch_normal_5)
    # output-layer
    model_output = Dense(10, activation="softmax", kernel_initializer='random_normal',
                         kernel_regularizer=regularizers.l2(0.01))(dropout1)
    model = Model(inputs=model_input, outputs=model_output)
    model.compile(loss=categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=2e-3, momentum=0.8), metrics=['accuracy'])
    #model.summary()
    return model

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
    model = create_model()
    #model = KerasClassifier(build_fn=create_model, nb_epoch=100, batch_size=128, verbose=0)

    # 创建一个实例history
    history = LossHistory()

    # 模型训练,checkpoint
    filepath = "./model-0/cnn4no_pool_single_gru1fc0shared-{epoch:02d}-{val_acc:.2f}.hdf5"
    # 中途训练效果提升, 则将文件保存, 每提升一次, 保存一次
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                 mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=2)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=5, mode='auto')
    hist = model.fit(wavs, labels, batch_size=16, epochs=100, validation_data=(val_wavs, val_labels),
                     callbacks=[reduce_lr, history, checkpoint, early_stopping])

    # with open('./history/train8.txt', 'w') as f:
    #     f.write(str(hist.history))
    # 模型保存
    #model.save("./model/cnn4no_pool_single_gru1fc0shared.h5")
    history.end_draw()

    # <batch_size> dropout=0.4
    # batch_size = 128, epoch=100, best_val_acc=0.95250;
    # batch_size = 64, epoch=100, best_val_acc=0.96100;
    # batch_size = 32, epoch=100, best_val_acc=0.95950
    # batch_size = 16, epoch=100, best_val_acc=0.96700
