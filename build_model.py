#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-7-4 下午6:14
# @Author  : tangming
# @File    : build_model.py


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

def create_model():
    print "start to train the model..."
    # 模型输入层
    model_input = Input(shape=(100, 200, 1), name="model_input")
    # CONV-RELU-POOL 1
    conv1 = Conv2D(16, (7, 7), activation="relu", kernel_regularizer=regularizers.l2(0.01))(model_input)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(1, 2), padding='same')(conv1)
    batch_normal1 = BatchNormalization(axis=-1)(pool1)
    # CONV-RELU-POOL 2
    conv2 = Conv2D(32, (5, 5), activation="relu", kernel_regularizer=regularizers.l2(0.01))(batch_normal1)
    pool2 = MaxPooling2D(pool_size=(3, 3), strides=(1, 2), padding='same')(conv2)
    batch_normal2 = BatchNormalization(axis=-1)(pool2)
    # CONV-RELU-POOL 3
    conv3 = Conv2D(32, (5, 5), activation="relu", kernel_regularizer=regularizers.l2(0.01))(batch_normal2)
    pool3 = MaxPooling2D(pool_size=(3, 3), strides=(1, 2), padding='same')(conv3)
    batch_normal3 = BatchNormalization(axis=-1)(pool3)
    # CONV-RELU-POOL 4
    conv4 = Conv2D(32, (3, 3), activation="relu", kernel_regularizer=regularizers.l2(0.01))(batch_normal3)
    pool4 = MaxPooling2D(pool_size=(3, 3), strides=(1, 2), padding='same')(conv4)
    batch_normal4 = BatchNormalization(axis=-1)(pool4)


    # check the output shape of the conv layer
    # model = Model(inputs=model_input, outputs=batch_normal4, name="cnn3gru1fc1")
    # model.summary()
    # the result of the check
    # filter_width = 84
    # filter_height = 5
    # filter_channel = 32
    # change the location of the channel into the rnn layer
    gru_input = Reshape((84, 320))(batch_normal4)

    # single-GRU-layers
    gru_output = GRU(128, kernel_regularizer=regularizers.l2(0.01), return_sequences=True)(gru_input)
    batch_normal_5 = BatchNormalization(axis=-1)(gru_output)
    dropout1 = Dropout(0.4)(batch_normal_5)
    # single-GRU-layers
    gru_output2 = GRU(256, kernel_regularizer=regularizers.l2(0.01), return_sequences=False)(dropout1)
    batch_normal_6 = BatchNormalization(axis=-1)(gru_output2)
    dropout2 = Dropout(0.4)(batch_normal_6)
    #ful-connected layer

    # output-layer
    model_output = Dense(10, activation="softmax", kernel_initializer='random_normal',
                         kernel_regularizer=regularizers.l2(0.01))(dropout2)
    model = Model(inputs=model_input, outputs=model_output)
    model.compile(loss=categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=2e-3, momentum=0.8), metrics=['accuracy'])
    model.summary()
    return model
model = create_model()