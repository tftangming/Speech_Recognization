#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-6-5 下午2:06
# @Author  : tangming
# @File    : generate_wav_spectrum.py

import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import PIL.Image as Image
import os
from os import listdir
from os.path import isfile, join

""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning): #帧移设为200,帧长设为400
    win = window(frameSize) #帧长为400的汉明窗
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    # print hopSize  # 200
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frameSize / 2.0))), sig)
    #print samples.shape
    # cols for windowing
    cols = int(np.ceil((len(samples) - frameSize) / float(hopSize)) + 1)
    #print cols # 80
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    #print samples.strides[0]  #8
    # 分帧
    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize),
                                      strides=(samples.strides[0] * hopSize, samples.strides[0]), writeable=False).copy()
    #print frames.shape  # 80,400
    # 加窗
    frames *= win
    # 快速傅里叶变换
    return np.fft.rfft(frames)  # (80,400)=>(80,201)

""" scale frequency axis logarithmically """
def logscale_spec(spec, sr=44100, factor=20., alpha=1.0, f0=0.9, fmax=1):
    spec = spec[:, 0:256]
    timebins, freqbins = np.shape(spec)
    scale = np.linspace(0, 1, freqbins)  # ** factor
    # http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=650310&url=http%3A%2F%2Fieeexplore.ieee.org%2Fiel4%2F89%2F14168%2F00650310
    scale = np.array(
        map(lambda x: x * alpha if x <= f0 else (fmax - alpha * f0) / (fmax - f0) * (x - f0) + alpha * f0, scale))
    scale *= (freqbins - 1) / max(scale)

    newspec = np.complex128(np.zeros([timebins, freqbins]))
    allfreqs = np.abs(np.fft.fftfreq(freqbins * 2, 1. / sr)[:freqbins + 1])
    freqs = [0.0 for i in range(freqbins)]
    totw = [0.0 for i in range(freqbins)]
    for i in range(0, freqbins):
        if (i < 1 or i + 1 >= freqbins):
            newspec[:, i] += spec[:, i]
            freqs[i] += allfreqs[i]
            totw[i] += 1.0
            continue
        else:
            # scale[15] = 17.2
            w_up = scale[i] - np.floor(scale[i])
            w_down = 1 - w_up
            j = int(np.floor(scale[i]))

            newspec[:, j] += w_down * spec[:, i]
            freqs[j] += w_down * allfreqs[i]
            totw[j] += w_down

            newspec[:, j + 1] += w_up * spec[:, i]
            freqs[j + 1] += w_up * allfreqs[i]
            totw[j + 1] += w_up

    for i in range(len(freqs)):
        if (totw[i] > 1e-6):
            freqs[i] /= totw[i]

    return newspec, freqs

""" obtain spectrogram"""
def save_spectrogram(audiopath, name, binsize=400, alpha=1, offset=0):
    samplerate, samples = wav.read(audiopath)
    #print samples.shape
    #samples = samples[:, channel]

    # short time fourier transform of audio signal
    s = stft(samples, binsize)
    #print s.shape  #(21, 513)
    # scale frequency axis logarithmically
    sshow, freq = logscale_spec(s, factor=1, sr=samplerate, alpha=alpha)
    sshow = sshow[2:, :]
    # amplitude to decibel
    sum = np.sum(np.int64(np.abs(sshow) <= 0))
    if sum > 0:
        return
    ims = 20. * np.log10(np.abs(sshow) / 10e-6)
    timebins, freqbins = np.shape(ims)
    #print ims.shape
    ims = np.pad(ims,
                      (((100 - ims.shape[0]) / 2, 100 - ims.shape[0] - (100 - ims.shape[0]) / 2),
                       (0, 0)), mode='constant', constant_values=0)
    ims = np.transpose(ims)
    # ims = ims[0:256, offset:offset+768] # 0-11khz, ~9s interval
    ims = ims[0:200, :]  # 0-11khz, ~10s interval
    print ims.shape
    image = Image.fromarray(ims)
    image = image.convert('RGB')
    image.save(name)
    #np.save(name, ims)

def cal_spectrogram(audiopath, binsize=400, alpha=1, offset=0):
    samplerate, samples = wav.read(audiopath)
    #print samples.shape

    # short time fourier transform of audio signal
    s = stft(samples, binsize)
    #print s.shape  #(80, 201)
    # scale frequency axis logarithmically
    sshow, freq = logscale_spec(s, factor=1, sr=samplerate, alpha=alpha)
    #print sshow.shape
    sshow = sshow[2:, :]
    # amplitude to decibel
    sum = np.sum(np.int64(np.abs(sshow) <= 0))
    if sum > 0:
        return
    ims = 20. * np.log10(np.abs(sshow) / 10e-6)
    timebins, freqbins = np.shape(ims)

    ims = np.transpose(ims)
    # ims = ims[0:256, offset:offset+768] # 0-11khz, ~9s interval
    ims = ims[0:200, :]  # 0-11khz, ~10s interval
    return ims

def dir_to_spectrogram(audio_dir, spectrogram_dir):
    # Creates spectrograms of all the audio files in a dir
    file_names = [f for f in listdir(audio_dir) if isfile(join(audio_dir, f)) and '.wav' in f]
    for file_name in file_names:
        #print(file_name)
        audio_path = audio_dir + "/" + file_name
        spectogram_path = spectrogram_dir + "/" + file_name.replace('.wav', '.npy')
        save_spectrogram(audio_path, spectogram_path)

if __name__ == '__main__':
    # dataset_path = "/home/tangming/Final_speech_recognization/aug_test_data_myself/"
    # spectrogram_path = "/home/tangming/Final_speech_recognization/aug_spec_test_data_myself/"
    # dirs = os.listdir(dataset_path)  # 获取各个类别的目录列表
    # for sub_dir in dirs:
    #     print "pre_process", sub_dir
    #     audio_dir = dataset_path + sub_dir
    #     spectrogram_dir = spectrogram_path + sub_dir
    #     if not os.path.exists(spectrogram_dir):
    #         os.makedirs(spectrogram_dir)
    #     dir_to_spectrogram(audio_dir, spectrogram_dir)
    wavs = cal_spectrogram("/home/tangming/cj_3A6jQmui.wav")
    save_spectrogram("/home/tangming/cj_3A6jQmui.wav", "test.jpg")

