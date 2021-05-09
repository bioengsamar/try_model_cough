# -*- coding: utf-8 -*-
"""
Created on Sat May  8 22:26:50 2021

@author: Youssef
"""
from scipy.signal import butter, lfilter
from scipy.io import wavfile


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y
samplerate, data = wavfile.read('cough_sound.wav')
print(data)
if __name__ == '__main__':
    filtered_data= butter_bandpass_filter(data, 5, 30, samplerate, order=5)
    print(filtered_data)
    

