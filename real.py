from tensorflow.keras.models import load_model
import pyaudio
import struct
import time
import librosa
import numpy as np

#cough_model = load_model("vgg16_model_15_frozen_layers.json")

# constants
CHUNK = 55125            # samples per frame
FORMAT = pyaudio.paInt16     # audio format (bytes per sample?)
CHANNELS = 1                 # single channel for microphone
RATE = 22050                 # samples per second

# pyaudio class instance
p = pyaudio.PyAudio()

# stream object to get data from microphone
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK
)

print('stream started')

# for measuring frame rate
frame_count = 0
start_time = time.time()

while True:
    
    # binary data
    data = stream.read(CHUNK)  
    #print(len(data))
    # convert data to integers, make np array, then offset it by 127
    data_int = struct.unpack(str(2 * CHUNK) + 'B', data)
    
    # create np array and offset by 128
    data_np = (np.array(data_int, dtype='b')).astype('float64')
    print(len(data_np))
    # Compute spectrogram
   # M = librosa.feature.melspectrogram(data_np, RATE, 
                                       #fmax = RATE/2, # Maximum frequency to be used on the on the MEL scale
                                       #n_fft=2048, 
                                       #hop_length=512, 
                                       #n_mels = 96, # As per the Google Large-scale audio CNN paper
                                       #power = 2) # Power = 2 refers to squared amplitude
    
    #print(M)
   
    