from tensorflow.keras.models import load_model
import pyaudio
import struct
import time
import librosa
import numpy as np
from matplotlib import cm
import pylab
import cv2
import librosa.display

model = load_model("cough_detector.model")

import microphones
desc, mics, indices = microphones.list_microphones()
print(mics)
MICROPHONE_INDEX = indices[0]


# Find description that matches the mic index
mic_desc = ""
for k in range(len(indices)):
    
    i = indices[k]
    #print(i)
    if (i==MICROPHONE_INDEX):
        print(i)
        mic_desc = mics[k]
print("Using mic: %s" % mic_desc)

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
    data_np = (np.array(data_int, dtype='b')).astype('float32')
    length = data_np.shape[0] / RATE
    #print(length)
    # Compute spectrogram
    M = librosa.feature.melspectrogram(data_np, RATE, 
                                       fmax = RATE/2, # Maximum frequency to be used on the on the MEL scale
                                       n_fft=2048, 
                                       hop_length=512, 
                                       n_mels = 96, # As per the Google Large-scale audio CNN paper
                                       power = 2) # Power = 2 refers to squared amplitude
    
    #print(M.shape)
    frame_count += 1
    log_power = librosa.power_to_db(M, ref=np.max)# Covert to dB (log) scale
    # # Plotting the spectrogram and save as JPG without axes (just the image)
    pylab.figure(figsize=(3,3))
    pylab.axis('off') 
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    librosa.display.specshow(log_power, cmap=cm.jet)
    pylab.savefig('record_'+str(frame_count)+ '_.jpg', bbox_inches=None, pad_inches=0)
    pylab.close()
    img_array = cv2.imread('record_'+str(frame_count)+ '_.jpg')  # convert to array
    new_array = cv2.resize(img_array, (224, 224))  # resize to normalize data size
    new_array=np.array(new_array).reshape(-1, 224, 224, 3)
    new_array = np.array(new_array, dtype="float32")
    #print(new_array.shape)
    (cough, notcough) = model.predict(new_array)[0]
    print("cough:",cough)
    print("notcough:",notcough)