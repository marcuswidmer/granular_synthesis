from tkinter import *
import numpy as np
import math as m
import struct
import time
import wave
import scipy.signal as signal
from pyaudio import PyAudio, paContinue, paInt16, paFloat32
from scipy.io import wavfile
from spectrum import LEVINSON
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
from threading import Event, Thread 


WINDOW_SIZE = 4*512 # 512 = 12 ms
GRAIN_SIZE = 10*256
p = PyAudio()
audio_data = np.zeros(WINDOW_SIZE)

fs, data = wavfile.read('hitting_glass_float_slow.wav')
data = (data[:,0]+data[:,1])/2
lower_freq_mod = 1
samp_length = int(lower_freq_mod * len(data))
actual_samp_length = len(data)
grain_samp_length = 10*samp_length
i = 0
num_windows = 30
num_windows_in_grain_sample = int(grain_samp_length/WINDOW_SIZE)
num_grains = int(samp_length/GRAIN_SIZE)
print(num_grains)
num_pitches = 1


audio_buffer = np.zeros([num_windows*WINDOW_SIZE,2])
grain_sample = np.zeros([samp_length,2])
##print("----------------------record device list---------------------")
#info = p.get_host_api_info_by_index(0)
#numdevices = info.get('deviceCount')
#for i in range(0, numdevices):
#        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
#            print ('Input Device id',i,'-', p.get_device_info_by_host_api_device_index(0, i).get('name'))


## --------- Making grains  ---------------
win = np.hanning(GRAIN_SIZE)
freq_mod = np.linspace(lower_freq_mod,1,num_pitches)
freq_mod = np.array([1,2,3,4,5,6,7,8])
num_pitches = len(freq_mod)
grain_mat = np.zeros([GRAIN_SIZE,num_grains,num_pitches])

for j in range(num_pitches):
    data_mod = signal.resample(data,int(freq_mod[j]*actual_samp_length))
    data_mod = data_mod[0:samp_length]
    for iii in range(num_grains):
       grain_mat[:,iii,j] = data_mod[iii*GRAIN_SIZE:(iii+1)*GRAIN_SIZE]*win

len_rand_seqs = 1000*WINDOW_SIZE
rand_time_arr = np.random.randint(0,int(grain_samp_length-GRAIN_SIZE)-1,len_rand_seqs)
rand_grain_arr = np.random.randint(0,num_grains,len_rand_seqs)
rand_ch_arr = np.random.randint(0,2,len_rand_seqs)
rand_pitch_arr = np.random.randint(0,num_pitches,len_rand_seqs)
reps = 3000


grain_sample = np.zeros([grain_samp_length,2])
def callback(in_data, frame_count, time_info, status):
    global i,grain_sample
    sample_counter = np.mod(i,num_windows_in_grain_sample)
    if sample_counter == 0:
        grain_sample = np.zeros([grain_samp_length,2])
        print('in the making_')
        for ii in range(reps):

            rand_counter = np.mod(i*reps+ii,len_rand_seqs)
            random_time_ind = rand_time_arr[rand_counter]
            random_grain = rand_grain_arr[rand_counter]
            random_channel = rand_ch_arr[rand_counter]
            random_pitch = rand_pitch_arr[rand_counter]

            grain_sample[random_time_ind:random_time_ind+GRAIN_SIZE,random_channel]+=grain_mat[:,random_grain,random_pitch]
    out_data=struct.pack('%sf' % (2*WINDOW_SIZE),*grain_sample[sample_counter*WINDOW_SIZE:(sample_counter+1)*WINDOW_SIZE,:].reshape(2*WINDOW_SIZE))
    
    i+=1
    return (out_data, paContinue)

# open stream using callback (3)
stream = p.open(format=paFloat32,
                channels=2,
                rate=fs,
                output = True,
                frames_per_buffer=WINDOW_SIZE,
                stream_callback=callback)

# start the stream
stream.start_stream()
while(1):
    pass

stream.stop_stream()
stream.close()

# close PyAudio
p.terminate()
