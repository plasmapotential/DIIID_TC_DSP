#filter_fullshot.py

# Date:         20190220
# Description:  Filter by frequency component
# Engineer:     Tom Looby

import os
import numpy as np
import pandas
import time
import datetime as dt
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import argrelextrema

#=============================================================================
#                       User defined constants
#=============================================================================
#Root directory where we are working
root_dir = '/home/workhorse/school/grad/DIIID/SETC_DATA/'
filename = '172402.txt'
delimit = '\s+' #regex for any whitespace, change as necessary

#Number of Frequency Components to Annotate in Plot
n_freqs = 2    #Change as necessary
sample_freq = 1000    #Change as necessary
nyquist = 2*sample_freq

pi = 3.1415

#=============================================================================
#                         Import Dataset
#=============================================================================
#Read Data from file
TCfile = root_dir + filename
TC_data = pandas.read_csv(TCfile, sep=delimit)
#print(TC_data.iloc[:,1])

#=============================================================================
#                         Pass signal through filter
#=============================================================================
f1 = 75
f2 = 700
f3 = 300
f4 = 200

Wl1 = f1/nyquist
Wh1 = f2/nyquist
#Wl2 = 2*pi*f3/1000
Wh2 = f4/nyquist
Npoles = 3

#Bandstop Filter
b, a = signal.butter(Npoles, [Wl1,Wh1], 'bandstop')
output_signal1 = signal.filtfilt(b, a, TC_data.iloc[:,1])

#High Pass Filter
#b, a = signal.butter(Npoles, Wh1, 'high')
#output_signal = signal.filtfilt(b, a, TC_data.iloc[:5000,1])

#Low Pass Filter
b2, a2 = signal.butter(4, Wh2, 'low')
output_signal = signal.filtfilt(b2, a2, output_signal1)


#=============================================================================
#                         Frequency Analysis
#=============================================================================
#Calculate spectral density using Welch method with hanning window
#fs is sampling frequency
f, Pxx_den = signal.welch(output_signal, fs=sample_freq)
#f, Pxx_den = signal.welch(TC_data.iloc[:5000,1], fs=1000)

#Find Maximum Frequency Components
# determine the indices of the local maxima
pwrmaxInd = argrelextrema(Pxx_den, np.greater)
#Build array of power maximums and indices
max_arr = np.zeros((len(pwrmaxInd[0]),2))
max_arr[:,0] = pwrmaxInd[0]
max_arr[:,1] = Pxx_den[pwrmaxInd[0]]
#Sort smallest to largest
max_arr = max_arr[max_arr[:,1].argsort()]
#Select n_freqs largest of local maximums
local_maxs = max_arr[-n_freqs:]
#Find these maxima in original raw signal
peaks = np.zeros((n_freqs,2))
for i in range(n_freqs):
   peaks[i,0] = f[np.where(Pxx_den == local_maxs[i][1])]
   peaks[i,1] = Pxx_den[np.where(Pxx_den == local_maxs[i][1])]


#=============================================================================
#                         Plots
#=============================================================================
#Dictionaries for plotting fonts
title_font = {'fontname':'Arial', 'size':'18', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'} # Bottom vertical alignment for more space
axis_font = {'fontname':'Arial', 'size':'14'}

#Plot Time Series and Spectrum
plt.figure(1)
plt.subplot(211)
plt.plot(TC_data.iloc[:,0],output_signal,'g')
plt.title('Filtered Signal: BandStop Filter (f_c = {:d}, {:d}) =>'
          ' Lowpass Filter (f_c = {:d})'.format(f1,f2,f4), **title_font)
plt.xlabel('Time [s]', **axis_font)
plt.ylabel('Temperature [degC]', **axis_font)
plt.subplots_adjust(hspace=.5)

plt.subplot(212)
plt.semilogy(f, Pxx_den)
plt.title('Spectral Density Estimation: Welch Method with Hanning Window', **title_font)
plt.xlabel('Frequency [Hz]', **axis_font)
plt.ylabel('PSD [V**2/Hz]', **axis_font)
for i in range(n_freqs):
    plt.annotate('f={:f}'.format(peaks[i,0]),
     xy=(peaks[i,0]+10, peaks[i,1]*0.9), xycoords='data', **axis_font)
plt.subplots_adjust(hspace=.35)

plt.show()
