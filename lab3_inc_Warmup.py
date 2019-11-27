import numpy as np
import numpy.matlib
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from random import gauss
import cv2
from scipy.stats import uniform
from scipy.stats import norm
from PIL import Image as PilImg

import scipy


########################################################
########################################################
####        LAB 3 ######################################
########################################################
########################################################

########################################################
## Functions Start #####################################
########################################################



########################################################
## Functions End #######################################
########################################################


########################################################
### 1.1
########################################################
# # Setting the Constants
# a = 7
# fs = 200
# f_1 = 10
# f_2 = 10.541
# Ts = 1/fs
# t = np.arange(start=0, stop=1, step=Ts)
# # Sine Waves
# sine_wave = a * np.sin(2 * np.pi * f_1 * t)
# sine_wave_2 = a * np.sin(2 * np.pi * f_2 * t)
# sine_wave_3 = 127 * (np.sin(2 * np.pi * f_1 * t) + 1)
# # FFT and FFT derivatives
# F = np.fft.fft(sine_wave)
# F_abs = abs(F)
# F_abs_shift = abs(np.fft.fftshift(F_abs))
#
# F_log = np.log(F_abs)
# F_log_shift = np.log(F_abs_shift)
# # F_angle = np.fft.fftshift(np.angle(F))
#
# F_angle = np.angle(F)
#
# # f = 10.541
# F_2 = np.fft.fft(sine_wave_2)
# F_abs_2 = abs(F_2)
# F_abs_shift_2 = abs(np.fft.fftshift(F_abs_2))
#
# F_log_2 = np.log(F_abs_2)
# F_log_shift_2 = np.log(F_abs_shift_2)
# F_angle_2 = np.fft.fftshift(np.angle(F_2))
#
# # a = 127
# F_3 = np.fft.fft(sine_wave_3)
# F_abs_3 = abs(F_3)
# F_abs_shift_3 = abs(np.fft.fftshift(F_abs_3))
#
# F_log_3 = np.log(F_abs_3)
# F_log_shift_3 = np.log(F_abs_shift_3)
# F_angle_3 = np.fft.fftshift(np.angle(F_3))
#
# # Plots
# plt.figure(1)
# plt.subplot(2, 3, 1)
# plt.plot(t, sine_wave)
# plt.title('Sine Wave')
# plt.subplot(2, 3, 2)
# plt.plot(t, F_abs)
# plt.title('F Magnitude')
# plt.subplot(2, 3, 3)
# plt.plot(t, F_abs_shift)
# plt.title('F shift')
# plt.subplot(2, 3, 4)
# plt.plot(t, F_log)
# plt.title('F Log')
# plt.subplot(2, 3, 5)
# plt.plot(t, F_log_shift)
# plt.title('F Log Shift')
# plt.subplot(2, 3, 6)
# plt.plot(t, F_angle)
# plt.title('F Phase')
#
# plt.figure(2)
# plt.subplot(2, 3, 1)
# plt.plot(t, sine_wave_2)
# plt.title('Sine Wave')
# plt.subplot(2, 3, 2)
# plt.plot(t, F_abs_2)
# plt.title('F Magnitude')
# plt.subplot(2, 3, 3)
# plt.plot(t, F_abs_shift_2)
# plt.title('F shift')
# plt.subplot(2, 3, 4)
# plt.plot(t, F_log_2)
# plt.title('F Log')
# plt.subplot(2, 3, 5)
# plt.plot(t, F_log_shift_2)
# plt.title('F Log Shift')
# plt.subplot(2, 3, 6)
# plt.plot(t, F_angle_2)
# plt.title('F Phase')
#
# plt.figure(3)
# plt.subplot(2, 3, 1)
# plt.plot(t, sine_wave_3)
# plt.title('Sine Wave')
# plt.subplot(2, 3, 2)
# plt.plot(t, F_abs_3)
# plt.title('F Magnitude')
# plt.subplot(2, 3, 3)
# plt.plot(t, F_abs_shift_3)
# plt.title('F shift')
# plt.subplot(2, 3, 4)
# plt.plot(t, F_log_3)
# plt.title('F Log')
# plt.subplot(2, 3, 5)
# plt.plot(t, F_log_shift_3)
# plt.title('F Log Shift')
# plt.subplot(2, 3, 6)
# plt.plot(t, F_angle_3)
# plt.title('F Phase')
# plt.show()

########################################################
### 1.2
########################################################
# # Setting the Constants
# a = 7
# fs = 200
# f_1 = 10
# f_2 = 10.541
# Ts = 1/fs
# t = np.arange(start=0, stop=1, step=Ts)
# # Sine Waves
# sine_wave = a * np.sin(2 * np.pi * f_1 * t)
# # FFT and iFFT
# F = np.fft.fft(sine_wave)
# F_inv = np.fft.ifft(F)
# delta = sine_wave - F_inv
# # Plots
# plt.subplot(3, 1, 1)
# plt.plot(t, sine_wave)
# plt.title('Original')
# plt.subplot(3, 1, 2)
# plt.plot(t, F_inv)
# plt.title('Inverse')
# plt.subplot(3, 1, 3)
# plt.plot(delta)
# plt.title('Error between')
# plt.show()
#
########################################################
### 1.3 - 1D Convolution
########################################################
# # Arrays
# p = [1, 2, 3, 1, 2]
# h = [1, 1, 0, 0, 0]
#
# conv_ph = np.convolve(p, h)
# print(conv_ph)
# plt.subplot(3, 1, 1)
# plt.stem(p)
# plt.xticks(np.arange(0, 10, 1))
# plt.yticks(np.arange(0, 5, 1))
# plt.title('f[n]')
# plt.xlabel('n')
# plt.ylabel('Magnitude')
#
# plt.subplot(3, 1, 2)
# plt.stem(h)
# plt.xticks(np.arange(0, 10, 1))
# plt.yticks(np.arange(0, 5, 1))
# plt.title('h[n]')
# plt.xlabel('n')
# plt.ylabel('Magnitude')
# plt.xlabel('n')
# plt.ylabel('Magnitude')
#
# plt.subplot(3, 1, 3)
# plt.stem(conv_ph)
# plt.xticks(np.arange(0, 10, 1))
# plt.yticks(np.arange(0, 5, 1))
# plt.title('convolution of f[n] and h[n]')
# plt.xlabel('n')
# plt.ylabel('Magnitude')
# plt.show()
########################################################
### 1.4 - 1D filtered signals
########################################################


