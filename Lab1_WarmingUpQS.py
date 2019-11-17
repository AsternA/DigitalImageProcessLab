import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from random import gauss
import cv2



########################################
#PART 1 OF LAB #1 - WARM UP QUESTIONS
########################################

########################################
# ex 1.1 - Square inside a square
########################################
img1 = np.ones((400, 400, 3), dtype=np.uint8) * 50 #Create a uint8 400x400 matrix with backgroung pixels=50
img1[150:250, 150:250] = 100                       #Create the smaller sqaure with background pixels=100
fimg = Image.fromarray(img1, 'RGB')

fimg.show()



########################################
# ex 1.2 - Stripes uint8
########################################
# img2=np.zeros((200, 300, 3), dtype=np.uint8)   #Create the image size required and its type (uint8)
# for i in range(0,300,50):
#    img2[0:200, i:i+50] = i
#
# fimg2 = Image.fromarray(img2, 'RGB')
#
# fimg2.show()


###########################################
# #ex 1.3 - Sinusoidal Wave with fs = 200
###########################################
# a = 7
# fs = 200
# f = 10
# Ts = 1/fs
# t = np.arange(start = 0, stop = 1, step = Ts)
#
#
# y = a * np.sin(2 * np.pi * f * t)
# plt.plot(t, y)
# plt.show();
#
# fftsig = np.fft.fft(y)
# afftsig = abs(fftsig)
# x = np.arange(start = -100, stop = 100, step = 1)
#
# shiftfft = np.fft.fftshift(afftsig)
# plt.plot(x, shiftfft)
# plt.show()


########################################
#ex 1.3 Sinusoidal Wave with fs=12
########################################
# a = 7
# fs = 12
# f = 10
# Ts = 1/fs
# t = np.arange(start = 0, stop = 1, step = Ts)
#
# y = a * np.cos(2 * np.pi * f * t)
# plt.plot(t, y)
# plt.show()
#
# fftsig = np.fft.fft(y)
# afftsig = abs(fftsig)
# x = np.arange(start = -fs/2, stop = fs/2, step = 1)
# shiftfft = np.fft.fftshift(afftsig)
# plt.plot(x, shiftfft)
# plt.show()

########################################
#ex 1.4.2 - convolution
########################################
# f = [1, 2, 3, 1, 2]
# h = [1 ,1]
#
# conv = np.convolve(f, h)
# print(conv)
# plt.subplot(3,1,1)
# plt.stem(f)
# plt.xticks(np.arange(0, 10, 1))
# plt.yticks(np.arange(0, 5, 1))
# plt.title('f[n]')
# plt.xlabel('n')
# plt.ylabel('Magnitude')
#
# plt.subplot(3,1,2)
# plt.stem(h)
# plt.xticks(np.arange(0, 10, 1))
# plt.yticks(np.arange(0, 5, 1))
# plt.title('h[n]')
# plt.xlabel('n')
# plt.ylabel('Magnitude')
# plt.xlabel('n')
# plt.ylabel('Magnitude')
#
# plt.subplot(3,1,3)
# plt.stem(conv)
# plt.xticks(np.arange(0, 10, 1))
# plt.yticks(np.arange(0, 5, 1))
# plt.title('convolution of f[n] and h[n]')
# plt.xlabel('n')
# plt.ylabel('Magnitude')
#
# plt.show()

###############################################
#ex 1.4.3 - White Noise (4 different signals)
###############################################
# num_samples = 1000
# n1 = np.random.normal(0, 1, size=num_samples)
# plt.subplot(4,1,1)
# plt.plot(n1)
# plt.title('White Noise #1')
# n2 = [gauss(0.0, 1.0) for i in range(num_samples)]
# plt.subplot(4,1,2)
# plt.plot(n2)
# plt.title('White Noise #2')
#
# n3 = [gauss(0.0, 5.0) for i in range(num_samples)]
# plt.subplot(4,1,3)
# plt.plot(n3)
# plt.title('White Noise #3')
#
#
# n4 = np.random.normal(5, 5, size=num_samples)
# plt.subplot(4,1,4)
# plt.plot(n4)
# plt.title('White Noise #4')
#
#
#
#
# plt.show()
#
# # Histograms of White Noise
# plt.subplot(4,1,1)
# plt.hist(n1, density = 1)
# plt.title('Histogram of White Noise #1')
#
# plt.subplot(4,1,2)
# plt.hist(n2)
# plt.title('Histogram of White Noise #2')
#
# plt.subplot(4,1,3)
# plt.hist(n3)
# plt.title('Histogram of White Noise #3')
#
# plt.subplot(4,1,4)
# plt.hist(n4)
# plt.title('Histogram of White Noise #4')
#
#
#
# plt.show()

########################################
#ex 1.4.4 - Filter Noise from Signal
########################################

# a = 7
# fs = 200
# f = 10
# Ts = 1/fs
# t = np.arange(start = 0, stop = 1, step = Ts)
# y = a * np.sin(2 * np.pi * f * t)
# wn = [gauss(0.0, 1.0) for i in range(fs)]
# noisy_signal = y + wn
#
# h1 = [ 0.2, 0.2, 0.2, 0.2, 0.2]  #LPF given
#
# conv_ns = np.convolve(noisy_signal, h1)   #convolution between sinusoidal and LPF gives clean signal
#
# plt.subplot(3,1,1)
# plt.plot(t, y)
# plt.title('Original Signal')
# plt.subplot(3,1,2)
# plt.plot(t, noisy_signal)
# plt.title('Noisy Signal')
# plt.subplot(3,1,3)
# plt.plot(conv_ns)
# plt.title('Signal After Convolution with LPF')
# plt.show();


