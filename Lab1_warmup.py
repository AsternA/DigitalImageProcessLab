import numpy as np
import numpy.matlib
from PIL import Image
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from random import gauss
import cv2
#print(cv2.__version__)

font = cv2.FONT_HERSHEY_SIMPLEX
#--- Here I am creating the border---
black = [0,0,0]     #---Color of the border---
#constant=cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=black )
#--- Here I created a violet background to include the text ---
#white = np.zeros((100, constant.shape[1], 3), np.uint8)
#white[:] = (255, 255, 255)

# --- I then concatenated it vertically to the image with the border ---

#vcat = cv2.vconcat((violet, constant))
#
#ex 1
# k = 40
# img1 = np.ones((400, 400, 3), dtype=np.uint8) * k
# for i in range (k-10,k+10,1):
#     img1[150:250, 150:250] = i
#     constant=cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=black )
#     white = np.zeros((100, constant.shape[1], 3), np.uint8)
#     white[:] = (255, 255, 255)
#     vcat = cv2.vconcat((white,constant))
#     cv2.putText(vcat, "Background Intensity:{k} Object Intensity: {i}".format(k=k,i=i) , (30, 50), font, 0.5, (0, 0, 0), 1, 0)
#     cv2.imshow('vcat', vcat)
#     plt.hist(img1.ravel(),256,[0,256])
#     plt.show()
#     cv2.waitKey(0)



# ex

# fimg = Image.fromarray(img1, 'RGB')
# fimg.save('Lab1ex1.png')
#
# fimg.show()


# # Self practice:
# k = 50
# img = np.ones([400,400], dtype=np.uint8) * k
# #img1 = Image.fromarray(img)
# #for i in range (k - 10, k+10, 1):
# img[150:250, 150:250] = 80
# #img2 = Image.fromarray(img)
# plt.imshow(img)
#     #plt.subplot(2,1,1)
#     #plt.imshow(img2)
#     #cv2.subplot(2,1,1)
#     #cv2.imshow("window1", img1)
#     #plt.subplot(2,1,2)
#     #cv2.subplot(2,1,2)
#     #plt.hist(img)


# ex 1.2
# img2=np.zeros((200, 300, 3), dtype=np.uint8)
# for i in range(0,300,50):
#    img2[0:200, i:i+50] = i
#
# fimg2 = Image.fromarray(img2, 'RGB')
# fimg2.save('Lab1ex2.png')
# fimg2.show()
#
# #ex 1.3
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

# ex 1.3 with fs=12
# a = 7
# fs = 12
# f = 10
# Ts = 1 / fs
# t = np.arange(start=0, stop=1, step=Ts)
#
# y = a * np.cos(2 * np.pi * f * t)
# plt.plot(t, y)
# plt.show()
#
# fftsig = np.fft.fft(y)
# afftsig = abs(fftsig)
# x = np.arange(start=-fs / 2, stop=fs / 2, step=1)
# shiftfft = np.fft.fftshift(afftsig)
# plt.plot(x, shiftfft)
# plt.show()


##############################
# ex 1.4.2 - convolution
##############################
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
#

#ex 1.4.3 - white noise
#
# num_samples = 1000
# n1 = np.random.normal(0, 1, size=num_samples)
# plt.subplot(5,1,1)
# plt.plot(n1)
# plt.title('White Noise #1')
# n2 = [gauss(0.0, 1.0) for i in range(num_samples)]
# plt.subplot(5,1,2)
# plt.plot(n2)
# plt.title('White Noise #2')
#
# n3 = [gauss(0.0, 5.0) for i in range(num_samples)]
# plt.subplot(5,1,3)
# plt.plot(n3)
# plt.title('White Noise #3')
#
#
# n4 = np.random.normal(5, 5, size=num_samples)
# plt.subplot(5,1,4)
# plt.plot(n4)
# plt.title('White Noise #4')
#
# n5 = np.random.normal(10, 10, size=num_samples)
# plt.subplot(5,1,5)
# plt.plot(n5)
# plt.title('White Noise #5')
#
#
# plt.show()
#
# plt.subplot(5,1,1)
# plt.hist(n1, density= 1)
# plt.title('Histogram of White Noise #1')
#
# plt.subplot(5,1,2)
# plt.hist(n2)
# plt.title('Histogram of White Noise #2')
#
# plt.subplot(5,1,3)
# plt.hist(n3)
# plt.title('Histogram of White Noise #3')
#
# plt.subplot(5,1,4)
# plt.hist(n4)
# plt.title('Histogram of White Noise #4')
#
# plt.subplot(5,1,5)
# plt.hist(n5)
# plt.title('Histogram of White Noise #5')
#
# plt.show()

# ex 1.4.4
# a = 7
# fs = 200
# f = 10
# Ts = 1/fs
# t = np.arange(start = 0, stop = 1, step = Ts)
# y = a * np.sin(2 * np.pi * f * t)
# wn = [gauss(0.0, 1.0) for i in range(fs)]
# noisy_signal = y + wn
#
# h1 = [ 0.2, 0.2, 0.2, 0.2, 0.2]
#
# conv_ns = np.convolve(noisy_signal, h1)
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

############################################################################################################
############################################################################################################
#####################################  Exercise 2 ##########################################################
############################################################################################################
############################################################################################################

############################################################################################################
# # Ex 2.1.1 - Background and BBobject
############################################################################################################

# k = 200
# img1 = np.ones((400, 400), dtype=np.uint8) * k
# for i in range (k-10,k+11,1):
#     img1[150:250, 150:250] = i
#     plt.subplot(211)
#     plt.imshow(img1, cmap='gray', vmin=0, vmax=255)
#     plt.title("Background Intensity: {k} Object Intensity: {i}".format(k=k,i=i))
#     Cw = ((float(img1[200,200]) - float(img1[1,1]))/float(img1[1,1]))
#     plt.subplot(212)
#     plt.hist(img1.ravel(),256,[0,256])
#     plt.title("Contrast Webber: {}".format(Cw))
#     plt.show()
#     plt.pause(1)
#

## Checked by Almog on 6.11.19 - Looks Completed! ##

############################################################################################################
# # ex 2.1.2 - Mach Bands
############################################################################################################

# N = 255
# ramp = np.linspace(start=0, stop=N, num=N)
# ramp_arr = np.array(ramp, dtype='uint8')
#
# ramp_img = np.matlib.repmat(ramp_arr,N,1)
#
# div_into_8 = (ramp/32)
# floor_div = np.floor(div_into_8)
# mult = floor_div*32
# mult = np.uint8(mult)
# step = np.matlib.repmat(mult,N,1)
#
#
# plt.subplot(233)
# plt.plot(ramp_arr)
# plt.title("The First Line")
# plt.subplot(231)
# plt.imshow(ramp_img, cmap='gray', vmin=0, vmax=255)
# plt.title("Ramp Image")
# plt.subplot(232)
# plt.hist(ramp_img.ravel(), 256, [0, 256])
# plt.title("Ramp Historgram")
# plt.subplot(234)
# plt.imshow(step, cmap='gray', vmin=0, vmax=255)
# plt.title("8 Bands Image")
# plt.subplot(235)
# plt.hist(step, 8, [0, 256])
# plt.title("8 Bands Histogram")
# plt.subplot(236)
# plt.plot(ramp,step[1,:])
# plt.title("First Line")
# plt.show()

############################################################################################################
# # ex 2.2.1 - 2D Signals
############################################################################################################
#
# # Set the Constants
# N = 255
# Fs = N
# Ts = 1/Fs
# fx = 5
# fy = 10
# pi = 3.14
# tx = np.arange(0 ,1+Ts ,Ts)
# ty = tx.T
#
# X,Y = np.meshgrid(tx, ty)
# # Functions
# func_1 = 50*np.cos(2*pi*fx*X)+127
# func_1 = np.uint8(func_1)
#
# func_2 = 100*np.cos(2*pi*fy*Y)+127
# func_2 = np.uint8(func_2)
#
# func_3 = 127+127*np.cos(2*pi*fx*X+2*pi*fy*Y)
# func_3 = np.uint8(func_3)
#
# func_4 = (np.float32(func_1)+np.float32(func_2)+np.float32(func_3)) / 3
# func_4 = np.uint8(func_4)
#
# plt.subplot(341)
# plt.plot(tx, func_1[1,:])
# plt.subplot(345)
# plt.imshow(func_1, cmap='gray', vmin=0, vmax=255)
# # plt.subplot(349)
# # plt.imshow(func_1 ...
# plt.subplot(342)
# plt.plot(ty, func_2[:,1])
# plt.subplot(346)
# plt.imshow(func_2, cmap='gray', vmin=0, vmax=255)
# # plt.subplot(3410)
# # plt.imshow(func2 ...
# plt.subplot(343)
# plt.plot(tx, func_3[1,:])
# plt.subplot(347)
# plt.imshow(func_3, cmap='gray', vmin=0, vmax=255)
# # plt.subplot(3411)
# # plt.imshow(func3 ...
# plt.subplot(344)
# plt.plot(np.arange(0,N+1,1), func_4[:,:])
# plt.subplot(348)
# plt.imshow(func_4, cmap='gray', vmin=0, vmax=255)
# # plt.subplot(3412)
# # plt.imshow(func4 ...
# plt.show()

############################################################################################################
# # ex 2.2.2 - Campbell - Robson
############################################################################################################

# # Constants
# N = 256
# C = 127
# f_max = 25
# f_min = 5
# amp_max = 255
# amp_min = 1
#
# log_f = np.linspace(np.log(f_min),np.log(f_max),N)
# log_a = np.linspace(np.log(amp_min),np.log(amp_max),N)
#
# freq = np.exp(log_f)
# amp = np.exp(log_a)
#
# func_freq = (C*np.cos(2*np.pi*freq))+C
#
# func_amp = amp.T
#
# func_freq_mat = np.matlib.repmat(func_freq,N,1)
# func_amp_mat  = np.matlib.repmat(func_amp,N,1)
# func_amp_mat_T = func_amp_mat.T
#
# func_cmp_rob = ((func_amp_mat_T * func_freq_mat)/ N) +C
#
# plt.subplot(231)
# plt.imshow(func_freq_mat, cmap='gray', vmin=0, vmax=255)
#
# plt.subplot(232)
# plt.imshow(func_amp_mat_T, cmap='gray', vmin=0, vmax=255)
#
# plt.subplot(233)
# plt.imshow(func_cmp_rob, cmap='gray', vmin=0, vmax=255)
#
# plt.subplot(234)
# plt.plot(freq)
#
# plt.subplot(235)
# plt.plot(amp)
# plt.show()
#
# ## TODO - Need to add graph titles and be checked by Joanna
#
############################################################################################################
# # ex 2.2.2 - Campbell - Robson
############################################################################################################