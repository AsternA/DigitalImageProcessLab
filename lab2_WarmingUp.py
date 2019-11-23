import numpy as np
import numpy.matlib
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from random import gauss
import cv2
from scipy.stats import uniform
from scipy.stats import norm

import scipy
N = 256
##########################################################
##########################################################
#function for 1.1.2 - increasing brightness in Python
##########################################################
def brightness(img, value):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img



##########################################################
##########################################################


def brightness2(img, value):

    img2 = np.ones(np.shape(img[:, :]))
    cols = np.size(img2[1, :])
    rows = np.size(img2[:, 1])

    for i in range(0, cols, 1):
        for j in range(0, rows, 1):
            if (img[j, i] + value) > 255:
                img2[j, i] = 255
            else:
                img2[j, i] = img[j, i] + value
    return img2
##########################################################
##########################################################


def contrast(img, value):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    s = s * value
    lim = 255
    s[s > lim] = 255

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

##########################################################
##########################################################

def contrast2(img, value):

    img2 = np.ones(np.shape(img[:, :]))
    cols = np.size(img2[1, :])
    rows = np.size(img2[:, 1])

    for i in range(0, cols, 1):
        for j in range(0, rows, 1):
            if (img[j, i] * value) > 255:
                img2[j, i] = 255
            else:
                img2[j, i] = img[j, i] * value
    return img2
##########################################################
##########################################################

##########################################################
##########################################################

def negative(img):

    img2 = np.zeros(np.shape(img[:, :]))
    img2[:, :] = 255 - img[:, :]

    return img2
##########################################################
##########################################################
def create_histogram(img):
    """Create Histogram for Image"""
    rows, cols = img.shape
    hist = np.zeros(N)
    for i in range(0, 255, 1):
        for col in range(0, cols, 1):
            for row in range(0, rows, 1):
                if img[row,col] == i:
                    hist[i] += 1
    return hist


############################################################
############################################################
def histogram_equalization(img):
    """Equalize Histogram for Image"""
    rows, cols = img.shape
    hist = np.zeros(N)
    img2 = np.ceil((img) * 255)
    for i in range(0, 255, 1):
        for col in range(0, cols, 1):
            for row in range(0, rows, 1):
                if img2[row, col] == i:
                    hist[i] += 1
    return hist


############################################################
############################################################

def hist_stretch(img, t_func):
    """Stretch Histogram for any Image"""
    rows, cols = img.shape
    img_c = np.ones(np.shape(img[:, :]))
    for col in range(0, cols, 1):
        for row in range(0, rows, 1):
            img_c[row, col] = t_func[img[row, col]]
    return img_c
# ############################################################
# # ex. 1.1.2 - Adding Brightness Only to Tire.tif:
# ##########################################################


# I = plt.imread('/Users/almogstern/Desktop/Matlab Pictures/tire.tif')
# plt.subplot(3, 2, 1)
# plt.imshow(np.uint8(I), cmap='gray', vmin=0, vmax=255)
# plt.subplot(3, 2, 2)
# plt.hist(I.ravel(), 256, [0, 256])
# plt.title('Histogram for Tire.tif original image')
#
# II = brightness2(I, value=100)
# plt.subplot(3, 2, 3)
# plt.imshow(II, cmap='gray', vmin=0, vmax=255)
# plt.subplot(3, 2, 4)
# plt.hist(II.ravel(), 256, [0, 256])
# plt.title('Added brightness')
#
#
# III = II - 100
# plt.subplot(3, 2, 5)
# plt.imshow(III, cmap='gray', vmin=0, vmax=255)
# plt.subplot(3, 2, 6)
# plt.hist(III.ravel(), 256, [0, 256])
# plt.title('Subtracted brightness')
#
# plt.show()


# ################################################
# # #ex 1.1.3 - Adding Contrast to Tire.tif:
# ################################################
# #
# # I = cv2.imread("C:/Program Files/MATLAB/R2017b/toolbox/images/imdata/tire.tif")
# # plt.subplot(2, 3, 1)
# # plt.imshow(I)
# # histI = cv2.calcHist(I, [0], None, [256], [0, 256])
# # plt.subplot(2, 3, 4)
# # plt.hist(I.ravel(), 256, [0, 256])
# # plt.title('Histogram for Tire.tif original image')
# #
# # #II = I * 0.4
# # II = contrast(I, value=0.4)
# # plt.subplot(2, 3, 2)
# # plt.imshow(II)
# # plt.subplot(2, 3, 5)
# # plt.hist(II.ravel(), 256, [0, 256])
# # plt.title('Histogram for Tire.tif added brightness')
# #
# # III = II / 0.4
# # plt.subplot(2, 3, 3)
# # plt.imshow(III)
# # plt.subplot(2, 3, 6)
# # plt.hist(III.ravel(), 256, [0, 256])
# # plt.title('Histogram for Tire.tif substracted brightness')
# #
# #
# # plt.show()
#
# #######################################
# # 1.2 - random signal pdf, PDF, CDF
# #######################################
#
# # Creating a random uniform discrete signal
# uni_sig = np.random.randint(0, 255, 1000)
# plt.subplot(2,3,1)
# plt.plot(uni_sig)
# plt.title('Uniform Random Signal')
# #histogram of the random signal
# plt.subplot(2,3,2)
# hist_sig, bins, __ = plt.hist(uni_sig, color='green', bins=256)
# plt.title('Histogram for Uniform Random Signal')
#
# #computing the pdf for the uniform signal:
#
# #integ = sum(hist_sig[0][4:7]*np.diff(hist_sig[1][4:8]))
#
# sz = np.size(uni_sig)
# pdf1 = hist_sig / float(sz)
# cdf1 = np.cumsum(pdf1)
#
# plt.subplot(243)
# plt.plot(pdf1)
# plt.subplot(244)
# plt.plot(cdf1)
# #plt.plot(uni_sig, norm.pdf(uni_sig))
# dist = uniform(loc=0, scale=100)
# pdf1 = scipy.stats.norm.pdf(uni_sig)
# cdf1 = scipy.stats.norm.cdf(uni_sig)
# plt.subplot(2,3,3)
# plt.plot(integ)
#
# #Gaussian random signal:
# gauss_sig = [gauss(0.0, 5.0) for i in range(1000)]
# plt.subplot(2,3,4)
# plt.plot(gauss_sig)
# plt.title('Gaussian Random Signal')
# #histogram of the gaussian random signal:
# plt.subplot(2,3,5)
# hist_gauss_sig = plt.hist(gauss_sig, color='green', bins=256)
# plt.title('Histogram for Gaussian Random Signal')
#
# #computing the pdf for the Gaussian signal:
# dist2 = uniform(loc=0, scale=100)
# pdf2 = dist2.pdf(gauss_sig)
# cdf2 = dist2.cdf(gauss_sig)
# plt.subplot(2,3,6)
# plt.plot(cdf2)
# pdf = dist2.pdf(uni_sig)
# plt.subplot(2,3,3)
# plt.plot(pdf)
# plt.show()

#
# #############################################################
# #1.3 - Transformations
# #############################################################
# # L = 1000
# # x = np.linspace(0, 255, L)
# # y = 0.4 * x + 50
# # y_fft = scipy.fft(y)
# #
# # plt.subplot(1,2,1)
# # plt.plot(y)
# # plt.subplot(1,2,2)
# # plt.plot(y_fft)
# #
# # plt.show()
# #
# ################
# ## ALMOG BELOW #
# ################
#
# # # Transformations:
# # N = 256
# # n = np.linspace(0, N, N)
# # f = 0.05
# # A = 2
# # func = A*np.cos(2*np.pi*f*n)
# #
# # # Transformation of Func:
# # func_t = 7*func+20
# #
# # plt.subplot(311)
# # plt.plot(func)
# #
# # plt.subplot(312)
# # plt.plot(func_t)
# #
# # # Transformation Graph:
# # trans_grp = 7 * n + 20
# #
# # plt.subplot(313)
# # plt.plot(trans_grp)
# #
# # plt.show()
# #
#
# #########################################
# ## 2.1.1 Which is also 3.1
# #########################################
#
# N = 256
# n = np.linspace(0,N,N)
#
# img = plt.imread('/Users/almogstern/Desktop/Matlab Pictures/pout.tif')
# #hist_img = plt.hist(img, bins=N)
#
# # Change Brightness
# img_bright = brightness2(img, value=50)
# img_bright = np.uint8(img_bright)
# img_bright_t = np.zeros(np.size(n))
# for i in range(0,N):
#     if n[i] + 50 >= 255:
#         img_bright_t[i] = 255
#     else:
#         img_bright_t[i] = n[i] + 50
#
# # Change Contrast
# img_contrast = contrast2(img, value=0.4)
# img_contrast = np.uint8(img_contrast)
# img_contrast_t = n * 0.4
#
# # Negative
# img_negative = negative(img)
# img_negative = np.uint8(img_negative)
# img_negative_t = 256 - n
#
# # Original Image
# plt.subplot(3, 4, 1)
# plt.imshow(img, cmap='gray', vmin=0, vmax=255)
# plt.title('Original Img')
# plt.subplot(3, 4, 5)
# plt.hist(img.ravel(), 256, [0, 256])
# plt.title('Original Hist')
# plt.subplot(3, 4, 9)
# plt.plot(n,n)
# plt.title('Original Trans')
#
# # Image with Different Brightness
# plt.subplot(3, 4, 2)
# plt.imshow(img_bright, cmap='gray', vmin=0, vmax=255)
# plt.title('Bright Img')
# plt.subplot(3, 4, 6)
# plt.hist(img_bright.ravel(), 256, [0, 256])
# plt.title('Bright Hist')
# plt.subplot(3, 4 ,10)
# plt.plot(n, np.uint8(img_bright_t))
# plt.title('Bright Trans')
#
#
# # Image Multiplied by Value
# plt.subplot(3, 4, 3)
# plt.imshow(img_contrast, cmap='gray', vmin=0, vmax=255)
# plt.title('Contrast Img')
# plt.subplot(3, 4, 7)
# plt.hist(img_contrast.ravel(), 256, [0, 256])
# plt.title('Contrast Hist')
# plt.subplot(3, 4, 11)
# plt.plot(n, np.uint8(img_contrast_t))
# plt.title('Contrast Trans')
#
# # Negative Image
# plt.subplot(3, 4, 4)
# plt.imshow(img_negative, cmap='gray', vmin=0, vmax=255)
# plt.title('Neg Image')
# plt.subplot(3, 4, 8)
# plt.hist(img_negative.ravel(), 256, [0, 256])
# plt.title('Neg Hist')
# plt.subplot(3, 4, 12)
# plt.plot(n, np.uint8(img_negative_t))
# plt.title('Neg Trans')
#
# # Show the Images
# plt.show()
#
# #
# #########################################
# ## 2.1.2 Which is also 3.2
# #########################################
# # Creating your own Histogram
# N = 256
# # Create linspace
# n = np.linspace(0,N,N)
# # Loading the Image
# img = plt.imread('/Users/almogstern/Desktop/Matlab Pictures/pout.tif')
# # Calculating Histogram
# manual_hist = create_histogram(img) # Calculation time is longer than usual
# # PDF and CDF of Manual Histogram
# manual_pdf = manual_hist / sum(manual_hist)
#
# manual_cdf = np.cumsum(manual_pdf)
#
# # Plots
# plt.subplot(3, 4, (1, 4))
# plt.plot(manual_hist)
# plt.title('Manual Histogram and Histogram Command')
# plt.subplot(3, 4, (1, 4))
# plt.hist(img.ravel(), 256, [0, 256])
# #plt.title('plt.hist')
# plt.subplot(3, 4, (5, 8))
# plt.plot(manual_pdf)
# plt.title('PDF')
# plt.subplot(3, 4, (9, 12))
# plt.plot(manual_cdf)
# plt.title('CDF')
# plt.show()

# #########################################
# ## 3.3 - Threshold
# #########################################
# # Reading the Image
# img_cv = cv2.imread('/Users/almogstern/Desktop/Matlab Pictures/rice.png', cv2.CV_8UC1)
# img = plt.imread('/Users/almogstern/Desktop/Matlab Pictures/rice.png')
# img = img * 255
# rows, cols = img.shape
# # Setting the Threshold
# thr = 110
# thr_img = np.uint8(img > thr) * 255
#
# # Average Columns
# mcols = np.arange(cols)
# meancols = np.mean(img, axis=1, out=mcols)
#
# # Average Rows
# mrows = np.arange(rows)
# meanrows = np.mean(img.T, axis=1, out=mrows)
#
# # Adaptive Threshold
# ret, th3 = cv2.threshold(img_cv, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#
# # Plotting
# plt.subplot(2, 3, 1)
# plt.imshow(img, cmap='gray', vmin=0, vmax=255)
# plt.title('Original Image')
# plt.subplot(2, 3, 2)
# plt.imshow(thr_img, cmap='gray', vmin=0, vmax=255)
# plt.title('Single Threshold at 110')
# plt.subplot(2, 3, 3)
# plt.plot(mcols)
# plt.title('Colums Average')
# plt.subplot(2, 3, 4)
# plt.hist(img.ravel(), 256, [0, 256])
# plt.title('Img Histogram')
# plt.subplot(2, 3, 5)
# plt.plot(mrows)
# plt.title('Rows Average')
# plt.subplot(2, 3, 6)
# plt.imshow(th3, 'gray')
# plt.title('Adaptive Threshold')
# plt.show()

# #########################################
# ## 3.3.1 - Histogram Stretching
# #########################################
# # Constants
# N = 256
# num_slopes = 3
# len_points = 4
# # Loading the Image
# img = plt.imread('/Users/almogstern/Desktop/Matlab Pictures/pout.tif')
# # Creating the Transfer function
# # User input for custom transfer function
# point1_x = int(input("Enter x1: "))
# point1_y = int(input("Enter y1: "))
# point2_x = int(input("Enter x2: "))
# point2_y = int(input("Enter y2: "))
# # Function
# x = np.arange(len_points)
# y = np.arange(len_points)
# x[0] = 0
# x[1] = point1_x
# x[2] = point2_x
# x[3] = 255
# y[0] = 0
# y[1] = point1_y
# y[2] = point2_y
# y[3] = 255
#
# # Creating the Slops
# slope = np.zeros(num_slopes)
# slope[0] = y[1] / x[1]
# slope[1] = ((y[2] - y[1])/(x[2] - x[1]))
# slope[2] = ((y[3] - y[2])/(x[3] - x[2]))
#
# # Graph
# trans_func = np.zeros(N, dtype=float)
# for indx in range(0, 256,1):
#     if(indx<=x[1]):
#         trans_func[indx] = slope[0] * (indx)
#     elif indx <=x[2]:
#         trans_func[indx] = (slope[1] * ((indx) - x[1])) + y[1]
#     else:
#         trans_func[indx] = (slope[2] * ((indx) - x[2])) + y[2]
#
# # function
# after_t = hist_stretch(img, trans_func)
#
# # Plotting
# plt.subplot(3, 2, 1)
# plt.imshow(img, cmap='gray', vmin=0, vmax=255)
# plt.title('Original Image')
# plt.subplot(3, 2, 2)
# plt.hist(img.ravel(), 256, [0, 256])
# plt.title('Original Histogram')
# plt.subplot(3, 2, 3)
# plt.imshow(after_t, cmap='gray', vmin=0, vmax=255)
# plt.title('Img After Transform')
# plt.subplot(3, 2, 4)
# plt.hist(after_t.ravel(), 256, [0, 256])
# plt.title('Histogram After Strech')
# plt.subplot(3, 2, (5, 6))
# plt.plot(trans_func)
# plt.show()

# #########################################
# ## 3.3.2 - Histogram Equalization
# #########################################
# N = 256
# # Create linspace
# n = np.linspace(0,N,N)
# # Loading the Image
# img = plt.imread('/Users/almogstern/Desktop/Matlab Pictures/pout.tif')
# # Calculating Histogram
# manual_hist = create_histogram(img) # Calculation time is longer than usual
# # PDF and CDF of Manual Histogram
# manual_pdf = manual_hist / sum(manual_hist)
# manual_cdf = np.cumsum(manual_pdf)
#
# # Equalizing the Histogram
# img_cdf_after_t = hist_stretch(img, manual_cdf)
# hist_eq = histogram_equalization(img_cdf_after_t)
#
# # CDF of new equalized Histogram
# cdf_eq = np.cumsum(hist_eq)
#
# # Plots
# plt.subplot(3, 2, 1)
# plt.imshow(img, cmap='gray', vmin=0, vmax=255)
# plt.title('Original Image')
# plt.subplot(3, 2, 3)
# plt.hist(img.ravel(), 256, [0, 256])
# plt.title('Original Histogram')
# plt.subplot(3, 2, 5)
# plt.plot(manual_cdf)
# plt.title('Original CDF')
# plt.subplot(3, 2, 2)
# plt.imshow((img_cdf_after_t*255), cmap='gray', vmin=0, vmax=255)
# plt.title('Img After Hist Eq')
# plt.subplot(3, 2, 4)
# plt.hist((img_cdf_after_t*255).ravel(), 256, [0, 256])
# plt.title('Hist after Eq')
# plt.subplot(3, 2, 6)
# plt.plot(cdf_eq)
# plt.title('Equalized CDF')
# plt.show()
# #########################################
# ## 3.3.2 - Histogram based on CDF of pictures
# #########################################
# Load the Image
img_1 = plt.imread('/Users/almogstern/Desktop/Matlab Pictures/tire.tif')
img_2 = plt.imread('/Users/almogstern/Desktop/Matlab Pictures/cameraman.tif')
# Calculate the CDF of the Pictures
hist_img_1 = create_histogram(img_1)
pdf_img_1 = hist_img_1 / sum(hist_img_1)
cdf_img_1 = np.cumsum(pdf_img_1)

hist_img_2 = create_histogram(img_2)
pdf_img_2 = hist_img_2 / sum(hist_img_2)
cdf_img_2 = np.cumsum(pdf_img_2)

#
img_1_t = hist_stretch(img_1, cdf_img_2)    # New Image
img_1_t = np.uint8(img_1_t * 255)
# Create new CDF
hist_img_1_t = create_histogram(img_1_t)
pdf_img1_t = hist_img_1_t / sum(hist_img_1_t)
cdf_img1_t = np.cumsum(pdf_img1_t)

img_2_t = hist_stretch(img_2, cdf_img_1)    # New Image
img_2_t = np.uint8(img_2_t * 255)
# Create new CDF
hist_img_2_t = create_histogram(img_2_t)
pdf_img2_t = hist_img_2_t / sum(hist_img_2_t)
cdf_img2_t = np.cumsum(pdf_img2_t)


plt.subplot(2, 4, 1)
plt.imshow(img_1, cmap='gray', vmin=0, vmax=255)
plt.title('Original Image')
plt.subplot(2, 4, 2)
plt.plot(cdf_img_1)
plt.title('CDF Img 1')
plt.subplot(2, 4, 3)
plt.imshow((img_1_t*255), cmap='gray', vmin=0, vmax=255)
plt.title('New Image')
plt.subplot(2, 4, 4)
plt.plot(cdf_img1_t)
plt.title('New CDF')
plt.subplot(2, 4, 5)
plt.imshow(img_2, cmap='gray', vmin=0, vmax=255)
plt.title('Original Image')
plt.subplot(2, 4, 6)
plt.plot(cdf_img_2)
plt.title('CDF Img 2')
plt.subplot(2, 4, 7)
plt.imshow((img_2_t*255), cmap='gray', vmin=0, vmax=255)
plt.title('New Image')
plt.subplot(2, 4, 8)
plt.plot(cdf_img2_t)
plt.title('New CDF')
plt.show()

