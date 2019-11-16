import numpy as np
import numpy.matlib
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from random import gauss
import cv2
from scipy.stats import uniform
import scipy

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


def contrast(img, value):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    s = s * value
    lim = 255
    s[s > lim] = 255

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


############################################################
# ex. 1.1.2 - Adding Brightness Only to Tire.tif:
##########################################################

# I = cv2.imread("C:/Program Files/MATLAB/R2017b/toolbox/images/imdata/tire.tif")
# plt.subplot(2, 3, 1)
# plt.imshow(I)
# plt.subplot(2, 3, 4)
# plt.hist(I.ravel(), 256, [0, 256])
# plt.title('Histogram for Tire.tif original image')
#
# II = brightness(I, value=50)
# plt.subplot(2, 3, 2)
# plt.imshow(II)
# plt.subplot(2, 3, 5)
# plt.hist(II.ravel(), 256, [0, 256])
# plt.title('Histogram for Tire.tif added brightness')
#
#
# III = II - 50
# plt.subplot(2, 3, 3)
# plt.imshow(III)
# plt.subplot(2, 3, 6)
# plt.hist(III.ravel(), 256, [0, 256])
# plt.title('Histogram for Tire.tif substracted brightness')
#
# plt.show()


################################################
# #ex 1.1.3 - Adding Contrast to Tire.tif:
################################################
#
# I = cv2.imread("C:/Program Files/MATLAB/R2017b/toolbox/images/imdata/tire.tif")
# plt.subplot(2, 3, 1)
# plt.imshow(I)
# histI = cv2.calcHist(I, [0], None, [256], [0, 256])
# plt.subplot(2, 3, 4)
# plt.hist(I.ravel(), 256, [0, 256])
# plt.title('Histogram for Tire.tif original image')
#
# #II = I * 0.4
# II = contrast(I, value=0.4)
# plt.subplot(2, 3, 2)
# plt.imshow(II)
# plt.subplot(2, 3, 5)
# plt.hist(II.ravel(), 256, [0, 256])
# plt.title('Histogram for Tire.tif added brightness')
#
# III = II / 0.4
# plt.subplot(2, 3, 3)
# plt.imshow(III)
# plt.subplot(2, 3, 6)
# plt.hist(III.ravel(), 256, [0, 256])
# plt.title('Histogram for Tire.tif substracted brightness')
#
#
# plt.show()

#######################################
# 1.2 - random signal pdf, PDF, CDF
#######################################

#Creating a random uniform discrete signal
# uni_sig = np.random.randint(0, 400, 1000)
# plt.subplot(2,3,1)
# plt.plot(uni_sig)
# plt.title('Uniform Random Signal')
# #histogram of the random signal
# plt.subplot(2,3,2)
# hist_sig, bins, __ = plt.hist(uni_sig, color='green', bins=50)
# plt.title('Histogram for Uniform Random Signal')
#
# #computing the pdf for the uniform signal:
#
# integ = sum(hist_sig[0][4:7]*np.diff(hist_sig[1][4:8]))
#
#
# # dist = uniform(loc=0, scale=100)
# # pdf1 = scipy.stats.norm.pdf(uni_sig)
# # cdf1 = scipy.stats.norm.cdf(uni_sig)
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
# hist_gauss_sig = plt.hist(gauss_sig, color='green', bins=50)
# plt.title('Histogram for Gaussian Random Signal')
#
# #computing the pdf for the Gaussian signal:
# dist2 = uniform(loc=0, scale=100)
# pdf2 = dist2.pdf(gauss_sig)
# cdf2 = dist2.cdf(gauss_sig)
# plt.subplot(2,3,6)
# plt.plot(cdf2)
# plt.show()


#############################################################
#1.3 - Transformations
#############################################################
L = 1000
x = np.linspace(0, 255, L)
y = 0.4 * x + 50
y_fft = scipy.fft(y)

plt.subplot(1,2,1)
plt.plot(y)
plt.subplot(1,2,2)
plt.plot(y_fft)

plt.show()