import numpy as np
import numpy.matlib
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from random import gauss
import cv2

# ex. 1.1.2 - Adding Brightness Only to Tire.tif:

I = plt.imread("C:/Program Files/MATLAB/R2017b/toolbox/images/imdata/tire.tif")
plt.subplot(2, 3, 1)
plt.imshow(I)
#histI = cv2.calcHist(I, [0], None, [256], [0, 256])
plt.subplot(2, 3, 4)
plt.hist(I.ravel(), 256, [0, 256])
plt.title('Histogram for Tire.tif original image')




II = I + 50

plt.subplot(2, 3, 2)
plt.imshow(II)
#histII = cv2.calcHist(II, [0], None, [256], [0, 256])
plt.subplot(2, 3, 5)
plt.hist(II.ravel(), 256, [0, 256])
plt.title('Histogram for Tire.tif added brightness')



III = II - 50
plt.subplot(2, 3, 3)
plt.imshow(III)
#histIII = cv2.calcHist(III, [0], None, [256], [0, 256])
plt.subplot(2, 3, 6)
plt.hist(III.ravel(), 256, [0, 256])
plt.title('Histogram for Tire.tif substracted brightness')


plt.show()
# cv2.imshow("Substracted Brightness", III)
# cv2.waitKey(0)

##Show all three images horizontally for optical comparison
# all_images = cv2.hconcat((I, II, III))
# #cv2.imwrite('allImages.jpg', all_images)
# cv2.imshow("all three images", all_images)
#
# cv2.waitKey(0)

#ex 1.1.3 - Adding Contrast to Tire.tif:


I = cv2.imread("C:/Program Files/MATLAB/R2017b/toolbox/images/imdata/tire.tif")
plt.subplot(2, 3, 1)
plt.imshow(I)
histI = cv2.calcHist(I, [0], None, [256], [0, 256])
plt.subplot(2, 3, 4)
plt.hist(I.ravel(), 256, [0, 256])
plt.title('Histogram for Tire.tif original image')

II = I * 0.4
plt.subplot(2, 3, 2)
plt.imshow(II)
plt.subplot(2, 3, 5)
plt.hist(II.ravel(), 256, [0, 256])
plt.title('Histogram for Tire.tif added brightness')

III = II / 0.4
plt.subplot(2, 3, 3)
plt.imshow(III)
plt.subplot(2, 3, 6)
plt.hist(III.ravel(), 256, [0, 256])
plt.title('Histogram for Tire.tif substracted brightness')


plt.show()