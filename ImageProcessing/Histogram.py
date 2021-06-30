import cv2
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('../Photos/toy.png',0)

hist, bins = np.histogram(img.flatten(), 256, [0, 256])

cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()

plt.plot(cdf_normalized, color = 'b')
plt.hist(img.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])

plt.legend(('cdf','histogram'), loc = 'upper left')
# plt.imshow(img, cmap='gray')

equ = cv.equalizeHist(img)
res = np.hstack((img,equ))
cv2.imshow("equalize Hist", res)
plt.show()
