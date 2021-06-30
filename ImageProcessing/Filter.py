import cv2
import numpy as np

img = cv2.imread("../Photos/image1.jpg", 0)
cv2.imshow("image ", img)

kernel = np.ones((5, 5), np.float32)/25
dst = cv2.filter2D(img, -1, kernel)
cv2.imshow("Convolution ", dst)

blur = cv2.blur(img,(5, 5))
cv2.imshow("Mean ", blur)

blur = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imshow("Gaussian", blur)

median = cv2.medianBlur(img, 3)
cv2.imshow("Median", median)


cv2.waitKey(0)
