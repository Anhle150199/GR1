import cv2
import imutils
import numpy as np
import  matplotlib.pyplot as plt

step = 0
image = cv2.imread("../Photos/objects.png")
image = cv2.resize(image, None, fx=0.7, fy=0.7)
img = image

img = cv2.medianBlur(img, 3)
img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
cv2.imshow("HSV", img)

img_gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, width, channels = img.shape
cv2.imshow("1", img_gray1)

thresh = cv2.threshold(img_gray1, 100, 250, cv2.THRESH_BINARY_INV)[1]
cv2.imshow("thresh1", thresh)
thresh = cv2.dilate(thresh, None, iterations=1)
thresh = cv2.medianBlur(thresh, 9)
# thresh = cv2.dilate(thresh, None, iterations=1)
cv2.imshow("thresh2", thresh)

cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

for (i, c) in enumerate(cnts):
    ((x, y), r) = cv2.minEnclosingCircle(c)
    cv2.putText(image, "#{}".format(i+1), (int(x) - 10, int(y)+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    step = i+1
cv2.drawContours(image, cnts, -1, (0, 0, 255), 2)
cv2.putText(image, "this image has #{} aircraft F16".format(step), (0,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
# plt.imshow(img_gray1)

cv2.imshow("img", image)
plt.show()
cv2.waitKey(0)