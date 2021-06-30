import cv2
import numpy as np

img = cv2.imread("../Photos/avatar2.jpg", 0)
blur = cv2.GaussianBlur(img, (5,5), 0)
canny = cv2.Canny(blur, 100, 200)

cv2.imshow("canny", canny)
cv2.waitKey(0)