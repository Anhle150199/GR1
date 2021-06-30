import cv2
import numpy as np

img = cv2.imread("../Photos/nha1.png")
img = cv2.resize(img,dsize=None, fx=0.8, fy=0.8)
gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create() #create sift

kp, des = sift.detectAndCompute(gimg, None)
print(des.shape)
# kp = sift.detect(gimg,None)

img= cv2.drawKeypoints(gimg, kp, img,None)


cv2.imshow("put", img)
cv2.waitKey(0)