import cv2
from matplotlib import pyplot as plt

img = cv2.imread("../Photos/chocopie2.png", 0)

# Initiate STAR detector
orb = cv2.ORB_create()
kp, des = orb.detectAndCompute(img, None)
# kp, des = orb.compute(img, kp)
img2 = cv2.drawKeypoints(img,kp,None)
dst_with_size = cv2.drawKeypoints(img, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
dst_without_size = cv2.drawKeypoints(img, kp, None)
#
img2 = cv2.hconcat((dst_with_size, dst_without_size))
cv2.imshow("img", img2)
cv2.waitKey(0)