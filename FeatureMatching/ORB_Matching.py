import cv2
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

# def display(img):
#     fig = plt.figure(figsize=(120, 10))
#     ax = fig.add_subplot(111)
#     ax.imshow(img, cmap= 'gray')
    # cv2.imshow("output", img)

reeses1 = cv2.imread("../Photos/chocopie1.png",0)
reeses2 = cv2.imread("../Photos/chocopie2.png", 0)
# display(reeses)

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(reeses1, None)
kp2, des2 = orb.detectAndCompute(reeses2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
matches = bf.match(des1, des2)
sigle_match = matches[0]
# sigle_match.distance
matches = sorted(matches, key = lambda x:x.distance)
reeses_matches = cv2.drawMatches(reeses1, kp1, reeses2, kp2, matches[:50], None, flags=2)
# display(reeses_matches)
# plt.show()
cv2.imshow("xxx",reeses_matches)
cv2.waitKey(0)
