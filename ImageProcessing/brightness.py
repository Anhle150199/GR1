import cv2
import numpy as np

# Gamma transformation
def gamma(image, gamma = 2):
  lookUpTable = np.empty((1, 256), np.uint8)
  for i in range(256):
    lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
  image = cv2.LUT(image, lookUpTable)
  return image

# Log Transformation
def log(img):
  img_log = (np.log(img + 1) / (np.log(1 + np.max(img)))) * 255
  img_log = np.array(img_log, dtype=np.uint8)
  return img_log
img = cv2.imread("../Photos/toy.png",0)

img = log(img)

cv2.imshow("gama", img)
cv2.waitKey(0)