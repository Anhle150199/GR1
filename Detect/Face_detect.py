import numpy as np
import cv2
from matplotlib import pyplot as plt


def faceDetect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier("../haarcascades/haarcascade_frontalface_alt.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    i = 0
    for x, y, w, h in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
        i += 1
    print("The face's count is detected: ", i)
    return img


img = cv2.imread("../Photos/face2.jpg")
img = cv2.resize(img, None, fx=0.5, fy=0.5)
img = faceDetect(img)
plt.imshow(img)
plt.show()
cv2.waitKey(0)
