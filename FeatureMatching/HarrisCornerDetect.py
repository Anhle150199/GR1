import cv2
import numpy as np

def detect_corner(img, blockSize=2, ksize=3, k=0.04, threshold=0.1):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, blockSize, ksize, k)

    dst = cv2.dilate(dst, None)
    img[dst > threshold * dst.max()] = [255, 0, 0]
    return img


img = cv2.imread("../Photos/chess.png")
out_path = detect_corner(img, blockSize=2, ksize=5, k=0.04, threshold=0.15)

cv2.imshow('dst', out_path)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
