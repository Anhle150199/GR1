import cv2
import numpy as np

step = 0
image = cv2.imread("../Photos/coins.jpg")
# image = cv2.resize(image, None, fx=0.5, fy=0.5)
cv2.imshow("image1", image)

# find contour
imgBlur = cv2.GaussianBlur(image, (5, 5), 0)
imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

_, thresh = cv2. threshold(imgGray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

thresh = cv2.dilate(thresh, None, iterations=3)
thresh = cv2.erode(thresh, None, iterations=3)
background = cv2.dilate(thresh, None, iterations=3)
cv2.imshow("bg", background)

distMap = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
cv2.normalize(distMap, distMap, 0.0, 255.0, cv2.NORM_MINMAX)
distMap = np.uint8(distMap)
cv2.imshow("distMap", distMap)

foreground = cv2.threshold(distMap, 100, 255, cv2.THRESH_BINARY)[1]
foreground = cv2.erode(foreground, None, 2)
cv2.imshow("foreground", foreground)

unknowZones = cv2.subtract(background, foreground)
cv2.imshow("unknowZones", unknowZones)
ret, markers = cv2.connectedComponents(foreground, connectivity=8, ltype=cv2.CV_32S)
markers = markers+1
markers[unknowZones == 255] = 0
markers = cv2.watershed(image, markers)

print(markers)
cnts = []
for m in np.unique(markers):
    if m < 2:
        continue
    mask = np.zeros(markers.shape, dtype="uint8")
    mask[markers == m] = 255

    c = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts.extend(c)

for (i, c) in enumerate(cnts):
    ((x, y), r) = cv2.minEnclosingCircle(c)
    cv2.putText(image, "#{}".format(i+1), (int(x) - 10, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 111), 2)
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
