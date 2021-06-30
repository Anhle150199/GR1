import cv2 as cv
import numpy as np

img = cv.imread("../Photos/nha1.png")
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edges = cv.Canny(imgGray, 50, 150, apertureSize=3)

# use HoughLines
lines = cv.HoughLines(edges, 1, np.pi/180, 200)
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 -1000*(a))
    cv.line(img, (x1,y1), (x2,y2), (0,0,255), 2)

#use HoughLinesP
lines = cv.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=200, maxLineGap=20)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(img, (x1, y1), (x2, y2), (0,0,255), 3)
cv.imshow("output", img)
cv.waitKey(0)
cv.destroyAllWindows()


# blurred = cv.GaussianBlur(imgGray, (11,11), 0)
# # HoughCircles
# circles = cv.HoughCircles(blurred, cv.HOUGH_GRADIENT, 1, 100, param1=100, param2=90, minRadius=0, maxRadius=200)
#
# if circles is not None:
#     circles = np.round(circles[0, :]).astype("int")
#     for(x,y,r) in circles:
#         cv.circle(img, (x,y), r, (0,0,255), 3)
#         cv.circle(img, (x-2, y-2), (x+2, y+2), (0,0,255), -1)
#
# cv.imshow("img", img)
# cv.waitKey()
