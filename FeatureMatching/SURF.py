import cv2

img1 = cv2.imread("/content/chocopie1.png")
img2 = cv2.imread("/content/chocopie4.png")

surf = cv2.SURF_create()
kp1, des1 = surf.detectAndCompute(img1, None)
kp2, des2 = surf.detectAndCompute(img2, None)

bf = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=False)
matches = bf.match(des1, des2)

matches = sorted(matches, key=lambda x: x.distance)
reeses_matches = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None, flags=2)

cv2_imshow(reeses_matches)