import cv2

filename = 'Photos/face4.jpg'

hog = cv2.HOGDescriptor()

hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Load an image
image = cv2.imread(filename)
bounding_boxes = hog.detectMultiScale(image,winStride=(4, 4),padding=(8, 8), scale=1.05)[0]

# Draw bounding boxes on the image
for (x, y, w, h) in bounding_boxes:
    cv2.rectangle(image,
          (x, y),
          (x + w, y + h),
          (0, 0, 255),
          4)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

