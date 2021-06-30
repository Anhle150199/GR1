from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt

# reading the image
img = imread("../Detect/object-detector-master/data/dataset/pos/pos (92).jpg")
fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                	cells_per_block=(2, 2), visualize=True, multichannel=True)
print(hog_image)
plt.axis("off")
plt.imshow(hog_image, cmap="gray")
plt.show()