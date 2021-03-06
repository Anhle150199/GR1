import cv2
import numpy as np

image = cv2.imread("../Photos/gao3.png", 0)
dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
dft_shift[227:233, 219:225] = 255
dft_shift[227:233, 236:242] = 255

f_ishift = np.fft.ifftshift(dft_shift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# min, max = np.amin(img_back, (0, 1)), np.amax(img_back, (0, 1))
img = cv2.normalize(img_back, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

cv2.imshow("image", image)
cv2.imshow("img", img)
cv2.waitKey(0)