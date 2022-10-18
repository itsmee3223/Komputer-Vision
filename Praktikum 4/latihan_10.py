import cv2
import math
import numpy as np

img = cv2.imread("../images/color_ball_small.jpg", cv2.IMREAD_GRAYSCALE)
img2 = img.copy()

kernel = np.array([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]], np.single)
hs = np.floor(kernel.shape[0] / 2 ).astype(np.uint32)

for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        tmp = 0
        for ky in range(kernel.shape[0]):
            for kx in range(kernel.shape[0]):
                py = min(max(y + ky - hs, 0), img.shape[0] - 1)
                px = min(max(x + kx - hs, 0), img.shape[0] - 1)
                tmp = tmp + img[py,px] * kernel[ky, kx]

        img2[y,x] = np.floor(tmp).astype(np.uint8)

img3 = cv2.blur(img, (3, 3), cv2.BORDER_DEFAULT)

cv2.imshow("img", img)
cv2.imshow("Filtered", img2)
cv2.imshow("Refrence Filtered", img3)

cv2.waitKey()
cv2.destroyAllWindows()