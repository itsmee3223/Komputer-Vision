import cv2
import math
import numpy as np

img = cv2.imread("../images/color_ball_small.jpg", cv2.IMREAD_GRAYSCALE)
img2 = img.copy()

kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], np.single)
hs = np.floor(kernel.shape[0] / 2 ).astype(np.uint32)

for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        tmp = 0
        for ky in range(kernel.shape[0]):
            for kx in range(kernel.shape[0]):
                py = min(max(y + ky - hs, 0), img.shape[0] - 1)
                px = min(max(x + kx - hs, 0), img.shape[0] - 1)
                tmp = tmp + img[py,px] * kernel[ky, kx]

        if(tmp > 255):
            tmp = 255
        if(tmp < 0):
            tmp = 0

        img2[y,x] = np.floor(tmp).astype(np.uint8)

cv2.imshow("img", img)
cv2.imshow("Filtered", img2)

cv2.waitKey()
cv2.destroyAllWindows()