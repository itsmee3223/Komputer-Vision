import cv2
import numpy as np

img = cv2.imread("../images/kotak_kotak.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

dst = cv2.dilate(dst, None)
corner_img = img.copy()

corner_img[dst > 0.01 * dst.max()] = [0, 0, 255]

cv2.imshow("Original", img)
cv2.imshow("Detected Corner", corner_img)

cv2.waitKey(0)
cv2.destroyAllWindows()