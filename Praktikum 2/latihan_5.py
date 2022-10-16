import numpy as np
import cv2 as cv

img = cv.imread("../images/color_ball.jpg")
cv.imshow("Image Original", img)
img = cv.cvtColor(img, cv.COLOR_BGR2HSV)


for y in range(img.shape[0]):
  for x in range(img.shape[1]):
    if img[y,x,0] > 90 and img[y,x,0] < 120 and img[y,x,1] > 40:
      img[y,x] = img[y,x]
    else:
      img[y,x] = [0,0,0]

img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
cv.imshow("Image Filtered", img)

while(True):
    if cv.waitKey(1) & 0xFF == ord('q'):
        break