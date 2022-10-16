import numpy as np
import cv2 as cv

img = cv.imread("../images/color_ball.jpg")
cv.imshow("Image Original", img)

for y in range(img.shape[0]):
  for x in range(img.shape[1]):
    if img[y,x,0] > 180 and (img[y,x,1] > 90 and img[y,x,1] < 150 ) and (img[y,x,2] < 20):
      img[y,x] = img[y,x]
    else:
      img[y,x] = [0,0,0]

cv.imshow("Image Filtered", img)

while(True):
    if cv.waitKey(1) & 0xFF == ord('q'):
        break