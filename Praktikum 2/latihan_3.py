import numpy as np
import cv2 as cv

img = cv.imread("../images/color_ball.jpg")
cv.imshow("gambar", img)
# baca basic informasi
print(type(img)) #type dari img yang sudah diread
print(img.shape)
img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
#get pixel in BGR format
print(img[133,237,:])
print(img[188,466,:])
print(img[343,160,:])

while(True):
    if cv.waitKey(1) & 0xFF == ord('q'):
        break