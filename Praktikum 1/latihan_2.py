import numpy as np
import cv2

img = cv2.imread("../images/color_ball.jpg")

cv2.imshow("Image", img)

# baca basic informasi
print(type(img)) #type dari img yang sudah diread
print(img.shape)

#get pixel in BGR format
print(img[133,237,:])
print(img[188,466,:])
print(img[343,160,:])

while(True):
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break