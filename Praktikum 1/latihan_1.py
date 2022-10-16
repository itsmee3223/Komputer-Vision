import numpy as np
import cv2

img = cv2.imread("../images/color_ball.jpg")
cv2.imshow("Image", img)

print(type(img)) #type dari img yang sudah diread
print(img.shape)

while(True):
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break