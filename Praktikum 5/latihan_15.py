import numpy as np
import cv2

img = cv2.imread("../images/kotak_kotak.jpg")

img_grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_blurred = cv2.GaussianBlur(img_grayscale, (5, 5), 1.5)
canny_edges = cv2.Canny(img_blurred, 100, 200)

lines = cv2.HoughLines(canny_edges, 1, np.pi/180, 100)

for r_theta in lines:
    arr = np.array(r_theta[0], dtype=np.float64)
    r, theta = arr
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*r
    y0 = b*r

    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Edges Image", canny_edges)
cv2.imshow("Result Image", img)

cv2.waitKey()
cv2.destroyAllWindows()