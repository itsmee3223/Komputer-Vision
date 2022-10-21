import cv2
import numpy as np

def find_harris_corners(img, k, window_size, thershold):
    corner_list = []
    offseet = int(window_size / 2)
    y_range = img.shape[0] - offseet
    x_range = img.shape[1] - offseet

    dy, dx = np.gradient(img)
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2

    for y in range(offseet, y_range):
        for x in range(offseet, x_range):
            start_y = y - offseet
            end_y = y + offseet + 1
            start_x = x - offseet
            end_x = x + offseet + 1

            windowIxx = Ixx[start_y : end_y, start_x : end_x]
            windowIxy = Ixy[start_y : end_y, start_x : end_x]
            windowIyy = Iyy[start_y : end_y, start_x : end_x]

            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy

            r = det - k*(trace**2)

            if r > thershold:
                corner_list.append([x, y, r])
    
    return corner_list


def draw_corner(img, corner_list):
    for corner in corner_list:
        img[corner[1], corner[0]] = (0, 0, 255)

    return img


k = 0.04
window_size = 5
thershold = 10000000.00
img = cv2.imread("../images/kotak_kotak.jpg")
img_grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_blurred = cv2.GaussianBlur(img_grayscale, (3, 3), 1.2)

corner_list = find_harris_corners(img_blurred, k, window_size, thershold)
corner_img = draw_corner(img.copy(), corner_list)

cv2.imshow("Original", img)
cv2.imshow("Detected Corner", corner_img)

cv2.waitKey(0)
cv2.destroyAllWindows()