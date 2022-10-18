import numpy as np
import cv2
import math

def bilinear_interpolation(image, y, x):
  height = image.shape[0]
  width = image.shape[1]

  x1 = max(min(math.floor(x), width - 1), 0)
  y1 = max(min(math.floor(y), height - 1), 0)
  x2 = max(min(math.ceil(x), width - 1), 0)
  y2 = max(min(math.ceil(y), height - 1), 0)

  a = image[y1, x1].astype(float)
  b = image[y2, x1].astype(float)
  c = image[y1, x2].astype(float)
  d = image[y2, x2].astype(float)

  dx = x - x1
  dy = y - y1

  new_pixel = np.zeros((image.shape[2]))
  new_pixel = a * (1 - dx) * (1- dy)
  new_pixel += b * dy * (1 - dx)
  new_pixel += c * dx * (1 - dy)
  new_pixel += d * dx * dy

  return new_pixel

def resize(image, new_height, new_width):
  new_image = np.zeros((new_height, new_width, img.shape[2]), image.dtype)

  orig_height = image.shape[0]
  orig_width = image.shape[1]

  # compute center column and center row
  x_orig_center = (orig_width - 1) / 2
  y_orig_center = (orig_height - 1) / 2

  # compute center column and center row
  x_scale_center = (new_width - 1) / 2
  y_scale_center = (new_height - 1) / 2

  # compute the scale in both axes
  scale_x = orig_width / new_width
  scale_y = orig_height / new_height

  for y in range(new_height):
    for x in range(new_width):
      x_ = (x - x_scale_center) * scale_x + x_orig_center
      y_ = (y - y_scale_center) * scale_y + y_orig_center

      new_image[y, x] = bilinear_interpolation(image, y_, x_)

  return  new_image


img = cv2.imread("../images/color_ball_small.jpg")

ratio = 2

new_width = math.floor(img.shape[1] * ratio)
new_height = math.floor(img.shape[0] * ratio)

resized_img = resize(img, new_height, new_width)

# refrence for testing
refrence_resized_img = cv2.resize(img, (new_width, new_height), interpolation = cv2.INTER_LINEAR)

cv2.imshow("img", img)
cv2.imshow("resized_img", resized_img)
cv2.imshow("refrence_resized_img", refrence_resized_img)

cv2.waitKey()
cv2.destroyAllWindows()