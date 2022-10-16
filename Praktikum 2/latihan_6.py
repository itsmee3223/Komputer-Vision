import numpy as np
import cv2

# define a video capture object
vid = cv2.VideoCapture(0)
params = cv2.SimpleBlobDetector_Params()

params.minThreshold = 100
params.maxThreshold = 255

# filter by area
params.filterByArea = True
params.minArea = 500
params.maxArea = 250000

# filter by circularity
params.filterByCircularity = True
params.minCircularity = 0.4

# filter by convexity
params.filterByConvexity = True
params.minConvexity = 0.3

# filter by inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

ver = (cv2.__version__).split('.')
if int(ver[0]) < 3:
  detector = cv2.SimpleBlobDetector(params)
else:
  detector = cv2.SimpleBlobDetector_create(params)

lower_limit = np.array([0,50,50])
upper_limit = np.array([20,255,255])

kernel = np.ones((5, 5), np.uint8)
while(True):
  # capture video frame by frame
  _,im = vid.read()
  hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

  # thershold the HSV image to get only blue color
  mask = cv2.inRange(hsv, lower_limit, upper_limit)
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

  # bitwise-AND mask and originial image
  res = cv2.bitwise_and(im, im, mask = mask)

  # cv2.imshow('frame', im)
  # cv2.imshow('mask', mask)
  cv2.imshow('res', res)

  # im3 = cv2.bitwise_not(im3)
  keypoints = detector.detect(cv2.bitwise_not(mask))

  # imcv = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
  im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

  # cv2.imshow("Mask", im3)
  cv2.imshow("Detection", im_with_keypoints)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# after loop release cap object
vid.release()
cv2.destroyAllWindows()