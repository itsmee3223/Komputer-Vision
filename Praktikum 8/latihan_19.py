from sys import flags
import cv2
import numpy as np

def wrapImages(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0,0], [0, rows1], [cols1, rows1], [cols1, 0]]).reshape(-1,1,2)
    temp_points = np.float32([[0,0], [0,rows2], [cols2, rows2], [cols2, 0]]).reshape(-1,1,2)

    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)
    
    list_of_points = np.concatenate((list_of_points_1, list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.05)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.05)

    translation_dist = [-x_min, -y_min]

    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max - x_min, y_max - y_min))
    output_img[translation_dist[1]:rows1 + translation_dist[1], translation_dist[0]:cols1 + translation_dist[0]] = img1

    return output_img


img1 = cv2.imread("../images/p1.jpg")
img_gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread("../images/p2.jpg")
img_gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

detect = cv2.ORB_create()
kp1, des1 = detect.detectAndCompute(img_gray1, None)
kp2, des2 = detect.detectAndCompute(img_gray2, None)

cv2.imshow("img_1 keypoints", cv2.drawKeypoints(img1, kp1, None))
cv2.imshow("img_2 keypoints", cv2.drawKeypoints(img2, kp2, None))

match = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
matches = match.match(des1, des2)
matches = sorted(matches, key=lambda x:x.distance)
good = matches[:20]

draw_params = dict(matchColor=(0,255,0), singlePointColor=None, flags=2)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:30],None,**draw_params)
cv2.imshow("Matches", img3)

MIN_MATCH_COUNT = 10
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h,w = img_gray1.shape
    pts = np.float32([ [0,0], [0,h-1], [w-1,h-2], [w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)
    img_gray2 = cv2.polylines(img_gray1, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    cv2.imshow("Overlap", img_gray2)

    dst = wrapImages(img1, img2, M)
    cv2.imshow("Wraped", dst)
else:
    print("Not enough matches are found - %d%d", (len(good)/MIN_MATCH_COUNT))


cv2.waitKey(0)
cv2.destroyAllWindows()