import cv2
import numpy as np
import matplotlib.pyplot as plt

def BF_FeatureMatcher(des1, des2):
    brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    no_of_matches = brute_force.match(des1, des2)

    no_of_matches = sorted(no_of_matches,key=lambda x:x.distance)

    return no_of_matches

def draw_matches(pic1, kpt1, pic2, kpt2, best_match, max_features):
    output_image = cv2.drawMatches(pic1, kpt1, pic2, kpt2, best_match[:max_features], None, flags=2)

    return output_image


if __name__ == '__main__':
    img_path_1 = "../images/monas_1.jpg"
    img_path_2 = "../images/monas_rotate_1.jpg"
    img_path_3 = "../images/monas_rotate_2.jpg"
    img_path_4 = "../images/tugu_jogja.jpg"

    img1 = cv2.imread(img_path_1)
    img2 = cv2.imread(img_path_2)
    img3 = cv2.imread(img_path_3)
    img4 = cv2.imread(img_path_4)

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    gray4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)

    detect = cv2.ORB_create()
    key_point1, descrip1 = detect.detectAndCompute(gray1, None)
    key_point2, descrip2 = detect.detectAndCompute(gray2, None)
    key_point3, descrip3 = detect.detectAndCompute(gray3, None)
    key_point4, descrip4 = detect.detectAndCompute(gray4, None)

    mathces_1 = BF_FeatureMatcher(descrip1, descrip2)
    mathces_2 = BF_FeatureMatcher(descrip1, descrip3)
    mathces_3 = BF_FeatureMatcher(descrip1, descrip4)

    tot_matches_1 = len(mathces_1)
    tot_matches_2 = len(mathces_2)
    tot_matches_3 = len(mathces_3)

    print(f"Features Matches 1 & 2 {tot_matches_1}")
    print(f"Features Matches 1 & 3 {tot_matches_2}")
    print(f"Features Matches 1 & 4 {tot_matches_3}")

    output_1 = draw_matches(gray1, key_point1, gray2, key_point2, mathces_1, 90)
    output_2 = draw_matches(gray1, key_point1, gray3, key_point3, mathces_2, 90)
    output_3 = draw_matches(gray1, key_point1, gray4, key_point4, mathces_3, 90)

    plt.rcParams["figure.autolayout"] = True
    fig, ax = plt.subplots(nrows=3, ncols=1)

    ax[0].axis("off")
    ax[0].imshow(output_1, cmap="gray")
    ax[1].axis("off")
    ax[1].imshow(output_2, cmap="gray")
    ax[2].axis("off")
    ax[2].imshow(output_3, cmap="gray")
    plt.show()