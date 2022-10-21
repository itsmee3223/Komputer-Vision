import numpy as np
import cv2
import matplotlib.pyplot as plt

# hitung akumulator hough
def hough_lines_acc(img, rho_resolution=1, theta_resolution=1):
    height, width = img.shape
    img_diagonal = np.ceil(np.sqrt(height**2 + width**2))
    rhos = np.arange(-img_diagonal, img_diagonal + 1, rho_resolution)
    thetas = np.deg2rad(np.arange(-90, 90, theta_resolution))

    # insialisasi hough accumulator sesuai ukuran rho dan theta
    H = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(img) # cari tepian (nilai selain 0)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for j in range(len(thetas)): # untuk semua theta dan rho
            rho = int((x * np.cos(thetas[j]) + y * np.sin(thetas[j]) + img_diagonal ))

            H[rho, j] += 1

    return H, rhos, thetas


# deteksi puncak
def hough_simple_peaks(H, num_peaks):
    indicies = np.argpartition(H.flatten(), -2)[-num_peaks:]

    return np.vstack(np.unravel_index(indicies, H.shape)).T


# deteksi puncak dengan cara yang lebih biak
def hough_peaks(H, num_peaks, thershold=0, nhood_size=3):
    indices = []
    H1 = np.copy(H)

    for i in range(num_peaks):
        idx = np.argmax(H1)
        H1_idx = np.unravel_index(idx, H1.shape)
        indices.append(H1_idx)

        idx_y, idx_x = H1_idx

        if (idx_x - (nhood_size / 2)) < 0: min_x = 0
        else: min_x = idx_x - (nhood_size / 2)
        if ((idx_x + (nhood_size / 2) + 1) > H.shape[0]) : max_x = H.shape[0]
        else: max_x = idx_x + (nhood_size / 2) + 1

        if (idx_y - (nhood_size / 2)) < 0: min_y = 0
        else: min_y = idx_y - (nhood_size / 2)
        if ((idx_y + (nhood_size / 2) + 1) > H.shape[0]) : max_y = H.shape[0]
        else: max_y = idx_y + (nhood_size / 2) + 1

        for x in range(int(min_x), int(max_x)):
            for y in range(int(min_y), int(max_y)):
                H1[y, x] = 0
                
                if (x == min_x or y == (max_x - 1)):
                    H[y, x] = 255

                if(y == min_y or y == (max_y - 1)):
                    H[y, x] = 255
    
    return indices, H


# plot (tampilkan) akumulator hough
def plot_hough_acc(H, plot_tittle="Hough Accumulator Plot"):
    fig = plt.figure(figsize=(10, 10))
    fig.canvas.set_window_title(plot_tittle)

    plt.imshow(H, cmap='jet')
    plt.xlabel("Theta Direction"), plt.ylabel("Rho Direction")
    plt.tight_layout()
    plt.show()

# deteksi garis berdasarkan hough
def hough_lines_draw(img, indicies, rhos, thetas):
    for i in range(len(indicies)):
        rho = rhos[indicies[i][0]]
        theta = thetas[indicies[i][1]]

        a = np.cos(theta)
        b = np.sin(theta)

        x0 = a*rho
        y0 = b*rho

        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


img = cv2.imread("../images/kotak_kotak.jpg")
img2 = img.copy()

img_grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_blurred = cv2.GaussianBlur(img_grayscale, (5, 5), 1.5)
canny_edges = cv2.Canny(img_blurred, 100, 200)

cv2.imshow("Original Image", img)
cv2.imshow("Canny Edges", canny_edges)
H, rhos, thetas = hough_lines_acc(canny_edges)
indicies, H = hough_peaks(H, 3, nhood_size=11)

hough_lines_draw(img2, indicies, rhos, thetas)
cv2.imshow("Garis Terdeteksi", img2)

plot_hough_acc(H)

cv2.waitKey(0)
cv2.destroyAllWindows()