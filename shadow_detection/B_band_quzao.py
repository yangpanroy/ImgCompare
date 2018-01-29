import cv2
import numpy as np
from skimage import img_as_ubyte


def get_shadow_img(path):
    img = cv2.imread(path)
    b, g, r = cv2.split(img)
    b = b / 255.0
    g = g / 255.0
    r = r / 255.0
    # cv2.imshow("B_band", b)
    rows, cols = b.shape
    nor_b = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            nor_b[i, j] = b[i, j] / (r[i, j] + g[i, j] + b[i, j])
    # nor_b = nor_b * 255.0
    nor_b = img_as_ubyte(nor_b)
    # cv2.imshow("Norm_B_band", nor_b)
    b = cv2.split(img)[0]
    ret1, th1 = cv2.threshold(b, 0, 255, cv2.THRESH_OTSU)
    ret2, th2 = cv2.threshold(nor_b, 0, 255, cv2.THRESH_OTSU)
    print ret1
    print ret2
    # cv2.imshow("threshold_result_B_band", th1)
    # cv2.imshow("threshold_result_Norm_B_band", th2)
    shadow_area = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if th2[i, j] == 255 and th1[i, j] == 0:
                shadow_area[i, j] = 255
    # cv2.imshow("shadow_area", shadow_area)
    # cv2.waitKey(0)
    return shadow_area


img = cv2.imread("C:/Users/qq619/Desktop/change_detection.jpg", 0)
shadow1 = get_shadow_img("C:/Users/qq619/Desktop/2015.jpg")
shadow2 = get_shadow_img("C:/Users/qq619/Desktop/2016.jpg")
quzao = cv2.imread("C:/Users/qq619/Desktop/quzao9.jpg", 0)

rows, cols = img.shape
shadow_change = np.zeros((rows, cols))
for i in range(rows):
    for j in range(cols):
        if shadow1[i, j] != shadow2[i, j]:
            shadow_change[i, j] = 255
            if img[i, j] != 0:
                img[i, j] = 0

cv2.imshow("shadow1", shadow1)
cv2.imshow("shadow2", shadow2)
cv2.imshow("shadow_change", shadow_change)
cv2.imshow("img - shadow_change", img)
cv2.imwrite("C:/Users/qq619/Desktop/shadow_change.jpg", shadow_change)
cv2.imwrite("C:/Users/qq619/Desktop/img-shadow_change.jpg", img)
cv2.waitKey(0)

result = cv2.medianBlur(img, 9)
cv2.imwrite("C:/Users/qq619/Desktop/result.jpg", result)
