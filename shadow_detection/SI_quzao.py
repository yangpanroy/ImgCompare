# coding=utf-8
import datetime
from sklearn.decomposition import PCA
import cv2
import numpy as np
from skimage import img_as_ubyte


def rgbtohsi(rgb_lwpImg):
    rows = int(rgb_lwpImg.shape[0])
    cols = int(rgb_lwpImg.shape[1])
    b, g, r = cv2.split(rgb_lwpImg)
    # 归一化到[0,1]
    b = b / 255.0
    g = g / 255.0
    r = r / 255.0
    hsi_lwpImg = rgb_lwpImg.copy()
    H, S, I = cv2.split(hsi_lwpImg)
    for i in range(rows):
        for j in range(cols):
            num = 0.5 * ((r[i, j] - g[i, j]) + (r[i, j] - b[i, j]))
            den = np.sqrt((r[i, j] - g[i, j]) ** 2 + (r[i, j] - b[i, j]) * (g[i, j] - b[i, j]))
            theta = float(np.arccos(num / den))

            if den == 0:
                H = 0
            elif b[i, j] <= g[i, j]:
                H = theta
            else:
                H = 2 * 3.14169265 - theta

            min_RGB = min(min(b[i, j], g[i, j]), r[i, j])
            sum = b[i, j] + g[i, j] + r[i, j]
            if sum == 0:
                S = 0
            else:
                S = 1 - 3 * min_RGB / sum

            H = H / (2 * 3.14159265)
            I = sum / 3.0
            # 输出HSI图像，扩充到255以方便显示，一般H分量在[0,2pi]之间，S和I在[0,1]之间
            hsi_lwpImg[i, j, 0] = H * 255
            hsi_lwpImg[i, j, 1] = S * 255
            hsi_lwpImg[i, j, 2] = I * 255
    return hsi_lwpImg


def get_shadow_img(path):
    img = cv2.imread(path)
    rows, cols, dim = img.shape
    img2 = np.reshape(img, (rows * cols, 3))
    pca = PCA(n_components=1)
    newData = pca.fit_transform(img2)

    for i in range(rows * cols):
        if newData[i, 0] > 0:
            newData[i, 0] = 0
    minData = np.min(newData)
    norData = newData / minData
    newImg = np.reshape(norData, (rows, cols))

    hsi_lwpImg = rgbtohsi(img)
    hsi_lwpImg = hsi_lwpImg / 255.0  # 归一化

    SI = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            SI[i, j] = (newImg[i, j] - hsi_lwpImg[i, j, 2]) * (1 + hsi_lwpImg[i, j, 1]) / (
                    newImg[i, j] + hsi_lwpImg[i, j, 2] + hsi_lwpImg[i, j, 1])
    SI = img_as_ubyte(SI)
    ret, th = cv2.threshold(SI, 0, 255, cv2.THRESH_OTSU)
    print ret
    return th


start = datetime.datetime.now()
img = cv2.imread("C:/Users/qq619/Desktop/21123.jpg", 0)
shadow1 = get_shadow_img("D:/yinchuanyingxiang/2015fenge/pipei/2015gf21123.TIF")
shadow2 = get_shadow_img("D:/yinchuanyingxiang/2016fenge/RGB/2016gf21123.TIF")
rows, cols = img.shape
shadow_change = np.zeros((rows, cols))
for i in range(rows):
    for j in range(cols):
        if shadow1[i, j] != shadow2[i, j]:
            shadow_change[i, j] = 255
            if img[i, j] != 0:
                img[i, j] = 0

# cv2.imshow("shadow1", shadow1)
# cv2.imshow("shadow2", shadow2)
# cv2.imshow("shadow_change", shadow_change)
# cv2.imshow("img - shadow_change", img)
# cv2.waitKey()
# cv2.destroyAllWindows()
cv2.imwrite("C:/Users/qq619/Desktop/shadow_change.jpg", shadow_change)
cv2.imwrite("C:/Users/qq619/Desktop/img-shadow_change.jpg", img)

result = cv2.medianBlur(img, 9)
cv2.imwrite("C:/Users/qq619/Desktop/result.jpg", result)
end = datetime.datetime.now()
print (end-start).seconds
