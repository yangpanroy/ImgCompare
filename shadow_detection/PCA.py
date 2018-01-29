# coding=utf-8
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


# img = cv2.imread("C:/Users/qq619/Desktop/2016.jpg")
img = cv2.imread("D:/pic_div/2016/6.png")
rows, cols, dim = img.shape
img2 = np.reshape(img, (rows * cols, 3))
pca = PCA(n_components=1)
newData = pca.fit_transform(img2)

for i in range(rows*cols):
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
        SI[i, j] = (newImg[i, j] - hsi_lwpImg[i, j, 2]) * (1 + hsi_lwpImg[i, j, 1]) / (newImg[i, j] + hsi_lwpImg[i, j, 2] + hsi_lwpImg[i, j, 1])
SI = img_as_ubyte(SI)
ret, th = cv2.threshold(SI, 0, 255, cv2.THRESH_OTSU)
print ret

cv2.imshow('newImg', th)
cv2.waitKey(0)

# img = cv2.imread("D:/pic_div/2015/0.png")
# b, g, r = cv2.split(img)
# # r = r / 255.0
# # g = g / 255.0
# # b = b / 255.0
# cv2.imshow('img_b', b)
# # rows, cols = b.shape
# # nor_b = np.zeros((rows, cols))
# # for i in range(rows):
# #     for j in range(cols):
# #         nor_b[i, j] = b[i, j] / (r[i, j] + g[i, j] + b[i, j])
# # cv2.imshow('nor_b', nor_b)
# #
# # ret1, th1 = cv2.threshold(b, 0, 1, cv2.THRESH_OTSU)
# # ret2, th2 = cv2.threshold(nor_b, 0, 1, cv2.THRESH_OTSU)
# # print ret1
# # print ret2
# # cv2.imshow('B_fenge', th1)
# cv2.imshow('nor_b_fenge', th1)
#
# cv2.waitKey(0)
