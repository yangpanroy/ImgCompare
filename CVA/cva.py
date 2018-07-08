# coding=utf-8
import cv2
import histprocessing as hp
import numpy as np
from skimage import img_as_ubyte
from sklearn import preprocessing


def histeq(refImg, oriImg):
    outImg1 = hp.histMatching(oriImg[:, :, 0], refImg[:, :, 0])
    outImg2 = hp.histMatching(oriImg[:, :, 1], refImg[:, :, 1])
    outImg3 = hp.histMatching(oriImg[:, :, 2], refImg[:, :, 2])
    outImg = cv2.merge([outImg1, outImg2, outImg3])
    return outImg


def normalize(dis_map):
    m = np.mean(dis_map)
    mx = max(dis_map)
    mn = min(dis_map)
    return [(float(i) - m) / (mx - mn) for i in dis_map]


img1 = cv2.imread("F:/13qu/15/2015gf213.TIF")
img2 = cv2.imread("F:/13qu/17/RGB/201702gf213.TIF")
img1_pipei = histeq(img1, img2)
# 防止尺寸不匹配
row1, col1, dim1 = img1.shape
row2, col2, dim2 = img2.shape
if row1 > row2:
    det = row1 - row2
    img1 = img1[det:, :, :]
elif row1 < row2:
    det = row2 - row1
    img2 = img2[det:, :, :]
if col1 > col2:
    det = col1 - col2
    img1 = img1[:, :-det, :]
elif col1 < col2:
    det = col2 - col1
    img2 = img2[:, :-det, :]

row, col, dim = img1.shape
dis_map = np.zeros((row, col))
for i in range(row):
    for j in range(col):
        dis_map[i][j] = pow(pow(abs(int(img1_pipei[i][j][0]) - int(img2[i][j][0])), 2) + pow(abs(int(img1_pipei[i][j][0]) - int(img2[i][j][0])), 2) + pow(abs(int(img1_pipei[i][j][0]) - int(img2[i][j][0])), 2), 0.5)
# dis_map = normalize(dis_map)
min_max_scaler = preprocessing.MinMaxScaler()
dis_map = min_max_scaler.fit_transform(dis_map)
dis_map = img_as_ubyte(dis_map)
ret, th = cv2.threshold(dis_map, 0, 255, cv2.THRESH_OTSU)
print ret
cv2.imwrite("F:/13qu/201702gf213_cva.jpg", th)
