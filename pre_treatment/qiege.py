# coding=utf-8
import tifffile as tiff
import cv2

img_rows = 1000
img_cols = 1000
mask = cv2.imread("D:/qiefen/2017/yc0.TIF")

for i in range(0, 8):

    for j in range(0, 5):
        tupian = mask[i * img_rows: (i + 1) * img_rows, j * img_cols:  (j + 1) * img_cols, :]

        cv2.imwrite("D:/pic_div/2017/{}.tif".format((i) * 5 + j), tupian)
