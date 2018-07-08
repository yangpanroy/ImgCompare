# coding=utf-8
from sklearn.decomposition import PCA
import cv2
import numpy as np
from skimage import img_as_ubyte
import os
import sys
import tifffile as tiff


def view_bar(num, total):
    rate = num * 1.0 / total
    rate_num = int(rate * 100)
    r = '\r[%s%s]%d%%' % ("*" * rate_num, "-" * (100 - rate_num), rate_num,)
    sys.stdout.write(r)
    sys.stdout.flush()


def get_vegetation_img(img_path):
    img2 = tiff.imread(img_path)
    print img2.shape
    Red = img2[:, :, 0]
    Blue = img2[:, :, 2]
    NIR = img2[:, :, 3]
    Red = Red / 255.0
    Blue = Blue / 255.0
    NIR = NIR / 255.0
    NDVI = ((NIR - Red) / (NIR + Red))
    NDVI = img_as_ubyte(NDVI)
    # ret, th = cv2.threshold(NDVI, 0, 255, cv2.THRESH_OTSU)
    ret, th = cv2.threshold(NDVI, 127, 255, cv2.THRESH_BINARY)
    print ret
    return th


predictions_dir = "C:/Users/qq619/Desktop/predictions/1701-1702/img-shadow_change/"
label1_dir = "D:/yinchuanyingxiang/201701fenge/"
label2_dir = "D:/yinchuanyingxiang/201702fenge/"
f_list1 = os.listdir(predictions_dir)
file_num = 0

for file_name in f_list1:
    if os.path.splitext(file_name)[1] == '.jpg':
        file_id = os.path.splitext(file_name)[0].split("gf")[1]
        img = cv2.imread(predictions_dir + file_name, 0)
        img1_path = label1_dir + "201701gf" + file_id + '.TIF'
        img2_path = label2_dir + "201702gf" + file_id + '.TIF'
        vegetation1 = get_vegetation_img(img1_path)
        vegetation2 = get_vegetation_img(img2_path)
        rows, cols = img.shape
        vegetation_change = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                if vegetation1[i, j] != vegetation2[i, j]:
                    vegetation_change[i, j] = 255
                    if img[i, j] != 0 and (vegetation1[i, j] == 255 or vegetation2[i, j] == 255):
                        img[i, j] = 0
        cv2.imwrite(predictions_dir + "img-vegetation_change/" + file_name, img)
        result = cv2.medianBlur(img, 9)
        cv2.imwrite(predictions_dir + 'img-vegetation_change/quzao/' + file_name, result)
        file_num = file_num + 1
        view_bar(file_num, len(f_list1))
