# coding=utf-8
import cv2
import numpy as np

# 将标签二值图像制作为npy格式的数据集

img_path = '/media/files/yp/rbm/label03.png'
img = cv2.imread(img_path)
# cv2.imshow('img', img)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, thresh1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
# # cv2.imshow('binary', thresh1)
# cv2.imwrite('/media/files/yp/rbm/label02_binary.jpg', thresh1)
label = []
# height, width = np.shape(thresh1)
height, width = np.shape(img_gray)
for i in range(height):
    for j in range(width):
        if img_gray[i][j] == 0:
            label.append(0)  # 没变化标签
        else:
            label.append(1)  # 变化了标签
np.save("/media/files/yp/rbm/theano/label03.npy", label)
