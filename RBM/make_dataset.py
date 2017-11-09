# coding=utf-8
import cv2
import numpy as np

# 利用3*3的滑块将图像做成npy格式的数据集

img1_path = '/media/files/yp/rbm/1503.png'
img2_path = '/media/files/yp/rbm/1603.png'
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
# cv2.imshow('img', img)
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
height, width = np.shape(img1_gray)
height2, width2 = np.shape(img2_gray)
kernel_width = 3  # 定义滑块的边长
t = kernel_width - 1

if (height == height2) and (width == width2):
    img1_expand = np.zeros([height + t, width + t])
    img2_expand = np.zeros([height + t, width + t])
    for i in range(height):
        for j in range(width):
            img1_expand[i + t / 2][j + t / 2] = img1_gray[i][j]
            img2_expand[i + t / 2][j + t / 2] = img2_gray[i][j]
    content = []
    for i in range(t / 2, height + t / 2):
        for j in range(t / 2, width + t / 2):
            content.append([img1_expand[i - 1][j - 1], img1_expand[i - 1][j], img1_expand[i - 1][j + 1],
                            img1_expand[i][j - 1], img1_expand[i][j], img1_expand[i][j + 1],
                            img1_expand[i + 1][j - 1], img1_expand[i + 1][j], img1_expand[i + 1][j + 1],
                            img2_expand[i - 1][j - 1], img2_expand[i - 1][j], img2_expand[i - 1][j + 1],
                            img2_expand[i][j - 1], img2_expand[i][j], img2_expand[i][j + 1],
                            img2_expand[i + 1][j - 1], img2_expand[i + 1][j], img2_expand[i + 1][j + 1]
                            ])
    np.save("/media/files/yp/rbm/dataset03.npy", content)
    print 'Dataset has generated successfully'
else:
    print 'Can not match images size'
