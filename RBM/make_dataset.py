# coding=utf-8
import cv2
import numpy as np

# 利用3*3的滑块将图像做成npy格式的数据集

img1_path = '/media/files/yp/rbm/1506.png'
img2_path = '/media/files/yp/rbm/1606.png'
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
# cv2.imshow('img', img)
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# ret1, thresh1 = cv2.threshold(img1_gray, 127, 255, cv2.THRESH_BINARY)
# ret2, thresh2 = cv2.threshold(img2_gray, 127, 255, cv2.THRESH_BINARY)
height, width = np.shape(img1_gray)
height2, width2 = np.shape(img2_gray)
kernel_width = 5  # 定义滑块的边长
t = kernel_width - 1

if (height == height2) and (width == width2):
    img1_expand = np.zeros([height + t, width + t])
    img2_expand = np.zeros([height + t, width + t])
    for i in range(height):
        for j in range(width):
            img1_expand[i + t / 2][j + t / 2] = img1_gray[i][j]/255.0
            img2_expand[i + t / 2][j + t / 2] = img2_gray[i][j]/255.0
    content = []
    for i in range(t / 2, height + t / 2):
        for j in range(t / 2, width + t / 2):
            content.append(
                [img1_expand[i - 2][j - 2], img1_expand[i - 2][j - 1], img1_expand[i - 2][j], img1_expand[i - 2][j + 1],
                 img1_expand[i - 2][j + 2],
                 img1_expand[i - 1][j - 2], img1_expand[i - 1][j - 1], img1_expand[i - 1][j], img1_expand[i - 1][j + 1],
                 img1_expand[i - 1][j + 2],
                 img1_expand[i][j - 2], img1_expand[i][j - 1], img1_expand[i][j], img1_expand[i][j + 1],
                 img1_expand[i][j + 2],
                 img1_expand[i + 1][j - 2], img1_expand[i + 1][j - 1], img1_expand[i + 1][j], img1_expand[i + 1][j + 1],
                 img1_expand[i + 1][j + 2],
                 img1_expand[i + 2][j - 2], img1_expand[i + 2][j - 1], img1_expand[i + 2][j], img1_expand[i + 2][j + 1],
                 img1_expand[i + 2][j + 2],
                 img2_expand[i - 2][j - 2], img2_expand[i - 2][j - 1], img2_expand[i - 2][j], img2_expand[i - 2][j + 1],
                 img2_expand[i - 2][j + 2],
                 img2_expand[i - 1][j - 2], img2_expand[i - 1][j - 1], img2_expand[i - 1][j], img2_expand[i - 1][j + 1],
                 img2_expand[i - 1][j + 2],
                 img2_expand[i][j - 2], img2_expand[i][j - 1], img2_expand[i][j], img2_expand[i][j + 1],
                 img2_expand[i][j + 2],
                 img2_expand[i + 1][j - 2], img2_expand[i + 1][j - 1], img2_expand[i + 1][j], img2_expand[i + 1][j + 1],
                 img2_expand[i + 1][j + 2],
                 img2_expand[i + 2][j - 2], img2_expand[i + 2][j - 1], img2_expand[i + 2][j], img2_expand[i + 2][j + 1],
                 img2_expand[i + 2][j + 2]
                 ])
    train_lenth = int(len(content) * 0.7)
    np.save("/media/files/yp/rbm/train06.npy", content[:train_lenth])
    np.save("/media/files/yp/rbm/valid06.npy", content[train_lenth:])
    np.save("/media/files/yp/rbm/dataset06.npy", content)

    label = np.load("/media/files/yp/rbm/label06.npy")
    train_label = label[:train_lenth]
    valid_label = label[train_lenth:]
    np.save("/media/files/yp/rbm/train_label06.npy", train_label)
    np.save("/media/files/yp/rbm/valid_label06.npy", valid_label)

    print 'Dataset has generated successfully'
else:
    print 'Can not match images size'
