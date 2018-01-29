# coding=utf-8
import cv2
import numpy as np

# 利用7*7的滑块将图像做成npy格式的数据集

img1_path = '/media/files/yp/rbm/15/1502.png'
img2_path = '/media/files/yp/rbm/16/1602.png'
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
# 是否分割标签 TODO 执行代码前更改 FLAG 和 路径
SPLIT_LABEL = 0
label_path = "/media/files/yp/rbm/label/binary/label11.npy"
train_label_path = "/media/files/yp/rbm/label/binary/train_11.npy"
valid_label_path = "/media/files/yp/rbm/label/binary/valid_11.npy"
# 是否分割数据 TODO 执行代码前更改 FLAG 和 路径
SPLIT_DATA = 0
for dim in range(3):
    # 分离通道 0：B 1：G 2：R
    img1_gray = cv2.split(img1)[dim]
    img2_gray = cv2.split(img2)[dim]

    height, width = np.shape(img1_gray)
    height2, width2 = np.shape(img2_gray)
    kernel_width = 7  # 定义滑块的边长
    t = kernel_width - 1

    if (height == height2) and (width == width2):
        img1_expand = np.zeros([height + t, width + t])
        img2_expand = np.zeros([height + t, width + t])
        for i in range(height):
            for j in range(width):
                img1_expand[i + t / 2][j + t / 2] = img1_gray[i][j] / 255.0
                img2_expand[i + t / 2][j + t / 2] = img2_gray[i][j] / 255.0
        content = []
        for i in range(t / 2, height + t / 2):
            for j in range(t / 2, width + t / 2):
                content.append(
                    [
                        img1_expand[i - 3][j - 3], img1_expand[i - 3][j - 2], img1_expand[i - 3][j - 1], img1_expand[i - 3][j], img1_expand[i - 3][j + 1], img1_expand[i - 3][j + 2], img1_expand[i - 3][j + 3],
                        img1_expand[i - 2][j - 3], img1_expand[i - 2][j - 2], img1_expand[i - 2][j - 1], img1_expand[i - 2][j], img1_expand[i - 2][j + 1], img1_expand[i - 2][j + 2], img1_expand[i - 2][j + 3],
                        img1_expand[i - 1][j - 3], img1_expand[i - 1][j - 2], img1_expand[i - 1][j - 1], img1_expand[i - 1][j], img1_expand[i - 1][j + 1], img1_expand[i - 1][j + 2], img1_expand[i - 1][j + 3],
                        img1_expand[i][j - 3], img1_expand[i][j - 2], img1_expand[i][j - 1], img1_expand[i][j], img1_expand[i][j + 1], img1_expand[i][j + 2], img1_expand[i][j + 3],
                        img1_expand[i + 1][j - 3], img1_expand[i + 1][j - 2], img1_expand[i + 1][j - 1], img1_expand[i + 1][j], img1_expand[i + 1][j + 1], img1_expand[i + 1][j + 2], img1_expand[i + 1][j + 3],
                        img1_expand[i + 2][j - 3], img1_expand[i + 2][j - 2], img1_expand[i + 2][j - 1], img1_expand[i + 2][j], img1_expand[i + 2][j + 1], img1_expand[i + 2][j + 2], img1_expand[i + 2][j + 3],
                        img1_expand[i + 3][j - 3], img1_expand[i + 3][j - 2], img1_expand[i + 3][j - 1], img1_expand[i + 3][j], img1_expand[i + 3][j + 1], img1_expand[i + 3][j + 2], img1_expand[i + 3][j + 3],
                        img2_expand[i - 3][j - 3], img2_expand[i - 3][j - 2], img2_expand[i - 3][j - 1], img2_expand[i - 3][j], img2_expand[i - 3][j + 1], img2_expand[i - 3][j + 2], img2_expand[i - 3][j + 3],
                        img2_expand[i - 2][j - 3], img2_expand[i - 2][j - 2], img2_expand[i - 2][j - 1], img2_expand[i - 2][j], img2_expand[i - 2][j + 1], img2_expand[i - 2][j + 2], img2_expand[i - 2][j + 3],
                        img2_expand[i - 1][j - 3], img2_expand[i - 1][j - 2], img2_expand[i - 1][j - 1], img2_expand[i - 1][j], img2_expand[i - 1][j + 1], img2_expand[i - 1][j + 2], img2_expand[i - 1][j + 3],
                        img2_expand[i][j - 3], img2_expand[i][j - 2], img2_expand[i][j - 1], img2_expand[i][j], img2_expand[i][j + 1], img2_expand[i][j + 2], img2_expand[i][j + 3],
                        img2_expand[i + 1][j - 3], img2_expand[i + 1][j - 2], img2_expand[i + 1][j - 1], img2_expand[i + 1][j], img2_expand[i + 1][j + 1], img2_expand[i + 1][j + 2], img2_expand[i + 1][j + 3],
                        img2_expand[i + 2][j - 3], img2_expand[i + 2][j - 2], img2_expand[i + 2][j - 1], img2_expand[i + 2][j], img2_expand[i + 2][j + 1], img2_expand[i + 2][j + 2], img2_expand[i + 2][j + 3],
                        img2_expand[i + 3][j - 3], img2_expand[i + 3][j - 2], img2_expand[i + 3][j - 1], img2_expand[i + 3][j], img2_expand[i + 3][j + 1], img2_expand[i + 3][j + 2], img2_expand[i + 3][j + 3]
                    ])
        train_lenth = int(len(content) * 0.7)

        if dim == 0:
            dim_name = 'B'
        if dim == 1:
            dim_name = 'G'
        if dim == 2:
            dim_name = 'R'

        if SPLIT_DATA == 1:
            np.save("/media/files/yp/rbm/dataset/train11_7x7_" + dim_name + ".npy", content[:train_lenth])
            np.save("/media/files/yp/rbm/dataset/valid11_7x7_" + dim_name + ".npy", content[train_lenth:])
        else:
            np.save("/media/files/yp/rbm/dataset/test02_7x7_" + dim_name + ".npy", content)

        if SPLIT_LABEL == 1:
            label = np.load(label_path)
            train_label = label[:train_lenth]
            valid_label = label[train_lenth:]
            np.save(train_label_path, train_label)
            np.save(valid_label_path, valid_label)
            SPLIT_LABEL = 0

        print 'Dataset has generated successfully'
    else:
        print 'Can not match images size'
