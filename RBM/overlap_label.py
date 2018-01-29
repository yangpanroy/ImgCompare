# coding=utf-8
import numpy as np
import cv2

changed_img = cv2.imread("C:/Users/qq619/Desktop/label08_predictions_img4_epoch50.jpg")
fangwu_img = cv2.imread("C:/Users/qq619/Desktop/fangwu1608_predictions_img.jpg")
changed_img = cv2.cvtColor(changed_img, cv2.COLOR_BGR2GRAY)
fangwu_img = cv2.cvtColor(fangwu_img, cv2.COLOR_BGR2GRAY)
height, width = np.shape(changed_img)
height2, width2 = np.shape(fangwu_img)
result = np.zeros([height, width])
if (height == height2) and (width == width2):
    for i in range(height):
        for j in range(width):
            if (changed_img[i, j] >= 127) and (fangwu_img[i, j] >= 127):
                result[i, j] = 255
    # cv2.imshow("变化的房屋", result)
    cv2.imwrite("C:/Users/qq619/Desktop/changed_fangwu.jpg", result)
    print 'Overlap label successfully'

    # 保存全彩变化图查看变化位置
    # img = cv2.imread("E:/image_compare/yp/1608.png")
    # for i in range(height):
    #     for j in range(width):
    #         if result[i, j] != 255:
    #             img[i, j, 0] = img[i, j, 0] / 2
    #             img[i, j, 1] = img[i, j, 1] / 2
    #             img[i, j, 2] = img[i, j, 2] / 2
    # cv2.imwrite("C:/Users/qq619/Desktop/changed_fangwu_rgb.jpg", img)
else:
    print 'Can not match images size'
