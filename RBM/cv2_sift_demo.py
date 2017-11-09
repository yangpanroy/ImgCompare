# -*- coding:utf-8 -*-  
__author__ = 'Microcosm'

import cv2
import numpy as np
import tifffile as tiff

img = cv2.imread("E:/image_compare/yp/16/1601.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT()
# kp 是所有128特征描述子的集合  
kp = sift.detect(gray, None)
print len(kp)

# 找到后可以计算关键点的描述符  
Kp, res = sift.compute(gray, kp)
print Kp  # 特征点的描述符
print res  # 是特征点个数*128维的矩阵

# 还可以用下面的函数直接检测并返回特征描述符  
kp2, res1 = sift.detectAndCompute(gray, None)
print "******************************"
print res1

img = cv2.drawKeypoints(img, kp, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imwrite("C:/Users/qq619/Desktop/1601features.png", img)

cv2.imshow("sift", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
