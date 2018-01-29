import cv2
import numpy as np

img = cv2.imread("C:/Users/qq619/Desktop/read_tif.tif", cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
outImg = cv2.convertScaleAbs(img)
# img2 = cv2.imread("D:/pic_div/2016/0.png")
cv2.imwrite("C:/Users/qq619/Desktop/read_tif.png", outImg)
