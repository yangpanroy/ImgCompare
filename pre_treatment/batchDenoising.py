import cv2
import os
predDir = "E:/result/UnsupervisedModel/xingqing/15-16/img/"
outDir = "E:/result/UnsupervisedModel/xingqing/15-16/img/quzao/"
fList = os.listdir(predDir)
for fileName in fList:
    filePath = predDir + fileName
    img = cv2.imread(filePath)
    result = cv2.medianBlur(img, 15)
    cv2.imwrite(outDir + fileName, result)
# img = cv2.imread("C:/Users/qq619/Desktop/gf213SDAE.jpg")
# result = cv2.medianBlur(img, 9)
# cv2.imwrite("C:/Users/qq619/Desktop/gf213SDAE-denoise.jpg", result)
