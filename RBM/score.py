# coding=utf-8
import cv2
import numpy as np


# 给出参考图和预测图进行评价
# 评价的指标有以下几个：
# TP：变化像素被正确检测的数目；
# FP：未变化像素被误检成为变化像素的数目（虚警数）；
# TN：未变化像素被正确检测的数目；
# FN：变化像素被检测为未变化像素的数目（漏检数）；


def score(label, prediction):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    sum = 0
    kappa = 0
    PCC = 0
    row1, col1 = label.shape
    row2, col2 = prediction.shape
    if row1*col1 != row2*col2:
        print "标签与预测图像尺寸不匹配！\n\n"
    else:
        for i in xrange(0, row1):
            for j in xrange(0, col1):
                if label[i][j] == 0:
                    if prediction[i][j] == 0:
                        TN = TN + 1
                    else:
                        FP = FP + 1
                else:
                    if prediction[i][j] == 0:
                        FN = FN + 1
                    else:
                        TP = TP + 1
        sum = row1 * col1
        pa = (TP + TN) / (sum * 1.0)
        pe = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / (sum * sum * 1.0)
        kappa = (pa - pe) / (1 - pe)
        PCC = (TP + TN) / (sum * 1.0)
    return sum, TP, FP, TN, FN, PCC, kappa


if __name__ == '__main__':
    label = cv2.imread("D:/pic_div/label/0.jpg", 0)
    prediction = cv2.imread("D:/pic_div/label/0.png", 0)
    sum, TP, FP, TN, FN, PCC, kappa = score(label, prediction)
    print "总像素的数目：{0}".format(sum)
    print "变化像素被正确检测的数目：{0}   正检率：{1}".format(TP, TP*1.0/(TP+FN))
    print "未变化像素被误检成为变化像素的数目（虚警数）：{0}".format(FP)
    print "未变化像素被正确检测的数目：{0}".format(TN)
    print "变化像素被检测为未变化像素的数目（漏检数）：{0}   漏检率：{1}".format(FN, FN*1.0/(TP+FN))
    print "百分比修正（准确数占比）：{0}".format(PCC)
    print "Kappa系数：{0}".format(kappa)
