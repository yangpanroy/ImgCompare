# --*-- coding=utf-8 --*--
import cv2
import numpy as np
from matplotlib import pyplot as plt

from find_obj import filter_matches, explore_match


def read_png(path):
    # 读取PNG文件
    img = cv2.imread(path)
    return img


def normalized_image(img_arr):
    # 归一化图像
    img_arr = img_arr / 255.0
    return img_arr


def matchSift(img1_gray, img2_gray):
    """
    匹配sift特征
    """
    sift = cv2.SIFT()
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)
    # 蛮力匹配算法,有两个参数，距离度量(L2(default),L1)，是否交叉匹配(默认false)
    bf = cv2.BFMatcher()
    # 返回k个最佳匹配
    matches = bf.knnMatch(des1, des2, k=2)
    # cv2.drawMatchesKnn expects list of lists as matches.
    # opencv2.4.13没有drawMatchesKnn函数，需要将opencv2.4.13/sources/samples/python2下的common.py和find_obj文件放入当前目录，并导入
    p1, p2, kp_pairs = filter_matches(kp1, kp2, matches, ratio=0.5)
    matche_img = explore_match('find_obj', img1_gray, img2_gray, kp_pairs)  # cv2 shows image
    cv2.imwrite("C:/Users/qq619/Desktop/05matches.png", matche_img)
    # 保存特征连线图

    cv2.waitKey()
    cv2.destroyAllWindows()
    return p1, p2, des1, des2, matches


def save_match_features(match_features):
    # 存储特征信息到txt
    filename = 'match_features.txt'
    file = open(filename, mode='a')
    for item in match_features:
        file.write(str(item[0].distance) + " " + str(item[1].distance) + "/n")
    file.close()


def loadDataSet(fileName, delim=' '):
    # 从txt读取匹配特征信息到数组
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float, line) for line in stringArr]
    return np.mat(datArr)


def pca(dataMat, topNfeat=9999999):
    """
    PCA算法进行降维
    :param dataMat: 输入的数据
    :param topNfeat: 要降维到的维度
    :return: lowDDataMat: 输出的低维数据
             reconMat: 原数据
    """
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals  # 特征中心化。即每一维的数据都减去该维的均值
    covMat = np.cov(meanRemoved, rowvar=0)  # 计算meanRemoved的协方差矩阵covMat
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # 计算协方差矩阵covMat的特征值eigVals和特征向量eigVects
    eigValInd = np.argsort(eigVals)  # 排序, 将特征值从小到大排序
    eigValInd = eigValInd[:-(topNfeat + 1):-1]  # 去除不想要的维度
    redEigVects = eigVects[:, eigValInd]  # 将对应的特征向量从大到小识别出来
    lowDDataMat = meanRemoved * redEigVects  # 将数据转化成为新的维度
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat


def plotBestFit(dataSet1, dataSet2):
    # 可视化显示PCA降维数据与原数据
    dataArr1 = np.array(dataSet1)
    dataArr2 = np.array(dataSet2)
    n = np.shape(dataArr1)[0]
    n1 = np.shape(dataArr2)[0]
    xcord1 = [];
    ycord1 = []
    xcord2 = [];
    ycord2 = []
    xcord3 = [];
    ycord3 = []
    j = 0
    for i in range(n):
        xcord1.append(dataArr1[i, 0]);
        ycord1.append(dataArr1[i, 1])
        xcord2.append(dataArr2[i, 0]);
        ycord2.append(dataArr2[i, 1])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    plt.xlabel('X1');
    plt.ylabel('X2');
    plt.show()


if __name__ == '__main__':
    img1 = read_png("E:/image_compare/yp/1505.png")
    img2 = read_png("E:/image_compare/yp/1605.png")

    # img1 = read_png("/media/files/yp/rbm/15/1502.png")
    # img2 = read_png("/media/files/yp/rbm/16/1602.png")

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 图像归一化，因为cv2的原因无法使用
    # norm_img1 = normalized_image(img1)
    # norm_img2 = normalized_image(img2)

    p1, p2, des1, des2, match_features = matchSift(img1_gray, img2_gray)

    # des = np.row_stack((des1, des2))
    # kp = kp1 + kp2
    # # np.savetxt('des_features', des)
    # # mata = np.loadtxt('des_features')
    # mata = des
    # lowDDataMat, reconMat = pca(mata, 1)
    # plotBestFit(lowDDataMat, reconMat)

    # f = open("C:/Users/qq619/Desktop/point1_pairs.txt", mode='a')
    # for i in range(len(p1)):
    #     f.write(str(p1[i][0]) + ' ' + str(p1[i][1]) + '\n')
    # f.close()
    #
    # f = open("C:/Users/qq619/Desktop/point2_pairs.txt", mode='a')
    # for i in range(len(p2)):
    #     f.write(str(p2[i][0]) + ' ' + str(p2[i][1]) + '\n')
    # f.close()

    print 'done'
