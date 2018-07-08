# coding=utf-8
import cv2
import sys

import datetime
import numpy as np


class Area:
    pixList = []
    changedPixNum = 0
    isChanged = False

    def __init__(self, rgb):
        self.rgb = rgb
        self.pixList = []
        self.changedPixNum = 0
        self.isChanged = False

    def addPix(self, row, col):
        self.pixList.append([row, col])

    def isPixInArea(self, row, col):
        for pix in self.pixList:
            if pix[0] == row and pix[1] == col:
                return True
        return False

    def setChanged(self):
        self.isChanged = True

    def addChangedPixNum(self):
        self.changedPixNum += 1

    def getChangedRate(self):
        return self.changedPixNum * 1.0 / len(self.pixList)

    def isPixInAreaRange(self, row, col, pix_distance):
        if len(self.pixList) != 0:
            for pix in self.pixList:
                distance = pow(pow(pix[0] - row, 2) + pow(pix[1] - col, 2), 0.5)
                if distance <= pix_distance:
                    return True
            return False
        else:
            return False

    def toString(self):
        print
        print "Area:" + str(self)
        print "RGB:" + str(self.rgb[0]) + " " + str(self.rgb[1]) + " " + str(self.rgb[2])
        print "Changed Pixel Number:" + str(self.changedPixNum)
        print "is Area Changed:" + str(self.isChanged)
        print "Pixel List:"
        for pix in self.pixList:
            print "(" + str(pix[0]) + "," + str(pix[1]) + ")",


def isPixInAreaList(rgb, areaList):
    result = []
    if len(areaList) != 0:
        for i in range(len(areaList)):
            area = areaList[i]
            if area.rgb[0] == rgb[0] and area.rgb[1] == rgb[1] and area.rgb[2] == rgb[2]:
                result.append(i)
        return result
    else:
        return -1


def convert2Areas(m_shift_img, pix_distance):
    # 将meanshift图像转为对象列表并返回
    areaList = []
    row, col, dim = m_shift_img.shape
    count = 0
    for i in range(row):
        for j in range(col):
            count += 1
            idxList = isPixInAreaList(m_shift_img[i][j], areaList)
            if idxList != -1 and len(idxList) != 0:
                # 如果这个像素的颜色值在对象列表内，则找到对应的对象，检查距离是否在pix_distance内
                # 若在pix_distance内，则将像素坐标添加到对象内
                # 若不在范围内，则新建一个对象，将像素坐标添加到对象内，将对象加入对象列表
                IN_AREA_RANGE = False
                for idx in idxList:
                    area = areaList[idx]
                    if area.isPixInAreaRange(i, j, pix_distance):
                        area.addPix(i, j)
                        IN_AREA_RANGE = True
                        # area.toString()
                        break
                if not IN_AREA_RANGE:
                    area = Area(m_shift_img[i][j])
                    area.addPix(i, j)
                    areaList.append(area)
                    # area.toString()
                del area
            else:
                # 如果像素的颜色值不在对象列表内，则新建一个对象，并将像素坐标添加到对象内
                area = Area(m_shift_img[i][j])
                area.addPix(i, j)
                areaList.append(area)
                # area.toString()
                del area
            # view_bar(count, row * col)
    return areaList


def view_bar(num, total):
    rate = num * 1.0 / total
    rate_num = int(rate * 100)
    r = '\r[%s%s]%d%%' % ("*" * rate_num, "-" * (100 - rate_num), rate_num,)
    sys.stdout.write(r)
    sys.stdout.flush()


def combineMS(m_shift_path, prediction_path, pix_distance):
    m_shift_img = cv2.imread(m_shift_path)
    prediction_img = cv2.imread(prediction_path)
    # 防止尺寸不匹配
    row1, col1, dim1 = m_shift_img.shape
    row2, col2, dim2 = prediction_img.shape
    if row1 > row2:
        det = row1 - row2
        m_shift_img = m_shift_img[det:, :, :]
    elif row1 < row2:
        det = row2 - row1
        prediction_img = prediction_img[det:, :, :]
    if col1 > col2:
        det = col1 - col2
        m_shift_img = m_shift_img[:, :-det, :]
    elif col1 < col2:
        det = col2 - col1
        prediction_img = prediction_img[:, :-det, :]

    areaList = convert2Areas(m_shift_img, pix_distance)  # 先将meanshift图像转为对象列表
    # for area in areaList:
    #     area.toString()
    prediction_img = prediction_img[:, :, 0]
    row, col = prediction_img.shape
    for i in range(row):
        for j in range(col):
            if prediction_img[i][j] == 255:
                # 若像素是变化像素，寻找该像素在哪个对象内，增加该对象的变化像素数量
                for area in areaList:
                    if area.isPixInArea(i, j):
                        area.addChangedPixNum()
                        break

    # 设定阈值，变化率大于等于阈值即认为对象变化，将变化的对象可视化出来
    result = np.zeros([row, col])
    th = 0.3  # 阈值
    for area in areaList:
        if area.getChangedRate() >= th:
            area.isChanged = True
            for pix in area.pixList:
                result[pix[0]][pix[1]] = 255
    year = m_shift_path.split("gf")[0]
    cv2.imwrite("/media/files/yp/rbm/pic_div/combine_MS/combined_MS_prediction_ref_to_" + year + ".jpg", result)
    return result, row, col


start = datetime.datetime.now()
m_shift_path1 = "/media/files/yp/rbm/pic_div/combine_MS/2015gf2457.TIF"
m_shift_path2 = "/media/files/yp/rbm/pic_div/combine_MS/2016gf2457.TIF"
prediction_path = "/media/files/yp/rbm/pic_div/combine_MS/2016gf2457.jpg"
pix_distance = 10
result1, row1, col1 = combineMS(m_shift_path1, prediction_path, pix_distance)
result2, row2, col2 = combineMS(m_shift_path2, prediction_path, pix_distance)
result = np.zeros([row1, col1])
for i in range(row1):
    for j in range(col1):
        if result1[i][j] == 0 and result2[i][j] == 0:
            continue
        else:
            result[i][j] = 255
cv2.imwrite("/media/files/yp/rbm/pic_div/combine_MS/combined_MS_prediction.jpg", result)
end = datetime.datetime.now()
print "耗时：{0}秒".format((end - start).seconds)
