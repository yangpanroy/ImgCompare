# coding=utf-8
import cv2
import sys
import os

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


def convert2Areas(m_shift_img, pix_distance):
    # 将meanshift图像转为对象列表并返回
    areaMap = {}  # 这里使用字典用键值对的方式存储，便于查找，RGB元组作为key，area列表作为value
    row, col, dim = m_shift_img.shape
    count = 0
    for i in range(row):
        for j in range(col):
            count += 1
            if areaMap.has_key(tuple(m_shift_img[i][j])):
                # 如果这个像素的颜色值在对象字典内，则找到对应的对象，检查距离是否在pix_distance内
                # 若在pix_distance内，则将像素坐标添加到对象内
                # 若不在范围内，则新建一个对象，将像素坐标添加到对象内，将对象加入对象列表
                areaList = areaMap.get(tuple(m_shift_img[i][j]))
                IN_AREA_RANGE = False
                for area in areaList:
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
                # 如果像素的颜色值不在对象字典内，则新建一个对象，并将像素坐标添加到对象内
                area = Area(m_shift_img[i][j])
                area.addPix(i, j)
                areaList = [area]
                rgb = tuple(m_shift_img[i][j])
                areaMap[rgb] = areaList
                # area.toString()
                del area
            # view_bar(count, row * col)
            # print count
            if count % 100000 == 0:
                print '{0}%'.format(count/10000.0)
    return areaMap


def view_bar(num, total):
    rate = num * 1.0 / total
    rate_num = int(rate * 100)
    # r = '\r[%s%s]%d%%' % ("*" * rate_num, "-" * (100 - rate_num), rate_num,)
    r = '\r%d%%' % (rate_num,)
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

    areaMap = convert2Areas(m_shift_img, pix_distance)  # 先将meanshift图像转为对象列表
    # for area in areaList:
    #     area.toString()
    prediction_img = prediction_img[:, :, 0]
    row, col = prediction_img.shape
    for i in range(row):
        for j in range(col):
            if prediction_img[i][j] == 255:
                # 若像素是变化像素，寻找该像素在哪个对象内，增加该对象的变化像素数量
                # 这里先获取RGB值，用RGB值查询字典，缩小搜索范围
                rgb = tuple(m_shift_img[i][j])
                areaList = areaMap.get(rgb)
                for area in areaList:
                    if area.isPixInArea(i, j):
                        area.addChangedPixNum()
                        break

    # 设定阈值，变化率大于等于阈值即认为对象变化，将变化的对象可视化出来
    result = np.zeros([row, col])
    th = 0.5  # 阈值
    for key in areaMap:
        areaList = areaMap.get(key)
        for area in areaList:
            if area.getChangedRate() >= th:
                area.isChanged = True
                for pix in area.pixList:
                    result[pix[0]][pix[1]] = 255
    # year = m_shift_path.split("gf")[0]
    # cv2.imwrite("/media/files/yp/rbm/pic_div/combine_MS/combined_MS_prediction_ref_to_" + year + ".jpg", result)
    return result, row, col


m_shift_path1 = "/media/files/yp/rbm/yinchuansanqu/2015/xingqing/"
m_shift_path2 = "/media/files/yp/rbm/yinchuansanqu/201702/"
prediction_path = "/media/files/yp/rbm/pic_div/predictions/img/1/"
# m_shift_path1 = "E:/result/pre_post_treatment/combined_MS/2015/"
# m_shift_path2 = "E:/result/pre_post_treatment/combined_MS/201702/"
# prediction_path = "E:/result/pre_post_treatment/combined_MS/predictions/"
pix_distance = 10
f_list1 = os.listdir(m_shift_path1)
f_list2 = os.listdir(m_shift_path2)
f_list3 = os.listdir(prediction_path)
file_num = 0
start = datetime.datetime.now()
for file_name in f_list1:
    if os.path.splitext(file_name)[1] == ".TIF":
        file_id = file_name.split('gf')[-1]
        another_file_name = "201702gf" + file_id
        third_name = another_file_name.split('.')[0] + '.jpg'
        if another_file_name in f_list2 and third_name in f_list3:
            print "正在处理第{0}/{1}个图像".format(file_num + 1, len(f_list2))
            img1_path = m_shift_path1 + file_name
            img2_path = m_shift_path2 + another_file_name
            pred_path = prediction_path + third_name
            result1, row1, col1 = combineMS(img1_path, pred_path, pix_distance)
            result2, row2, col2 = combineMS(img2_path, pred_path, pix_distance)
            result = np.zeros([row1, col1])
            for i in range(row1):
                for j in range(col1):
                    if result1[i][j] == 0 and result2[i][j] == 0:
                        continue
                    else:
                        result[i][j] = 255
            # cv2.imwrite("E:/result/pre_post_treatment/combined_MS/combineMS/" + third_name, result)
            cv2.imwrite("/media/files/yp/rbm/pic_div/predictions/combineMS/" + third_name, result)
            file_num += 1
            end = datetime.datetime.now()
            print "已用时间：{0}秒    预计时间：{1}秒    剩余时间：{2}秒".format((end - start).seconds,
                                                               (end - start).seconds / file_num * len(f_list2),
                                                               (end - start).seconds / file_num * len(f_list2) - (
                                                                       end - start).seconds)
