# coding=utf-8
import cv2
import datetime
import numpy as np
import os

import sys

import skimage.measure
from sklearn.decomposition import PCA
from skimage import img_as_ubyte
from colour import Color


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


class imgProcess(object):
    def __init__(self):
        pass

    def showTiff(self, path, imgType=None):
        img = cv2.imread(path, 3)
        img = img.astype(np.uint8)
        if len(img.shape) == 3:
            if imgType == "BGR":
                img = img[:, :, (2, 1, 0)]
        if len(img.shape) > 3:
            img = img[:, :, (0, 1, 2)]
        cv2.imshow(os.path.splitext(path)[-1], img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def rgb2Hsi(self, rgb_lwpImg):
        rows = int(rgb_lwpImg.shape[0])
        cols = int(rgb_lwpImg.shape[1])
        b, g, r = cv2.split(rgb_lwpImg)
        # 归一化到[0,1]
        b = b / 255.0
        g = g / 255.0
        r = r / 255.0
        hsi_lwpImg = rgb_lwpImg.copy()
        for i in range(rows):
            for j in range(cols):
                num = 0.5 * ((r[i, j] - g[i, j]) + (r[i, j] - b[i, j]))
                den = np.sqrt((r[i, j] - g[i, j]) ** 2 + (r[i, j] - b[i, j]) * (g[i, j] - b[i, j]))
                theta = float(np.arccos(num / den))

                if den == 0:
                    H = 0
                elif b[i, j] <= g[i, j]:
                    H = theta
                else:
                    H = 2 * 3.14169265 - theta

                min_RGB = min(min(b[i, j], g[i, j]), r[i, j])
                sum = b[i, j] + g[i, j] + r[i, j]
                if sum == 0:
                    S = 0
                else:
                    S = 1 - 3 * min_RGB / sum

                H = H / (2 * 3.14159265)
                I = sum / 3.0
                # 输出HSI图像，扩充到255以方便显示，一般H分量在[0,2pi]之间，S和I在[0,1]之间
                hsi_lwpImg[i, j, 0] = H * 255
                hsi_lwpImg[i, j, 1] = S * 255
                hsi_lwpImg[i, j, 2] = I * 255
        return hsi_lwpImg

    def getShadow(self, path):
        img = cv2.imread(path)
        rows, cols, dim = img.shape
        img2 = np.reshape(img, (rows * cols, 3))
        pca = PCA(n_components=1)
        newData = pca.fit_transform(img2)

        for i in range(rows * cols):
            if newData[i, 0] > 0:
                newData[i, 0] = 0
        minData = np.min(newData)
        norData = newData / minData
        newImg = np.reshape(norData, (rows, cols))

        hsi_lwpImg = self.rgb2Hsi(img)
        hsi_lwpImg = hsi_lwpImg / 255.0  # 归一化

        SI = np.zeros((rows, cols))
        for i in range(rows):
            for j in range(cols):
                SI[i, j] = (newImg[i, j] - hsi_lwpImg[i, j, 2]) * (1 + hsi_lwpImg[i, j, 1]) / (
                        newImg[i, j] + hsi_lwpImg[i, j, 2] + hsi_lwpImg[i, j, 1])
        SI = img_as_ubyte(SI)
        ret, th = cv2.threshold(SI, 0, 255, cv2.THRESH_OTSU)
        print(ret)
        return th

    def view_bar(self, num, total):
        rate = num * 1.0 / total
        rate_num = int(rate * 100)
        r = '\r[%s%s] %d%%   %s / %s' % ("*" * rate_num, "-" * (100 - rate_num), rate_num, num, total,)
        sys.stdout.write(r)
        sys.stdout.flush()

    def batchRemoveShadow(self, predictions_dir, label1_dir, label2_dir):
        resultDir = predictions_dir + "img-shadow_change/"
        denoiseResultDir = predictions_dir + "quzao/"
        if not os.path.exists(resultDir):
            os.makedirs(resultDir)
        if not os.path.exists(denoiseResultDir):
            os.makedirs(denoiseResultDir)
        f_list1 = os.listdir(predictions_dir)
        file_num = 0
        for file_name in f_list1:
            if os.path.splitext(file_name)[1] == '.jpg':
                file_id = os.path.splitext(file_name)[0].split("gf")[1]
                img = cv2.imread(predictions_dir + file_name, 0)
                img1_path = label1_dir + "2016gf" + file_id + '.TIF'
                img2_path = label2_dir + "201701gf" + file_id + '.TIF'
                shadow1 = self.getShadow(img1_path)
                shadow2 = self.getShadow(img2_path)
                rows, cols = img.shape
                shadow_change = np.zeros((rows, cols))
                for i in range(rows):
                    for j in range(cols):
                        if shadow1[i, j] != shadow2[i, j]:
                            shadow_change[i, j] = 255
                            if img[i, j] != 0:
                                img[i, j] = 0
                cv2.imwrite(resultDir + file_name, img)
                result = cv2.medianBlur(img, 9)
                cv2.imwrite(denoiseResultDir + file_name, result)
                file_num = file_num + 1
                self.view_bar(file_num, len(f_list1))

    def segmentImage(self, img, param):
        img = img.astype(np.float16)
        boolArray = img != param
        img[boolArray] = -1
        boolArray = img == param
        img[boolArray] = 1
        boolArray = img == -1
        img[boolArray] = 0
        return img

    def getLabelFromValue(self, m_shift_img, value):
        imgB = m_shift_img[:, :, 0]
        imgG = m_shift_img[:, :, 1]
        imgR = m_shift_img[:, :, 2]
        b = value[0]
        g = value[1]
        r = value[2]
        imgB = self.segmentImage(imgB, b)
        imgG = self.segmentImage(imgG, g)
        imgR = self.segmentImage(imgR, r)
        label = imgR * imgG * imgB
        label[label == 1] = 255
        return label

    def combineMS_fast(self, img_path, prediction_path, th):
        img = cv2.imread(img_path)
        prediction_img = cv2.imread(prediction_path)
        # 防止尺寸不匹配
        row1, col1, dim1 = img.shape
        row2, col2, dim2 = prediction_img.shape
        if row1 > row2:
            det = row1 - row2
            img = img[det:, :, :]
        elif row1 < row2:
            det = row2 - row1
            prediction_img = prediction_img[det:, :, :]
        if col1 > col2:
            det = col1 - col2
            img = img[:, :-det, :]
        elif col1 < col2:
            det = col2 - col1
            prediction_img = prediction_img[:, :-det, :]

        m_shift_img = self.meanshift(img)
        prediction_img = prediction_img[:, :, 0]
        prediction_img = cv2.medianBlur(prediction_img, 9)
        row, col = prediction_img.shape
        result = np.zeros([row, col])
        count = 0
        currentChangeImage = np.zeros([row, col])
        m_shift_img = m_shift_img.astype(np.float16)
        prediction_img = prediction_img.astype(np.float16)
        for i in range(row):
            for j in range(col):
                count = count + 1
                if prediction_img[i][j] == 255:
                    # 若当前像素为变化像素，获取当前像素的 BGR 值
                    value = m_shift_img[i][j]
                    value = tuple(value)
                    if result[i][j] == 0:
                        if currentChangeImage[i][j] == 0:
                            # 获得MS图像中值为当前BGR的所有像素，用二值图的形式表示
                            currentLabelFromValue = self.getLabelFromValue(m_shift_img, value)
                            # cv2.imshow('currentLabelFromValue', currentLabelFromValue)
                            # 进行连通区域检查，得到连通区域阶梯图
                            labelImage = skimage.measure.label(currentLabelFromValue)
                            # cv2.imshow('labelImage1', labelImage)
                            currentLabel = labelImage[i][j]
                            # 得到每个连通区域的属性，通过当前像素的label值，获得面积
                            props = skimage.measure.regionprops(labelImage)
                            areaSize = 1
                            for prop in props:
                                if prop.label == currentLabel:
                                    areaSize = prop.filled_area
                                    break
                            labelImage = self.segmentImage(labelImage, currentLabel)
                            labelImage[labelImage == 1] = 255
                            # cv2.imshow('labelImage2', labelImage)
                            currentChangeImage = labelImage * prediction_img
                            currentChangeImage[currentChangeImage != 0] = 1
                            currentChangeImage = currentChangeImage.astype(np.uint8)
                            # cv2.imshow('currentChangeImage', labelImage)
                            props = skimage.measure.regionprops(currentChangeImage)
                            changedPixNum = 0
                            for prop in props:
                                if prop.label == 1:
                                    changedPixNum = prop.filled_area
                                    break
                            if changedPixNum * 1.0 / areaSize >= th:
                                result = labelImage + result
                            else:
                                result = result - labelImage
                            # cv2.imshow('result', result)
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()
                self.view_bar(count, row * col)
                # print str(count) + " / " + str(row * col)

        result[result > 0] = 255
        result[result <= 0] = 0
        result = result.astype(np.uint8)
        return result

    def combineMS_normal(self, img_path, prediction_path, th=0.5):
        pix_distance = 10

        def convert2Areas(m_shift_img, pix_distance):
            # 将meanshift图像转为对象列表并返回
            areaMap = {}  # 这里使用字典用键值对的方式存储，便于查找，RGB元组作为key，area列表作为value
            row, col, dim = m_shift_img.shape
            count = 0
            for i in range(row):
                for j in range(col):
                    count += 1
                    if tuple(m_shift_img[i][j]) in areaMap:
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
                    self.view_bar(count, row * col)
                    # print count
                    # if count % 100000 == 0:
                    #     print('{0}%'.format(count / 10000.0))
            return areaMap

        img = cv2.imread(img_path)
        prediction_img = cv2.imread(prediction_path)
        # 防止尺寸不匹配
        row1, col1, dim1 = img.shape
        row2, col2, dim2 = prediction_img.shape
        if row1 > row2:
            det = row1 - row2
            img = img[det:, :, :]
        elif row1 < row2:
            det = row2 - row1
            prediction_img = prediction_img[det:, :, :]
        if col1 > col2:
            det = col1 - col2
            img = img[:, :-det, :]
        elif col1 < col2:
            det = col2 - col1
            prediction_img = prediction_img[:, :-det, :]

        m_shift_img = self.meanshift(img)
        areaMap = convert2Areas(m_shift_img, pix_distance)  # 先将meanshift图像转为对象列表
        # for area in areaList:
        #     area.toString()
        prediction_img = prediction_img[:, :, 0]
        prediction_img = self.medianBlur(prediction_img, 9)
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
        for key in areaMap:
            areaList = areaMap.get(key)
            for area in areaList:
                if area.getChangedRate() >= th:
                    area.isChanged = True
                    for pix in area.pixList:
                        result[pix[0]][pix[1]] = 255
        # year = m_shift_path.split("gf")[0]
        # cv2.imwrite("/media/files/yp/rbm/pic_div/combine_MS/combined_MS_prediction_ref_to_" + year + ".jpg", result)
        return result

    def combineMSSingle(self, img_path1, img_path2, prediction_path, outPath, th=0.5, flag="fast"):
        result1 = None
        result2 = None
        if flag == "fast":
            result1 = self.combineMS_fast(img_path1, prediction_path, th)
            result2 = self.combineMS_fast(img_path2, prediction_path, th)
        if flag == "hash":
            result1 = self.combineMS_normal(img_path1, prediction_path, th)
            result2 = self.combineMS_normal(img_path2, prediction_path, th)
        if flag == "upgrade":
            result1 = self.combineMS_fast_upgrade(img_path1, prediction_path)
            result2 = self.combineMS_fast_upgrade(img_path2, prediction_path)
        result = result1 + result2
        result[result != 0] = 255
        cv2.imwrite(outPath, result)

    def meanshift(self, img):
        # bf = cv2.bilateralFilter(img, d=0, sigmaColor=25, sigmaSpace=15)
        ms = cv2.pyrMeanShiftFiltering(img, sp=50, sr=30)
        return ms

    def combineMS_fast_upgrade(self, m_shift_path, prediction_path):
        def getLabelImage(img):
            result = np.zeros((img.shape[0], img.shape[1]), np.uint)
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    value = img[i][j] / 255.0
                    c = Color(rgb=tuple(value))
                    num = int(c.hex_l.split('#')[-1], 16)
                    result[i][j] = num
            return skimage.measure.label(result)

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

        prediction_img = prediction_img[:, :, 0]
        prediction_img = cv2.medianBlur(prediction_img, 9)
        row, col = prediction_img.shape
        result = np.zeros([row, col])
        count = 0
        currentChangeImage = np.zeros([row, col])
        m_shift_img = m_shift_img.astype(np.float16)
        prediction_img = prediction_img.astype(np.float16)

        labelImage = getLabelImage(m_shift_img)
        props = skimage.measure.regionprops(labelImage)
        labelMap = {}
        for prop in props:
            labelMap[prop.label] = prop.filled_area
        maxArea = max(labelMap.values())

        for i in range(row):
            for j in range(col):
                count = count + 1
                if prediction_img[i][j] == 255:
                    # 若当前像素为变化像素，获取当前像素的 BGR 值
                    if result[i][j] == 0:
                        if currentChangeImage[i][j] == 0:
                            currentLabel = labelImage[i][j]
                            areaSize = labelMap.get(currentLabel)
                            th = 1.0 / (2 - areaSize * 1.0 / maxArea)
                            currentLabelImage = self.segmentImage(labelImage, currentLabel)
                            # currentLabelImage[currentLabelImage == 1] = 255
                            # cv2.imshow('labelImage2', labelImage)
                            currentChangeImage = currentLabelImage * prediction_img
                            currentChangeImage[currentChangeImage != 0] = 255
                            currentChangeImage = currentChangeImage.astype(np.uint8)
                            # cv2.imshow('currentChangeImage', labelImage)
                            props = skimage.measure.regionprops(currentChangeImage)
                            changedPixNum = 0
                            for prop in props:
                                if prop.label == 255:
                                    changedPixNum = prop.filled_area
                                    break
                            if changedPixNum * 1.0 / areaSize >= th:
                                result = currentLabelImage + result
                            # cv2.imshow('result', result)
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()
                self.view_bar(count, row * col)
                # if count % 10000 == 0:
                #     print("{0}‰".format(count / 10000))
                # print str(count) + " / " + str(row * col)

        result[result != 0] = 255
        return result

    def medianBlur(self, img, kernelSize=15):
        return cv2.medianBlur(img, kernelSize)


if __name__ == "__main__":
    path1 = r"D:\yinchuanyingxiang\2015fenge\2015gf222.TIF"
    path2 = r"D:\yinchuanyingxiang\201702fenge\RGB\201702gf222.TIF"
    prediction_path = r"E:\result\UnsupervisedModel\yinchuansanqu\15_1702\201702gf222.jpg"
    out_dir = r"C:\Users\qq619\Desktop\result222_fastU.png"
    ip = imgProcess()
    ip.combineMSSingle(path1, path2, prediction_path, out_dir, 0.75, "upgrade")
