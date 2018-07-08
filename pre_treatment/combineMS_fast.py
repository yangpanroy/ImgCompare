# coding=utf-8
import datetime
import cv2
import numpy as np
import skimage.measure
import sys


def view_bar(num, total):
    rate = num * 1.0 / total
    rate_num = int(rate * 100)
    r = '\r[%s%s]%d%%  %d / %d' % ("*" * rate_num, "-" * (100 - rate_num), rate_num, num, total,)
    # r = '\r%d%%' % (rate_num,)
    # r = '\r%d / %d' % (num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()


def segmentImage(img, param):
    img = img.astype(np.float16)
    boolArray = img != param
    img[boolArray] = -1
    boolArray = img == param
    img[boolArray] = 1
    boolArray = img == -1
    img[boolArray] = 0
    return img


def getLabelFromValue(m_shift_img, value):
    imgB = m_shift_img[:, :, 0]
    imgG = m_shift_img[:, :, 1]
    imgR = m_shift_img[:, :, 2]
    b = value[0]
    g = value[1]
    r = value[2]
    imgB = segmentImage(imgB, b)
    imgG = segmentImage(imgG, g)
    imgR = segmentImage(imgR, r)
    label = imgR * imgG * imgB
    label[label == 1] = 255
    return label


def combineMS(m_shift_path, prediction_path, th):
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
    result = np.zeros([row, col], dtype=int)
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
                    # 获得MS图像中值为当前BGR的所有像素，用二值图的形式表示
                    currentLabelFromValue = getLabelFromValue(m_shift_img, value)
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
                    labelImage = segmentImage(labelImage, currentLabel)
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
            view_bar(count, row * col)
            # if count % 1000000 == 0:
            #     print("{0}‰".format(count / 1000000))
            # print str(count) + " / " + str(row * col)

    result[result > 0] = 255
    result[result <= 0] = 0
    result = result.astype(np.uint8)
    return result


start = datetime.datetime.now()
# m_shift_path1 = "/media/files/yp/rbm/13qu/2015/meanshift/2015gf746.TIF"
# m_shift_path2 = "/media/files/yp/rbm/13qu/2017/meanshift/201702gf746.TIF"
# prediction_path = "/media/files/yp/rbm/13qu/prediction/201702gf746.jpg"
m_shift_path1 = r"D:\yinchuanyingxiang\2015fenge\pipei\meanshift\2015gf2457.TIF"
m_shift_path2 = r"D:\yinchuanyingxiang\201702fenge\RGB\meanshift\201702gf2457.TIF"
prediction_path = r"E:\result\SupervisedModel\test\prediction\sdea_village\2016gf2457.jpg"
result1 = combineMS(m_shift_path1, prediction_path, 0.5)
# cv2.imwrite("C:/Users/qq619/Desktop/2015gf2239.jpg", result1)
result2 = combineMS(m_shift_path2, prediction_path, 0.5)
# cv2.imwrite("C:/Users/qq619/Desktop/201702gf2239.jpg", result2)
result = result1 + result2
result[result != 0] = 255
# cv2.imwrite("/media/files/yp/rbm/13qu/combineMS/201702gf746_fast75.jpg", result)
cv2.imwrite("C:/Users/qq619/Desktop/201702gf2457fastV.jpg", result)
end = datetime.datetime.now()
print("耗时：{0}秒".format((end - start).seconds))
