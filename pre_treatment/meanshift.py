# coding=utf-8
import arcpy
from arcpy.sa import *
import os
import sys


def view_bar(num, total):
    rate = num * 1.0 / total
    rate_num = int(rate * 100)
    r = '\r[%s%s]%d%%' % ("*" * rate_num, "-" * (100 - rate_num), rate_num,)
    sys.stdout.write(r)
    sys.stdout.flush()


# seg_raster = SegmentMeanShift("D:/pic_div/2016/0.tif", "19", "19", "10")
# seg_raster.save("C:/Users/qq619/Desktop/read_tif2.tif")

# for i in range(0, 40):
#     seg_raster = SegmentMeanShift("D:/pic_div/2016/" + str(i) + ".png", "19", "19", "10")
#     seg_raster.save("D:/pic_div/2016/meanshift/" + str(i) + ".tif")

path = "F:/13qu/17/RGB/"
# 返回所给路径下的图像数量
f_list = os.listdir(path)
file_num = 0
for i in f_list:
    if os.path.splitext(i)[1] == '.TIF':
        seg_raster = SegmentMeanShift(path + i, "19", "19", "10")
        seg_raster.save(path + "meanshift/" + i)
        file_num = file_num + 1
        view_bar(file_num, len(f_list))

# path = "D:/yinchuanyingxiang/2016fenge/RGB/"
# path2 = "D:/yinchuanyingxiang/2016fenge/RGB/meanshift/"
# # 返回所给路径下的图像数量
# f_list = os.listdir(path)
# f_list2 = os.listdir(path2)
# file_num = 0
# for i in f_list:
#     if os.path.splitext(i)[1] == '.TIF':
#         print i
#         if i not in f_list2:
#             seg_raster = SegmentMeanShift(path + i, "19", "19", "10")
#             seg_raster.save(path + "meanshift/" + i)
#             file_num = file_num + 1
#             # view_bar(file_num, len(f_list))
