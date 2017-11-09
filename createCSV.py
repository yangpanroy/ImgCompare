import csv

import shapefile

from model.Circle import Circle
from model.Point import Point

FIRST_TIME_FLAG = True
WRITE_FILE_PATH = "C:/Users/qq619/Desktop/wkt_converted.csv"  # 生成的文件所处位置

type_list = []
circle_list = []
part_list = []

# 准备写入csv文件
with open(WRITE_FILE_PATH, 'w', newline='') as csvfile:
    fieldnames = ['ImageId', 'ClassType', 'MultipolygonWKT']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if FIRST_TIME_FLAG is True:  # 第一次进入写表头
        writer.writeheader()
        FIRST_TIME_FLAG = False

        for item in range(100):
            # print(item+1)
            file_name = "qtx_" + str(item + 1)  # 文件名
            sf = shapefile.Reader("E:/银川/三百米裁切2/三百米裁切/shp/" + file_name + ".shp")  # 读取shp文件
            circle = Circle()
            for sr in sf.iterShapeRecords():
                shape_type = sr.record[0].decode('gb2312')  # 类型
                # print(shape_type)
                type_list.append(shape_type)
                part_list.append(len(sr.shape.parts))
                count = 0
                for x, y in sr.shape.points:
                    if count in sr.shape.parts:  # 块
                        # print(count)
                        # print(circle)
                        circle_list.append(circle.__repr__())
                        circle.clear_circle()
                    point = Point(x, y)
                    # print(point.__repr__())
                    circle.add_point(point)
                    count += 1
            # print(circle)
            circle_list.append(circle.__repr__())
            circle.clear_circle()

            del circle_list[0]

            # for az in type_list:
            #     print(az)
            # for sd in circle_list:
            #     print(sd)
            # for kj in part_list:
            #     print(kj)

            csv_type_list = []
            csv_multipolygonWKT = []
            s = "MULTIPOLYGON("
            index = 0
            temp = ""
            e = ")"
            for i in range(len(type_list)):
                if type_list[i] not in csv_type_list:
                    csv_type_list.append(type_list[i])
                    for j in range(index, index + part_list[i]):
                        temp = str(circle_list[j]) + ","
                    index = index + part_list[i]
                    # temp = temp[:-1]
                    csv_multipolygonWKT.append(temp)
                    temp = ""
                else:
                    for j in range(index, index + part_list[i]):
                        temp = temp + str(circle_list[j]) + ","
                    index = index + part_list[i]
                    csv_multipolygonWKT[csv_type_list.index(type_list[i])] = csv_multipolygonWKT[
                                                                                 csv_type_list.index(
                                                                                     type_list[i])] + temp
                    temp = ""

            for az in csv_type_list:
                print(az)

            for i in range(len(csv_multipolygonWKT)):
                csv_multipolygonWKT[i] = csv_multipolygonWKT[i][:-1]
                csv_multipolygonWKT[i] = s + csv_multipolygonWKT[i] + e

            for kj in csv_multipolygonWKT:
                print(kj)

            for i in range(len(csv_type_list)):
                writer.writerow(
                    {'ImageId': file_name, 'ClassType': csv_type_list[i], 'MultipolygonWKT': csv_multipolygonWKT[i]})

            type_list = []
            csv_type_list = []
            circle_list = []
            part_list = []

    print("生成成功！请查看" + WRITE_FILE_PATH)
