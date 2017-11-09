import shapefile

sf = shapefile.Reader("E:/银川/三百米裁切2/三百米裁切/shp/qtx_1.shp")
# i = 0
# for sr in sf.iterShapeRecords():
#     print(sr.record[0].decode('gb2312'))
#     for x,y in sr.shape.points:
#         print(x, y)
#         i = i + 1
# print(i)

print("shp文件中有" + str(len(sf.shapes())) + "个shapes")
# print(len(sf.shapes()))
shp = sf.shape(1)
print("以第一个shape为例")
print("第一个shape的类型为" + str(shp.shapeType))
print("数据范围(左下角的x，y坐标和右上角的x，y坐标)为" + str(shp.bbox))
print("这个shape有" + str(len(shp.parts)) + "个块")
# print(shp.parts)
print("这个shape有" + str(len(shp.points)) + "个坐标")
# print(len(shp.points))
print("坐标分别为：")
for point in shp.points:
    print(point)
