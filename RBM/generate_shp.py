# coding=utf-8
import os
from collections import defaultdict
import csv
import sys
import pylab
import cv2
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
import numpy as np
from collections import defaultdict


def mask_to_polygons(mask):
    epsilon = 2
    # first, find contours with cv2: it's much faster than shapely
    image, contours, hierarchy = cv2.findContours(((mask == 1) * 255).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                       for cnt in contours]
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= 1.:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= 1.])
            all_polygons.append(poly)

    all_polygons = MultiPolygon(all_polygons)

    if not all_polygons.is_valid:
        # return all_polygons.buffer(0)
        all_polygons = all_polygons.buffer(0)

        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])

    return all_polygons


sumpolygons = []
i = 0
for filename in os.listdir(r'E:\result\UnsupervisedModel\xingqing\15-16\img\quzao'):
    i = i + 1
    print(i)
    print(filename)
    new_mask = cv2.imread(r"E:\result\UnsupervisedModel\xingqing\15-16\img\quzao\{}".format(filename))
    new_mask = new_mask / 255
    new_maskd = new_mask[:, :, 0]
    im_size = new_maskd.shape[:2]

    new_pre2 = mask_to_polygons(new_maskd)

    import fileinput

    list2 = []
    for line in fileinput.input("D:/yinchuanyingxiang/2016fenge/xingqing/{}.tfw".format(filename.split(".")[0])):
        list2.append(float(line.split("\n")[0]))
    matrix = tuple(list2)
    scaled_pred_polygons = shapely.affinity.affine_transform(
        new_pre2, matrix=matrix)
    for ploy in scaled_pred_polygons:
        sumpolygons.append(ploy)

try:
    from osgeo import gdal
    from osgeo import ogr
except ImportError:
    import gdal
    import ogr


def WriteVectorFile(scaled_pred_polygons):
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")

    gdal.SetConfigOption("SHAPE_ENCODING", "")

    strVectorFile = "E:/result/UnsupervisedModel/xingqing/15-16/img/quzao/15-16XQDenoising.shp"

    ogr.RegisterAll()

    strDriverName = "ESRI Shapefile"
    oDriver = ogr.GetDriverByName(strDriverName)
    if oDriver == None:
        print("%s 驱动不可用！\n", strDriverName)

    oDS = oDriver.CreateDataSource(strVectorFile)
    if oDS == None:
        print("创建文件【%s】失败！", strVectorFile)

    papszLCO = []
    oLayer = oDS.CreateLayer("TestPolygon", None, ogr.wkbPolygon, papszLCO)
    if oLayer == None:
        print("图层创建失败！\n")

    oFieldID = ogr.FieldDefn("FieldID", ogr.OFTInteger)
    oLayer.CreateField(oFieldID, 1)

    oFieldName = ogr.FieldDefn("FieldName", ogr.OFTString)
    oFieldName.SetWidth(100)
    oLayer.CreateField(oFieldName, 1)

    oDefn = oLayer.GetLayerDefn()

    i = 0
    for poly in scaled_pred_polygons:
        oFeatureTriangle = ogr.Feature(oDefn)
        oFeatureTriangle.SetField(0, i)
        oFeatureTriangle.SetField(1, "bianhua")
        geomTriangle = ogr.CreateGeometryFromWkt(poly.to_wkt())
        oFeatureTriangle.SetGeometry(geomTriangle)
        oLayer.CreateFeature(oFeatureTriangle)
        i = i + 1

    oDS.Destroy()
    print("数据集创建完成！")


WriteVectorFile(sumpolygons)
