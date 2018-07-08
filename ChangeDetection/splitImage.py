import arcpy
import sys

imgPath = sys.argv[1]
imgDir = sys.argv[2]
imgPrefixName = sys.argv[3]
imgSuffixName = sys.argv[4]
gridPath = sys.argv[5]

arcpy.SplitRaster_management(imgPath, imgDir, imgPrefixName, "POLYGON_FEATURES", imgSuffixName, "NEAREST",
                             split_polygon_feature_class=gridPath)
