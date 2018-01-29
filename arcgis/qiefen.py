##====================================
##Split Raster
##Usage: SplitRaster_management in_raster out_folder out_base_name SIZE_OF_TILE
##                              | NUMBER_OF_TILES | TIFF | BMP | ENVI | ESRI BIL |
##                              ESRI BIP | ESRI BSQ | GIF | GRID | IMAGINE IMAGE |
##                              JP2 | JPG | PNG {NEAREST | BILINEAR | CUBIC |
##                              MAJORITY} {num_rasters} {tile_size} {overlap}
##                              {PIXELS | METERS | FEET | DEGREES | KILOMETERS |
##                              MILES} {cell_size} {origin}

try:
    import arcpy

    # arcpy.env.workspace = r"\\myServer\PrjWorkspace\RasGP"

    ##Equally split a large TIFF image by number of images
    arcpy.SplitRaster_management(in_raster = "D:/qiefen/2016/yinchuan20000.TIF", out_folder = "C:/Users/qq619/Desktop/qiefen", out_base_name = "qiefen", split_method = "SIZE_OF_TILE", format = "TIFF", resampling_type = "NEAREST", tile_size = "1000 1000", overlap = "0", units = "PIXELS")

    ##Equally split a large TIFF image by size of images
    # arcpy.SplitRaster_management("large.tif", "splitras", "size2", "SIZE_OF_TILE", \
    #                              "TIFF", "BILINEAR", "#", "3500 3500", "4", "PIXELS", \
    #                              "#", "-50 60")

except:
    print "Split Raster exsample failed."
    print arcpy.GetMessages()