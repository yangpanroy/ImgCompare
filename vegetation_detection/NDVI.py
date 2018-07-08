import cv2
import numpy as np
import tifffile as tiff
from skimage import img_as_ubyte


img2 = tiff.imread("D:/yinchuanyingxiang/2016fenge/xingqing/2016gf384.TIF")
print img2.shape
Red = img2[:, :, 0]
Blue = img2[:, :, 2]
NIR = img2[:, :, 3]
Red = Red / 255.0
Blue = Blue / 255.0
NIR = NIR / 255.0
NDVI = ((NIR - Red)/(NIR + Red))
NDVI = img_as_ubyte(NDVI)
# ret, th = cv2.threshold(NDVI, 0, 255, cv2.THRESH_OTSU)
ret, th = cv2.threshold(NDVI, 127, 255, cv2.THRESH_BINARY)
print ret
EVI = (NIR - Red) / (NIR + 6.0 * Red - 7.5 * Blue + 1) * 2.5
EVI_img = np.zeros(img2.shape[0:2])
result = np.zeros(img2.shape[0:2])
for i in xrange(img2.shape[0]):
    for j in xrange(img2.shape[1]):
        if 0.8 < EVI[i, j]:
            EVI_img[i, j] = 255
            if th[i, j] == 255:
                result[i, j] = 255
cv2.imshow("result", result)
cv2.imshow("EVI_img", EVI_img)
cv2.imshow("NDVI", th)
cv2.waitKey()
cv2.destroyAllWindows()
