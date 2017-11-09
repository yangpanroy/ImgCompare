# coding=utf-8
import tifffile as tiff
from matplotlib import pylab
import cv2
from PIL import Image
import numpy as np

img_path = "E:/image_compare/yp/15/1502.png"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = Image.open(img_path).convert('L')
img.show()

pylab.show()
