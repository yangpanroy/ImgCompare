# -*- coding: utf-8 -*-
from PIL import Image
from pylab import *
from numpy import *
import os

"""
注意：由于使用的RBM为VLFeat在Windows的exe和dll文件依赖，所以无法运行在ubuntu！
"""


def process_image(imagename, resultname, params="--edge-thresh 10 --peak-thresh 5"):
    """ 处理一幅图像，然后将结果保存在文件中"""

    if imagename[-3:] != 'pgm':
        # 创建一个pgm文件
        im = Image.open(imagename).convert('L')
        im.save('tmp.pgm')
        imagename = 'tmp.pgm'
    cmmd = str("sift " + imagename + " --output=" + resultname + " " + params)
    os.system(cmmd)
    print('processed', imagename, 'to', resultname)


def read_features_from_file(filename):
    """读取特征属性值，然后将其以矩阵的形式返回"""
    f = loadtxt(filename)
    return f[:, :4], f[:, 4:]  # 特征位置，描述子


def write_featrues_to_file(filename, locs, desc):
    """将特征位置和描述子保存到文件中"""
    savetxt(filename, hstack((locs, desc)))


def plot_features(im, locs, circle=False):
    """显示带有特征的图像
       输入：im（数组图像），locs（每个特征的行、列、尺度和朝向）"""

    def draw_circle(c, r):
        t = arange(0, 1.01, .01) * 2 * pi
        x = r * cos(t) + c[0]
        y = r * sin(t) + c[1]
        plot(x, y, 'b', linewidth=2)

    imshow(im)
    if circle:
        for p in locs:
            draw_circle(p[:2], p[2])
    else:
        plot(locs[:, 0], locs[:, 1], 'ob')
    axis('off')


imname = "C:/Users/qq619/Desktop/GMM_based_Change_Detection_Framework_Using_Very_High_resolution_Satellite_Imagery" \
         "-master/dataset/k02-05m-cropped.png "
im1 = array(Image.open(imname).convert('L'))
process_image(imname, 'k02-05m-cropped.sift')
l1, d1 = read_features_from_file('k02-05m-cropped.sift')

figure()
gray()
plot_features(im1, l1, circle=True)
show()
