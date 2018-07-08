# coding=utf-8
import tifffile as tiff


def divide_img(img1_path, img2_path):
    img1 = tiff.imread(img1_path)
    img2 = tiff.imread(img2_path)
    h1, w1, b1 = img1.shape
    h2, w2, b2 = img2.shape
    if h1 == h2 and w1 == w2 and b1 == b2:
        sub_img_size = 1000
        h_num = h1 / sub_img_size
        w_num = w1 / sub_img_size
        for i in range(0, h_num):
            for j in range(0, w_num):
                h_start = i * sub_img_size
                h_end = (i + 1) * sub_img_size
                w_start = j * sub_img_size
                w_end = (j + 1) * sub_img_size
                sub_img1 = img1[h_start: h_end, w_start: w_end, :]
                tiff.imsave("D:/pic_div/1/{}.tif".format(i * w_num + j), sub_img1)
                sub_img2 = img2[h_start: h_end, w_start: w_end, :]
                tiff.imsave("D:/pic_div/2/{}.tif".format(i * w_num + j), sub_img2)
    else:
        print "给出的两个图像尺寸不匹配，请检查！"


if __name__ == '__main__':
    img1_path = 'D:/qiefen/2015/20150.TIF'
    img2_path = 'D:/qiefen/2016/yinchuan20000.TIF'
    divide_img(img1_path, img2_path)
