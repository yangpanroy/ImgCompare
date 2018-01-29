import cv2
import histprocessing as hp
import sys
import os


# refImg = cv2.imread('/media/files/yp/rbm/pic_div/2016/0.png')
# oriImg = cv2.imread('/media/files/yp/rbm/pic_div/2015/0.png')
# refImg = cv2.imread('C:/Users/qq619/Desktop/read_tif2.tif')
# oriImg = cv2.imread('C:/Users/qq619/Desktop/read_tif.tif')
#
# outImg1 = hp.histMatching(oriImg[:, :, 0], refImg[:, :, 0])
# outImg2 = hp.histMatching(oriImg[:, :, 1], refImg[:, :, 1])
# outImg3 = hp.histMatching(oriImg[:, :, 2], refImg[:, :, 2])
# outImg = cv2.merge([outImg1, outImg2, outImg3])
#
# # cv2.imshow('oriImg', oriImg)
# # cv2.imshow('refImg', refImg)
# # cv2.imshow('outImg', outImg)
#
# # cv2.imwrite("/media/files/yp/rbm/pic_div/pipei.png", outImg)
# cv2.imwrite("C:/Users/qq619/Desktop/pipei.tif", outImg)
#
# cv2.waitKey()
# cv2.destroyAllWindows()

def view_bar(num, total):
    rate = num * 1.0 / total
    rate_num = int(rate * 100)
    r = '\r[%s%s]%d%%' % ("*" * rate_num, "-" * (100 - rate_num), rate_num,)
    sys.stdout.write(r)
    sys.stdout.flush()


path1 = "D:/yinchuanyingxiang/2016fenge/"
path2 = "D:/yinchuanyingxiang/2015fenge/"
f_list1 = os.listdir(path1)
f_list2 = os.listdir(path2)
file_num = 0
for i in f_list1:
    if os.path.splitext(i)[1] == '.TIF':
        file_id = i.split("gf")[-1]
        another_file_name = "2015gf" + file_id
        if another_file_name in f_list2:
            refImg = cv2.imread(path2 + another_file_name)
            oriImg = cv2.imread(path1 + i)
            outImg1 = hp.histMatching(oriImg[:, :, 0], refImg[:, :, 0])
            outImg2 = hp.histMatching(oriImg[:, :, 1], refImg[:, :, 1])
            outImg3 = hp.histMatching(oriImg[:, :, 2], refImg[:, :, 2])
            outImg = cv2.merge([outImg1, outImg2, outImg3])
            cv2.imwrite(path1 + "pipei/" + i, outImg)
            file_num = file_num + 1
            view_bar(file_num, len(f_list1) / 4)
        else:
            print i
