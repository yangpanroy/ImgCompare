import cv2
import numpy as np
import os
import sys


def view_bar(num, total):
    rate = num * 1.0 / total
    rate_num = int(rate * 100)
    r = '\r[%s%s]%d%%' % ("*" * rate_num, "-" * (100 - rate_num), rate_num,)
    sys.stdout.write(r)
    sys.stdout.flush()


img1_dir = '/media/files/yp/rbm/yinchuansanqu/2015/'
img2_dir = '/media/files/yp/rbm/yinchuansanqu/2016/'

f_list1 = os.listdir(img1_dir)
f_list2 = os.listdir(img2_dir)
file_num = 0
for i in f_list1:
    if os.path.splitext(i)[1] == '.TIF':
        file_id = i.split("gf")[-1]
        another_file_name = "2016gf" + file_id
        if another_file_name in f_list2:
            img1_path = img1_dir + i
            img2_path = img2_dir + another_file_name
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            row1, col1, dim1 = img1.shape
            row2, col2, dim2 = img2.shape
            det = 0
            if row1 > row2:
                det = row1 - row2
                img1 = img1[det:, :, :]
            elif row1 < row2:
                det = row2 - row1
                img2 = img2[det:, :, :]
            if col1 > col2:
                det = col1 - col2
                img1 = img1[:, :-det, :]
            elif col1 < col2:
                det = col2 - col1
                img2 = img2[:, :-det, :]
            file_num = file_num + 1
            view_bar(file_num, len(f_list1))
