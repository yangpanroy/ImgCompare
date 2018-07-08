# coding=utf-8
import numpy as np
import cv2
import sys
import os


def view_bar(num, total):
    rate = num * 1.0 / total
    rate_num = int(rate * 100)
    r = '\r[%s%s]%d%%' % ("*" * rate_num, "-" * (100 - rate_num), rate_num,)
    sys.stdout.write(r)
    sys.stdout.flush()


# predictions = np.load("/media/files/yp/rbm/yinchuansanqu/predictions/21104.TIF_RGB.npy")
# # predictions = np.load("/media/files/yp/rbm/predictions.npy")
# length, classes = np.shape(predictions)
# predictions_label = []
# for i in range(length):
#     if predictions[i][0] > predictions[i][1]:
#         predictions_label.append(255)
#     else:
#         predictions_label.append(0)
# predictions_label = np.array(predictions_label)
# print 'read file done'
# img_path = '/media/files/yp/rbm/yinchuansanqu/2016/2016gf21104.TIF'
# img = cv2.imread(img_path)
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# height, width = np.shape(img_gray)
# predictions_img = predictions_label.reshape(height, width)
# # cv2.imshow('predictions_img', predictions_img)
# cv2.imwrite('/media/files/yp/rbm/yinchuansanqu/predictions/img/21104.jpg', predictions_img)

# predictions_dir = "C:\\Users\\qq619\\Desktop\\predictions\\"
# label_dir = "D:\\yinchuanyingxiang\\2016fenge\\pipei\\"
# f_list1 = os.listdir(predictions_dir)
# file_num = 0
# for i in f_list1:
#     n_list = os.path.splitext(i)[0].split('.')[0]
#     if os.path.splitext(i)[1] == '.npy':
#         predictions = np.load(predictions_dir + i)
#         # predictions = np.load("/media/files/yp/rbm/predictions.npy")
#         length, classes = np.shape(predictions)
#         predictions_label = []
#         for ind in range(length):
#             if predictions[ind][0] > predictions[ind][1]:
#                 predictions_label.append(255)
#             else:
#                 predictions_label.append(0)
#         predictions_label = np.array(predictions_label)
#         img_path = label_dir + "2016gf" + n_list + ".TIF"
#         img = cv2.imread(img_path)
#         img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         height, width = np.shape(img_gray)
#         predictions_img = predictions_label.reshape(height, width)
#         # cv2.imshow('predictions_img', predictions_img)
#         dir = predictions_dir + 'img\\' + n_list + '.jpg'
#         cv2.imwrite(dir, predictions_img)
#         file_num = file_num + 1
#         view_bar(file_num, len(f_list1))
# print '转化完成！'


# 以下代码用来重命名图片
predictions_dir = "C:\\Users\\qq619\\Desktop\\predictions\\img\\"
f_list1 = os.listdir(predictions_dir)
file_num = 0
for i in f_list1:
    n_list = os.path.splitext(i)[0]
    if os.path.splitext(i)[1] == '.jpg':
        img = cv2.imread(predictions_dir + i)
        dir = predictions_dir + 'renamed\\' + "2016gf" + n_list + ".jpg"
        cv2.imwrite(dir, img)
        file_num = file_num + 1
        view_bar(file_num, len(f_list1))
print '转化完成！'
