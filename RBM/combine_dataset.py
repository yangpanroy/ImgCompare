# coding=utf-8
import numpy as np
import os

# datasetR_path = "/media/files/yp/rbm/dataset/train11_R.npy"
# datasetG_path = "/media/files/yp/rbm/dataset/train11_G.npy"
# datasetB_path = "/media/files/yp/rbm/dataset/train11_B.npy"
# datasetR = np.load(datasetR_path)
# datasetG = np.load(datasetG_path)
# datasetB = np.load(datasetB_path)
# item = []
# content = []
# lenth = len(datasetR)
# for i in range(lenth):
#     item.extend(datasetR[i])
#     item.extend(datasetG[i])
#     item.extend(datasetB[i])
#     content.append(item)
#     item = []
# print np.shape(content)
# np.save("/media/files/yp/rbm/dataset/train11ms_RGB.npy", content)
# os.remove(datasetR_path)
# os.remove(datasetG_path)
# os.remove(datasetB_path)
#
# datasetR_path = "/media/files/yp/rbm/dataset/valid11_R.npy"
# datasetG_path = "/media/files/yp/rbm/dataset/valid11_G.npy"
# datasetB_path = "/media/files/yp/rbm/dataset/valid11_B.npy"
# datasetR = np.load(datasetR_path)
# datasetG = np.load(datasetG_path)
# datasetB = np.load(datasetB_path)
# item = []
# content = []
# lenth = len(datasetR)
# for i in range(lenth):
#     item.extend(datasetR[i])
#     item.extend(datasetG[i])
#     item.extend(datasetB[i])
#     content.append(item)
#     item = []
# print np.shape(content)
# np.save("/media/files/yp/rbm/dataset/valid11ms_RGB.npy", content)
# os.remove(datasetR_path)
# os.remove(datasetG_path)
# os.remove(datasetB_path)

# # TODO 将split_dataset的功能改写到这里，可以减少内存占用
# datasetR_path = "/media/files/yp/rbm/dataset/test02ms_R.npy"
# datasetG_path = "/media/files/yp/rbm/dataset/test02ms_G.npy"
# datasetB_path = "/media/files/yp/rbm/dataset/test02ms_B.npy"
# datasetR = np.load(datasetR_path)
# print 'load R dataset successfully'
# datasetG = np.load(datasetG_path)
# print 'load G dataset successfully'
# datasetB = np.load(datasetB_path)
# print 'load B dataset successfully'
# item = []
# content = []
# lenth = len(datasetR)
# for i in range(lenth):
#     item.extend(datasetR[i])
#     item.extend(datasetG[i])
#     item.extend(datasetB[i])
#     content.append(item)
#     item = []
#     if i % 1100000 == 0:
#         print i
# print np.shape(content)
# np.save("/media/files/yp/rbm/dataset/test02ms_RGB.npy", content)
# os.remove(datasetR_path)
# os.remove(datasetG_path)
# os.remove(datasetB_path) 

# 连接两个数据或标签的二进制文件
dataset1_path = "/media/files/yp/rbm/output/predictions/predictions02_test.npy"
dataset2_path = "/media/files/yp/rbm/output/predictions/predictions02_9.npy"
dataset1 = np.load(dataset1_path)
dataset2 = np.load(dataset2_path)
print np.shape(dataset1)
print np.shape(dataset2)
result = np.vstack((dataset1, dataset2))
print np.shape(result)
np.save("/media/files/yp/rbm/output/predictions/predictions02_test.npy", result)
# os.remove(dataset1_path)
# os.remove(dataset2_path)
