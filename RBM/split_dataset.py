# coding=utf-8
import numpy as np

dataset_path = "/media/files/yp/rbm/dataset/test02ms_RGB.npy"
label_path = "/media/files/yp/rbm/label/binary/label02ms.npy"

dataset = np.load(dataset_path)
label = np.load(label_path)

batches_num = 10

if len(dataset) == len(label):
    batches_size = len(dataset) / batches_num + 1
    for i in range(batches_num):
        if (batches_size * (i + 1)) < len(dataset):
            split_dataset = dataset[batches_size * i: batches_size * (i + 1)]
            split_label = label[batches_size * i: batches_size * (i + 1)]
        else:
            split_dataset = dataset[batches_size * i:]
            split_label = label[batches_size * i:]
        np.save("/media/files/yp/rbm/dataset/test02_ms/test02ms_" + str(i) + "_RGB.npy", split_dataset)
        np.save("/media/files/yp/rbm/label/binary/label02ms_" + str(i) + ".npy", split_label)
        print np.shape(split_dataset)
    print '分割完成！'

else:
    print '数据和标签不匹配！'
