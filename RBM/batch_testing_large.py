# coding=utf-8
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from yadlt.models.autoencoders import stacked_denoising_autoencoder
from yadlt.utils import utilities
import datetime
import gc


def make_dataset(img1, img2, kernel_width=5, offset=0):
    batch = 1000000
    offset = offset * batch  # 偏移量，从偏移量开始读取本批数据，分批读取减少内存占用

    content_R = []
    content_G = []
    content_B = []
    for dim in xrange(3):
        # 分离通道 0：B 1：G 2：R
        count = 0
        img1_gray = cv2.split(img1)[dim]
        img2_gray = cv2.split(img2)[dim]

        height, width = np.shape(img1_gray)
        height2, width2 = np.shape(img2_gray)
        t = kernel_width - 1

        if (height == height2) and (width == width2):  # 检查两个图像尺寸是否一致
            # 考虑提取特征时边界像素周围可能没有像素的情况，采取补0处理
            img1_expand = np.zeros([height + t, width + t])
            img2_expand = np.zeros([height + t, width + t])
            img1_gray = img1_gray / 255.0
            img2_gray = img2_gray / 255.0
            img1_expand[t / 2:height + t / 2, t / 2:width + t / 2] = img1_gray
            img2_expand[t / 2:height + t / 2, t / 2:width + t / 2] = img2_gray
            content = []  # content用于存储特征，他的维度是[height*width, 50]
            for i in xrange(t / 2, height + t / 2):
                for j in xrange(t / 2, width + t / 2):
                    # 提取每个时相每个像素的5×5邻域像素特征，并将两时相特征拼接
                    count = count + 1
                    if count >= offset:
                        window1 = img1_expand[i - t / 2:i + t / 2 + 1, j - t / 2:j + t / 2 + 1]
                        window2 = img2_expand[i - t / 2:i + t / 2 + 1, j - t / 2:j + t / 2 + 1]
                        window1 = np.reshape(window1, (1, kernel_width * kernel_width))
                        window2 = np.reshape(window2, (1, kernel_width * kernel_width))
                        content.append(np.column_stack((window1, window2)).tolist())
                        if len(content) == batch:
                            break
                if len(content) == batch:
                    break
            if dim == 0:
                content_B = np.array(content)
            if dim == 1:
                content_G = np.array(content)
            if dim == 2:
                content_R = np.array(content)
        else:
            print "时相1和时相2的图像尺寸不匹配!"
    content_RGB = np.column_stack((content_R, content_G, content_B))
    content_RGB = np.reshape(content_RGB, (len(content), kernel_width * kernel_width * 6))
    return content_RGB


def init_flags():
    flags = tf.app.flags
    internal_FLAGS = flags.FLAGS

    # 全局配置
    flags.DEFINE_string('dataset', 'custom', '用哪个数据集. ["mnist", "cifar10", "custom"]')
    # flags.DEFINE_string('test_dataset', '/media/files/yp/rbm/pic_div/dataset/test0_RGB.npy', '测试集 .npy 文件的路径.')
    # flags.DEFINE_string('test_labels', '/media/files/yp/rbm/pic_div/label/binary/0.npy', '测试标签 .npy 文件的路径.')
    # flags.DEFINE_string('cifar_dir', '', ' cifar 10 数据集目录路径.')
    flags.DEFINE_boolean('do_pretrain', False, '是否使用无监督预训练网络.')
    # flags.DEFINE_string('save_predictions', '/media/files/yp/rbm/output/predictions/predictions0.npy',
    #                     '保存模型预测结果的 .npy 文件的路径.')
    # flags.DEFINE_string('save_layers_output_test', '', '保存模型各层对测试集输出的 .npy 文件的路径.')
    # flags.DEFINE_string('save_layers_output_train', '', '保存模型各层对训练集输出的 .npy 文件的路径.')
    flags.DEFINE_integer('seed', -1, '随机发生器的种子（> = 0）. 适用于测试超参数.')
    flags.DEFINE_string('name', 'change_detection_sdae_village_upgrade', '模型的名称.')
    flags.DEFINE_float('momentum', 0.5, '动量参数.')

    # 有监督的微调的参数
    flags.DEFINE_string('finetune_loss_func', 'softmax_cross_entropy', '损失函数. ["softmax_cross_entropy", "mse"]')
    flags.DEFINE_integer('finetune_num_epochs', 0, ' epochs 数量.')
    flags.DEFINE_float('finetune_learning_rate', 0.001, '学习率')
    flags.DEFINE_string('finetune_act_func', 'relu', '激活函数. ["sigmoid, "tanh", "relu"]')
    flags.DEFINE_float('finetune_dropout', 1, 'Dropout 参数.')
    flags.DEFINE_string('finetune_opt', 'sgd', '优化器["sgd", "adagrad", "momentum", "adam"]')
    flags.DEFINE_integer('finetune_batch_size', 20, '每个 mini-batch 的大小.')
    # 自动编码器层的特殊参数
    flags.DEFINE_string('dae_layers', '250,150,100', '将每层的节点数用逗号分隔开.')
    flags.DEFINE_string('dae_regcoef', '5e-4,', '自动编码器的正则化参数. 如果是0，没有正则化.')
    flags.DEFINE_string('dae_enc_act_func', 'sigmoid,', '编码器的激活函数. ["sigmoid", "tanh"]')
    flags.DEFINE_string('dae_dec_act_func', 'none,', '解码器的激活函数. ["sigmoid", "tanh", "none"]')
    flags.DEFINE_string('dae_loss_func', 'mse,', '损失函数. ["mse" or "cross_entropy"]')
    flags.DEFINE_string('dae_opt', 'sgd,', '优化器["sgd", "ada_grad", "momentum", "adam"]')
    flags.DEFINE_string('dae_learning_rate', '0.01,', '初试学习率.')
    flags.DEFINE_string('dae_num_epochs', '0,', ' epochs 数量.')
    flags.DEFINE_string('dae_batch_size', '10,', '每个 mini-batch 的大小.')
    flags.DEFINE_string('dae_corr_type', 'none,', '输入干扰的类型. ["none", "masking", "salt_and_pepper"]')
    flags.DEFINE_string('dae_corr_frac', '0.0,', '输入干扰的占比.')
    return internal_FLAGS


def do_predict(internal_FLAGS, internal_dataset):
    # 将自动编码器层参数从字符串转换为其特定类型
    dae_layers = utilities.flag_to_list(internal_FLAGS.dae_layers, 'int')
    dae_enc_act_func = utilities.flag_to_list(internal_FLAGS.dae_enc_act_func, 'str')
    dae_dec_act_func = utilities.flag_to_list(internal_FLAGS.dae_dec_act_func, 'str')
    dae_opt = utilities.flag_to_list(internal_FLAGS.dae_opt, 'str')
    dae_loss_func = utilities.flag_to_list(internal_FLAGS.dae_loss_func, 'str')
    dae_learning_rate = utilities.flag_to_list(internal_FLAGS.dae_learning_rate, 'float')
    dae_regcoef = utilities.flag_to_list(internal_FLAGS.dae_regcoef, 'float')
    dae_corr_type = utilities.flag_to_list(internal_FLAGS.dae_corr_type, 'str')
    dae_corr_frac = utilities.flag_to_list(internal_FLAGS.dae_corr_frac, 'float')
    dae_num_epochs = utilities.flag_to_list(internal_FLAGS.dae_num_epochs, 'int')
    dae_batch_size = utilities.flag_to_list(internal_FLAGS.dae_batch_size, 'int')

    # 检查参数
    assert all([0. <= cf <= 1. for cf in dae_corr_frac])
    assert all([ct in ['masking', 'salt_and_pepper', 'none'] for ct in dae_corr_type])
    assert internal_FLAGS.dataset in ['mnist', 'cifar10', 'custom']
    assert len(dae_layers) > 0
    assert all([af in ['sigmoid', 'tanh'] for af in dae_enc_act_func])
    assert all([af in ['sigmoid', 'tanh', 'none'] for af in dae_dec_act_func])

    utilities.random_seed_np_tf(internal_FLAGS.seed)

    # 创建编码、解码、微调函数和网络模型对象
    sdae = None

    dae_enc_act_func = [utilities.str2actfunc(af) for af in dae_enc_act_func]
    dae_dec_act_func = [utilities.str2actfunc(af) for af in dae_dec_act_func]
    finetune_act_func = utilities.str2actfunc(internal_FLAGS.finetune_act_func)

    sdae = stacked_denoising_autoencoder.StackedDenoisingAutoencoder(
        do_pretrain=internal_FLAGS.do_pretrain, name=internal_FLAGS.name,
        layers=dae_layers, finetune_loss_func=internal_FLAGS.finetune_loss_func,
        finetune_learning_rate=internal_FLAGS.finetune_learning_rate,
        finetune_num_epochs=internal_FLAGS.finetune_num_epochs,
        finetune_opt=internal_FLAGS.finetune_opt, finetune_batch_size=internal_FLAGS.finetune_batch_size,
        finetune_dropout=internal_FLAGS.finetune_dropout,
        enc_act_func=dae_enc_act_func, dec_act_func=dae_dec_act_func,
        corr_type=dae_corr_type, corr_frac=dae_corr_frac, regcoef=dae_regcoef,
        loss_func=dae_loss_func, opt=dae_opt,
        learning_rate=dae_learning_rate, momentum=internal_FLAGS.momentum,
        num_epochs=dae_num_epochs, batch_size=dae_batch_size,
        finetune_act_func=finetune_act_func)

    # 训练模型 (无监督预训练)
    if internal_FLAGS.do_pretrain:
        encoded_X, encoded_vX = sdae.pretrain(trX, vlX)

    teX = internal_dataset
    print('Saving the predictions for the test set...')
    internal_predictions = sdae.predict(teX)
    return internal_predictions


def read_predictions(internal_img2_dir, internal_predictions):
    length, classes = np.shape(internal_predictions)
    predictions_label = []
    for ind in xrange(length):
        if internal_predictions[ind][0] > internal_predictions[ind][1]:
            predictions_label.append(255)
        else:
            predictions_label.append(0)
    predictions_label = np.array(predictions_label)
    img = cv2.imread(internal_img2_dir)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = np.shape(img_gray)
    internal_predictions_img = predictions_label.reshape(height, width)
    return internal_predictions_img


def view_bar(num, total):
    rate = num * 1.0 / total
    rate_num = int(rate * 100)
    r = '\r[%s%s]%d%%' % ("*" * rate_num, "-" * (100 - rate_num), rate_num,)
    sys.stdout.write(r)
    sys.stdout.flush()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # 指定第二块GPU可用
    os.system('ulimit -n 8192')
    kernel_width = 5  # 定义滑块的边长
    # img1_dir = '/media/files/yp/rbm/yinchuansanqu/2015/'  # 时相1图像的路径
    # img2_dir = '/media/files/yp/rbm/yinchuansanqu/201702/sanqu/'  # 时相2图像的路径
    # predictions_dir = '/media/files/yp/rbm/pic_div/predictions/img/'  # 存/放预测结果的路径
    img1_dir = '//media/files/yp/rbm/13qu/2015/'  # 时相1图像的路径
    img2_dir = '//media/files/yp/rbm/13qu/2017/'  # 时相2图像的路径
    predictions_dir = '/media/files/yp/rbm/13qu/prediction/'  # 存/放预测结果的路径

    FLAGS = init_flags()

    f_list1 = os.listdir(img1_dir)
    f_list2 = os.listdir(img2_dir)
    f_list3 = os.listdir(predictions_dir)
    file_num = 0
    start = datetime.datetime.now()
    for file_name in f_list1:
        if os.path.splitext(file_name)[1] == '.TIF':
            file_id = file_name.split("gf")[-1]
            another_file_name = "201702gf" + file_id
            third_name = another_file_name.split('.')[0] + '.jpg'
            if another_file_name in f_list2 and third_name not in f_list3:
                img1_path = img1_dir + file_name
                img2_path = img2_dir + another_file_name
                print "\n==============开始制作dataset============="
                img1 = cv2.imread(img1_path)
                img2 = cv2.imread(img2_path)
                # 防止尺寸不匹配
                row1, col1, dim1 = img1.shape
                row2, col2, dim2 = img2.shape
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

                row, col, dim = img1.shape
                step = (row * col) / 1000000
                res = (row * col) % 1000000
                if res != 0:
                    step = step + 1
                for i in range(step):
                    print "开始制作第{0}/{1}批数据".format((i+1), step)
                    dataset = make_dataset(img1, img2, kernel_width, offset=i)
                    print "==============开始读取模型并批量进行预测============="
                    predictions = do_predict(FLAGS, dataset)
                    del dataset
                    predictions = predictions.astype(np.float16)
                    np.save("/media/files/yp/rbm/13qu/prediction/npy/" + str(i) + '.npy', predictions)
                if step == 1:
                    print "==============将二进制预测结果转化为图像============="
                    predictions_img = read_predictions(img2_path, predictions)
                    del predictions
                    cv2.imwrite(predictions_dir + another_file_name.split('.')[0] + '.jpg', predictions_img)
                    del predictions_img
                    os.remove("/media/files/yp/rbm/13qu/prediction/npy/0.npy")
                elif step >= 2:
                    npy1 = np.load("/media/files/yp/rbm/13qu/prediction/npy/0.npy")
                    npy2 = np.load("/media/files/yp/rbm/13qu/prediction/npy/1.npy")
                    combineNpy = np.row_stack((npy1, npy2))
                    del npy1, npy2
                    os.remove("/media/files/yp/rbm/13qu/prediction/npy/0.npy")
                    os.remove("/media/files/yp/rbm/13qu/prediction/npy/1.npy")
                    for i in range(2, step):
                        print(i)
                        npy = np.load("/media/files/yp/rbm/13qu/prediction/npy/" + str(i) + ".npy")
                        combineNpy = np.row_stack((combineNpy, npy))
                        del npy
                        os.remove("/media/files/yp/rbm/13qu/prediction/npy/" + str(i) + ".npy")
                    np.save("/media/files/yp/rbm/13qu/prediction/npy/changeDetection.npy",
                            np.array(combineNpy).astype(np.float16))
                    print "==============将二进制预测结果转化为图像============="
                    predictions_img = read_predictions(img2_path, np.array(combineNpy).astype(np.float16))
                    del predictions
                    os.remove("/media/files/yp/rbm/13qu/prediction/npy/changeDetection.npy")
                    cv2.imwrite(predictions_dir + another_file_name.split('.')[0] + '.jpg', predictions_img)
                    del predictions_img

                file_num = file_num + 1
                end = datetime.datetime.now()
                print "已用时间：{0}秒    预计时间：{1}秒    剩余时间：{2}秒".format((end - start).seconds,
                                                                   (end - start).seconds / file_num * len(f_list2),
                                                                   (end - start).seconds / file_num * len(f_list2) - (
                                                                               end - start).seconds)
                view_bar(file_num, len(f_list2))
                gc.collect()
