# coding=utf-8
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from yadlt.models.autoencoders import stacked_denoising_autoencoder
from yadlt.utils import utilities


def view_bar(num, total):
    rate = num * 1.0 / total
    rate_num = int(rate * 100)
    r = '\r[%s%s]%d%%' % ("*" * rate_num, "-" * (100 - rate_num), rate_num,)
    sys.stdout.write(r)
    sys.stdout.flush()


def num_in_path(path):
    # 返回所给路径下的图像数量
    files_num = 0
    f_list = os.listdir(path)
    for i in f_list:
        if os.path.splitext(i)[1] == '.TIF':
            files_num = files_num + 1
    return files_num


def check_num_in_path(path1, path2):
    # 检查所给路径下的图像数量是否一致，若一致，则返回图像数量
    if path1 != '' and path2 != '':
        if num_in_path(path1) == num_in_path(path2):
            return num_in_path(path1)
        else:
            return 0
    else:
        return 0


def make_label(label_dir, files_num):
    # 根据所给的label图像路径，批量将label图像转换成二进制标签文件
    for idx in range(0, files_num):
        label_img_path = label_dir + str(idx) + '.png'
        img = cv2.imread(label_img_path)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        label = []
        height, width = np.shape(img_gray)
        for i in range(height):
            for j in range(width):
                if img_gray[i][j] == 0:  # 黑色的地方表示没有变化
                    label.append([0, 1])
                else:  # 白色的地方表示有变化
                    label.append([1, 0])
        if not os.path.exists(label_dir + "binary/"):  # 判断binary文件夹是否存在
            os.makedirs(label_dir + "binary/")  # 若没有，在label图像路径下创建一个文件夹
        np.save(label_dir + "binary/" + str(idx) + ".npy", label)  # 保存二进制标签文件
        print label_dir + "binary/" + str(idx) + ".npy  已经成功生成！"


def make_dataset(img1_dir, img2_dir, files_num):
    # 根据所给的两时相图像路径，提取每个像素周围5×5的邻域矩阵作为特征，生成二进制数据文件
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
                # 防止尺寸不匹配
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

                content_R = []
                content_G = []
                content_B = []
                for dim in range(3):
                    # 分离通道 0：B 1：G 2：R
                    img1_gray = cv2.split(img1)[dim]
                    img2_gray = cv2.split(img2)[dim]

                    height, width = np.shape(img1_gray)
                    height2, width2 = np.shape(img2_gray)
                    kernel_width = 5  # 定义滑块的边长
                    t = kernel_width - 1

                    if (height == height2) and (width == width2):  # 检查两个图像尺寸是否一致
                        # 考虑提取特征时边界像素周围可能没有像素的情况，采取补0处理
                        img1_expand = np.zeros([height + t, width + t])
                        img2_expand = np.zeros([height + t, width + t])
                        for i in range(height):
                            for j in range(width):
                                # 归一化到[0,1]区间
                                img1_expand[i + t / 2][j + t / 2] = img1_gray[i][j] / 255.0
                                img2_expand[i + t / 2][j + t / 2] = img2_gray[i][j] / 255.0
                        content = []  # content用于存储特征，他的维度是[height*width, 50]
                        for i in range(t / 2, height + t / 2):
                            for j in range(t / 2, width + t / 2):
                                # 提取每个时相每个像素的5×5邻域像素特征，并将两时相特征拼接
                                content.append(
                                    [
                                        img1_expand[i - 2][j - 2], img1_expand[i - 2][j - 1], img1_expand[i - 2][j],
                                        img1_expand[i - 2][j + 1], img1_expand[i - 2][j + 2],
                                        img1_expand[i - 1][j - 2], img1_expand[i - 1][j - 1], img1_expand[i - 1][j],
                                        img1_expand[i - 1][j + 1], img1_expand[i - 1][j + 2],
                                        img1_expand[i][j - 2], img1_expand[i][j - 1], img1_expand[i][j],
                                        img1_expand[i][j + 1],
                                        img1_expand[i][j + 2],
                                        img1_expand[i + 1][j - 2], img1_expand[i + 1][j - 1], img1_expand[i + 1][j],
                                        img1_expand[i + 1][j + 1], img1_expand[i + 1][j + 2],
                                        img1_expand[i + 2][j - 2], img1_expand[i + 2][j - 1], img1_expand[i + 2][j],
                                        img1_expand[i + 2][j + 1], img1_expand[i + 2][j + 2],
                                        img2_expand[i - 2][j - 2], img2_expand[i - 2][j - 1], img2_expand[i - 2][j],
                                        img2_expand[i - 2][j + 1], img2_expand[i - 2][j + 2],
                                        img2_expand[i - 1][j - 2], img2_expand[i - 1][j - 1], img2_expand[i - 1][j],
                                        img2_expand[i - 1][j + 1], img2_expand[i - 1][j + 2],
                                        img2_expand[i][j - 2], img2_expand[i][j - 1], img2_expand[i][j],
                                        img2_expand[i][j + 1],
                                        img2_expand[i][j + 2],
                                        img2_expand[i + 1][j - 2], img2_expand[i + 1][j - 1], img2_expand[i + 1][j],
                                        img2_expand[i + 1][j + 1], img2_expand[i + 1][j + 2],
                                        img2_expand[i + 2][j - 2], img2_expand[i + 2][j - 1], img2_expand[i + 2][j],
                                        img2_expand[i + 2][j + 1], img2_expand[i + 2][j + 2]
                                    ])
                        if dim == 0:
                            content_B = content
                        if dim == 1:
                            content_G = content
                        if dim == 2:
                            content_R = content
                    else:
                        print "时相1和时相2的图像尺寸不匹配!"
                content_RGB = np.column_stack((content_R, content_G, content_B))
                # print content_RGB.dtype
                content_RGB = content_RGB.astype(np.float16)
                # print content_RGB.dtype
                # TODO 路径并非程序建立，需在后面补充代码
                np.save("/media/files/yp/rbm/yinchuansanqu/dataset/" + file_id + "_RGB.npy", content_RGB)
                # print "/media/files/yp/rbm/yinchuansanqu/dataset/" + file_id + "_RGB.npy  已经生成！"
                file_num = file_num + 1
                view_bar(file_num, len(f_list1))
    # for idx in range(0, files_num):
    #     img1_path = img1_dir + str(idx) + '.tif'
    #     img2_path = img2_dir + str(idx) + '.tif'
    #     img1 = cv2.imread(img1_path)
    #     img2 = cv2.imread(img2_path)
    #     content_R = []
    #     content_G = []
    #     content_B = []
    #     for dim in range(3):
    #         # 分离通道 0：B 1：G 2：R
    #         img1_gray = cv2.split(img1)[dim]
    #         img2_gray = cv2.split(img2)[dim]
    #
    #         height, width = np.shape(img1_gray)
    #         height2, width2 = np.shape(img2_gray)
    #         kernel_width = 5  # 定义滑块的边长
    #         t = kernel_width - 1
    #
    #         if (height == height2) and (width == width2):  # 检查两个图像尺寸是否一致
    #             # 考虑提取特征时边界像素周围可能没有像素的情况，采取补0处理
    #             img1_expand = np.zeros([height + t, width + t])
    #             img2_expand = np.zeros([height + t, width + t])
    #             for i in range(height):
    #                 for j in range(width):
    #                     # 归一化到[0,1]区间
    #                     img1_expand[i + t / 2][j + t / 2] = img1_gray[i][j] / 255.0
    #                     img2_expand[i + t / 2][j + t / 2] = img2_gray[i][j] / 255.0
    #             content = []  # content用于存储特征，他的维度是[height*width, 50]
    #             for i in range(t / 2, height + t / 2):
    #                 for j in range(t / 2, width + t / 2):
    #                     # 提取每个时相每个像素的5×5邻域像素特征，并将两时相特征拼接
    #                     content.append(
    #                         [
    #                             img1_expand[i - 2][j - 2], img1_expand[i - 2][j - 1], img1_expand[i - 2][j],
    #                             img1_expand[i - 2][j + 1], img1_expand[i - 2][j + 2],
    #                             img1_expand[i - 1][j - 2], img1_expand[i - 1][j - 1], img1_expand[i - 1][j],
    #                             img1_expand[i - 1][j + 1], img1_expand[i - 1][j + 2],
    #                             img1_expand[i][j - 2], img1_expand[i][j - 1], img1_expand[i][j], img1_expand[i][j + 1],
    #                             img1_expand[i][j + 2],
    #                             img1_expand[i + 1][j - 2], img1_expand[i + 1][j - 1], img1_expand[i + 1][j],
    #                             img1_expand[i + 1][j + 1], img1_expand[i + 1][j + 2],
    #                             img1_expand[i + 2][j - 2], img1_expand[i + 2][j - 1], img1_expand[i + 2][j],
    #                             img1_expand[i + 2][j + 1], img1_expand[i + 2][j + 2],
    #                             img2_expand[i - 2][j - 2], img2_expand[i - 2][j - 1], img2_expand[i - 2][j],
    #                             img2_expand[i - 2][j + 1], img2_expand[i - 2][j + 2],
    #                             img2_expand[i - 1][j - 2], img2_expand[i - 1][j - 1], img2_expand[i - 1][j],
    #                             img2_expand[i - 1][j + 1], img2_expand[i - 1][j + 2],
    #                             img2_expand[i][j - 2], img2_expand[i][j - 1], img2_expand[i][j], img2_expand[i][j + 1],
    #                             img2_expand[i][j + 2],
    #                             img2_expand[i + 1][j - 2], img2_expand[i + 1][j - 1], img2_expand[i + 1][j],
    #                             img2_expand[i + 1][j + 1], img2_expand[i + 1][j + 2],
    #                             img2_expand[i + 2][j - 2], img2_expand[i + 2][j - 1], img2_expand[i + 2][j],
    #                             img2_expand[i + 2][j + 1], img2_expand[i + 2][j + 2]
    #                         ])
    #             if dim == 0:
    #                 content_B = content
    #             if dim == 1:
    #                 content_G = content
    #             if dim == 2:
    #                 content_R = content
    #         else:
    #             print "时相1和时相2的图像尺寸不匹配!"
    #     content_RGB = np.column_stack((content_R, content_G, content_B))
    #     # TODO 路径并非程序建立，需在后面补充代码
    #     np.save("/media/files/yp/rbm/pic_div/dataset/test" + str(idx) + "_RGB.npy", content_RGB)
    #     print "/media/files/yp/rbm/pic_div/dataset/test" + str(idx) + "_RGB.npy  已经生成！"


def init_flags():
    flags = tf.app.flags
    FLAGS = flags.FLAGS

    # 全局配置
    flags.DEFINE_string('dataset', 'custom', '用哪个数据集. ["mnist", "cifar10", "custom"]')
    flags.DEFINE_string('test_dataset', '/media/files/yp/rbm/pic_div/dataset/test0_RGB.npy', '测试集 .npy 文件的路径.')
    flags.DEFINE_string('test_labels', '/media/files/yp/rbm/pic_div/label/binary/0.npy', '测试标签 .npy 文件的路径.')
    flags.DEFINE_string('cifar_dir', '', ' cifar 10 数据集目录路径.')
    flags.DEFINE_boolean('do_pretrain', False, '是否使用无监督预训练网络.')
    flags.DEFINE_string('save_predictions', '/media/files/yp/rbm/output/predictions/predictions0.npy',
                        '保存模型预测结果的 .npy 文件的路径.')
    flags.DEFINE_string('save_layers_output_test', '', '保存模型各层对测试集输出的 .npy 文件的路径.')
    flags.DEFINE_string('save_layers_output_train', '', '保存模型各层对训练集输出的 .npy 文件的路径.')
    flags.DEFINE_integer('seed', -1, '随机发生器的种子（> = 0）. 适用于测试超参数.')
    flags.DEFINE_string('name', 'change_detection_sdae', '模型的名称.')
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
    return FLAGS


def do_predict(FLAGS, predictions_dir, num):
    dataset_path = "/media/files/yp/rbm/yinchuansanqu/dataset/"
    f_list1 = os.listdir(dataset_path)
    for i in f_list1:
        if os.path.splitext(i)[1] == '.npy':
            # 将自动编码器层参数从字符串转换为其特定类型
            dae_layers = utilities.flag_to_list(FLAGS.dae_layers, 'int')
            dae_enc_act_func = utilities.flag_to_list(FLAGS.dae_enc_act_func, 'str')
            dae_dec_act_func = utilities.flag_to_list(FLAGS.dae_dec_act_func, 'str')
            dae_opt = utilities.flag_to_list(FLAGS.dae_opt, 'str')
            dae_loss_func = utilities.flag_to_list(FLAGS.dae_loss_func, 'str')
            dae_learning_rate = utilities.flag_to_list(FLAGS.dae_learning_rate, 'float')
            dae_regcoef = utilities.flag_to_list(FLAGS.dae_regcoef, 'float')
            dae_corr_type = utilities.flag_to_list(FLAGS.dae_corr_type, 'str')
            dae_corr_frac = utilities.flag_to_list(FLAGS.dae_corr_frac, 'float')
            dae_num_epochs = utilities.flag_to_list(FLAGS.dae_num_epochs, 'int')
            dae_batch_size = utilities.flag_to_list(FLAGS.dae_batch_size, 'int')

            # 检查参数
            assert all([0. <= cf <= 1. for cf in dae_corr_frac])
            assert all([ct in ['masking', 'salt_and_pepper', 'none'] for ct in dae_corr_type])
            assert FLAGS.dataset in ['mnist', 'cifar10', 'custom']
            assert len(dae_layers) > 0
            assert all([af in ['sigmoid', 'tanh'] for af in dae_enc_act_func])
            assert all([af in ['sigmoid', 'tanh', 'none'] for af in dae_dec_act_func])

            utilities.random_seed_np_tf(FLAGS.seed)

            def load_from_np(dataset_path):
                if dataset_path != '':
                    return np.load(dataset_path)
                else:
                    return None

            # 创建编码、解码、微调函数和网络模型对象
            sdae = None

            dae_enc_act_func = [utilities.str2actfunc(af) for af in dae_enc_act_func]
            dae_dec_act_func = [utilities.str2actfunc(af) for af in dae_dec_act_func]
            finetune_act_func = utilities.str2actfunc(FLAGS.finetune_act_func)

            sdae = stacked_denoising_autoencoder.StackedDenoisingAutoencoder(
                do_pretrain=FLAGS.do_pretrain, name=FLAGS.name,
                layers=dae_layers, finetune_loss_func=FLAGS.finetune_loss_func,
                finetune_learning_rate=FLAGS.finetune_learning_rate, finetune_num_epochs=FLAGS.finetune_num_epochs,
                finetune_opt=FLAGS.finetune_opt, finetune_batch_size=FLAGS.finetune_batch_size,
                finetune_dropout=FLAGS.finetune_dropout,
                enc_act_func=dae_enc_act_func, dec_act_func=dae_dec_act_func,
                corr_type=dae_corr_type, corr_frac=dae_corr_frac, regcoef=dae_regcoef,
                loss_func=dae_loss_func, opt=dae_opt,
                learning_rate=dae_learning_rate, momentum=FLAGS.momentum,
                num_epochs=dae_num_epochs, batch_size=dae_batch_size,
                finetune_act_func=finetune_act_func)

            # 训练模型 (无监督预训练)
            if FLAGS.do_pretrain:
                encoded_X, encoded_vX = sdae.pretrain(trX, vlX)

            FLAGS.test_dataset = dataset_path + i
            # FLAGS.test_labels = "/media/files/yp/rbm/pic_div/label/binary/" + str(idx) + ".npy"
            FLAGS.save_predictions = predictions_dir + i
            # teX, teY = load_from_np(FLAGS.test_dataset), load_from_np(FLAGS.test_labels)
            teX = load_from_np(FLAGS.test_dataset)
            # 计算模型在测试集上的准确率
            # print('Test set accuracy: {}'.format(sdae.score(teX, teY)))

            # 保存模型的预测
            if FLAGS.save_predictions:
                print('Saving the predictions for the test set...')
                predict = sdae.predict(teX).astype(np.float16)
                np.save(FLAGS.save_predictions, predict)

    # for idx in range(0, num):
    #     # 将自动编码器层参数从字符串转换为其特定类型
    #     dae_layers = utilities.flag_to_list(FLAGS.dae_layers, 'int')
    #     dae_enc_act_func = utilities.flag_to_list(FLAGS.dae_enc_act_func, 'str')
    #     dae_dec_act_func = utilities.flag_to_list(FLAGS.dae_dec_act_func, 'str')
    #     dae_opt = utilities.flag_to_list(FLAGS.dae_opt, 'str')
    #     dae_loss_func = utilities.flag_to_list(FLAGS.dae_loss_func, 'str')
    #     dae_learning_rate = utilities.flag_to_list(FLAGS.dae_learning_rate, 'float')
    #     dae_regcoef = utilities.flag_to_list(FLAGS.dae_regcoef, 'float')
    #     dae_corr_type = utilities.flag_to_list(FLAGS.dae_corr_type, 'str')
    #     dae_corr_frac = utilities.flag_to_list(FLAGS.dae_corr_frac, 'float')
    #     dae_num_epochs = utilities.flag_to_list(FLAGS.dae_num_epochs, 'int')
    #     dae_batch_size = utilities.flag_to_list(FLAGS.dae_batch_size, 'int')
    #
    #     # 检查参数
    #     assert all([0. <= cf <= 1. for cf in dae_corr_frac])
    #     assert all([ct in ['masking', 'salt_and_pepper', 'none'] for ct in dae_corr_type])
    #     assert FLAGS.dataset in ['mnist', 'cifar10', 'custom']
    #     assert len(dae_layers) > 0
    #     assert all([af in ['sigmoid', 'tanh'] for af in dae_enc_act_func])
    #     assert all([af in ['sigmoid', 'tanh', 'none'] for af in dae_dec_act_func])
    #
    #     utilities.random_seed_np_tf(FLAGS.seed)
    #
    #     def load_from_np(dataset_path):
    #         if dataset_path != '':
    #             return np.load(dataset_path)
    #         else:
    #             return None
    #
    #     # 创建编码、解码、微调函数和网络模型对象
    #     sdae = None
    #
    #     dae_enc_act_func = [utilities.str2actfunc(af) for af in dae_enc_act_func]
    #     dae_dec_act_func = [utilities.str2actfunc(af) for af in dae_dec_act_func]
    #     finetune_act_func = utilities.str2actfunc(FLAGS.finetune_act_func)
    #
    #     sdae = stacked_denoising_autoencoder.StackedDenoisingAutoencoder(
    #         do_pretrain=FLAGS.do_pretrain, name=FLAGS.name,
    #         layers=dae_layers, finetune_loss_func=FLAGS.finetune_loss_func,
    #         finetune_learning_rate=FLAGS.finetune_learning_rate, finetune_num_epochs=FLAGS.finetune_num_epochs,
    #         finetune_opt=FLAGS.finetune_opt, finetune_batch_size=FLAGS.finetune_batch_size,
    #         finetune_dropout=FLAGS.finetune_dropout,
    #         enc_act_func=dae_enc_act_func, dec_act_func=dae_dec_act_func,
    #         corr_type=dae_corr_type, corr_frac=dae_corr_frac, regcoef=dae_regcoef,
    #         loss_func=dae_loss_func, opt=dae_opt,
    #         learning_rate=dae_learning_rate, momentum=FLAGS.momentum,
    #         num_epochs=dae_num_epochs, batch_size=dae_batch_size,
    #         finetune_act_func=finetune_act_func)
    #
    #     # 训练模型 (无监督预训练)
    #     if FLAGS.do_pretrain:
    #         encoded_X, encoded_vX = sdae.pretrain(trX, vlX)
    #
    #     FLAGS.test_dataset = "/media/files/yp/rbm/pic_div/dataset/test" + str(idx) + "_RGB.npy"
    #     # FLAGS.test_labels = "/media/files/yp/rbm/pic_div/label/binary/" + str(idx) + ".npy"
    #     FLAGS.save_predictions = predictions_dir + str(idx) + ".npy"
    #     # teX, teY = load_from_np(FLAGS.test_dataset), load_from_np(FLAGS.test_labels)
    #     teX = load_from_np(FLAGS.test_dataset)
    #     # 计算模型在测试集上的准确率
    #     # print('Test set accuracy: {}'.format(sdae.score(teX, teY)))
    #
    #     # 保存模型的预测
    #     if FLAGS.save_predictions:
    #         print('Saving the predictions for the test set...')
    #         np.save(FLAGS.save_predictions, sdae.predict(teX))
    #
    #     def save_layers_output(which_set):
    #
    #         if which_set == 'test':
    #             teout = sdae.get_layers_output(teX)
    #             for i, o in enumerate(teout):
    #                 np.save(FLAGS.save_layers_output_test + '-layer-' + str(i + 1) + '-test', o)
    #
    #     # 保存模型每一层对测试集的输出
    #     if FLAGS.save_layers_output_test:
    #         print('Saving the output of each layer for the test set')
    #         save_layers_output('test')
    #
    #     # 保存模型每一层对训练集的输出
    #     if FLAGS.save_layers_output_train:
    #         print('Saving the output of each layer for the train set')
    #         save_layers_output('train')
    #
    print '-------------------------------------------预测过程已经完成---------------------------------------------------'


def read_predictions(label_dir, predictions_dir, files_num):
    f_list1 = os.listdir(predictions_dir)
    file_num = 0
    for i in f_list1:
        file_id = os.path.splitext(i)[0].split('.')[0]
        if os.path.splitext(i)[1] == '.npy':
            predictions = np.load(predictions_dir + i)
            # predictions = np.load("/media/files/yp/rbm/predictions.npy")
            length, classes = np.shape(predictions)
            predictions_label = []
            for ind in range(length):
                if predictions[ind][0] > predictions[ind][1]:
                    predictions_label.append(255)
                else:
                    predictions_label.append(0)
            predictions_label = np.array(predictions_label)
            img_path = label_dir + "2016gf" + file_id + '.TIF'
            img = cv2.imread(img_path)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = np.shape(img_gray)
            predictions_img = predictions_label.reshape(height, width)
            # cv2.imshow('predictions_img', predictions_img)
            cv2.imwrite(predictions_dir + 'img/' + file_id + '.jpg', predictions_img)
            file_num = file_num + 1
            view_bar(file_num, len(f_list1))

    # for idx in range(0, files_num):
    #     predictions = np.load(predictions_dir + str(idx) + ".npy")
    #     # predictions = np.load("/media/files/yp/rbm/predictions.npy")
    #     length, classes = np.shape(predictions)
    #     predictions_label = []
    #     for i in range(length):
    #         if predictions[i][0] > predictions[i][1]:
    #             predictions_label.append(255)
    #         else:
    #             predictions_label.append(0)
    #     predictions_label = np.array(predictions_label)
    #     img_path = label_dir + str(idx) + '.png'
    #     img = cv2.imread(img_path)
    #     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     height, width = np.shape(img_gray)
    #     predictions_img = predictions_label.reshape(height, width)
    #     # cv2.imshow('predictions_img', predictions_img)
    #     cv2.imwrite(predictions_dir + 'img/' + str(idx) + '.jpg', predictions_img)
    print '转化完成！'


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # 指定第二块GPU可用
    os.system('ulimit -n 2048')

    img1_dir = '/media/files/yp/rbm/yinchuansanqu/2015/'  # 时相1图像的路径
    img2_dir = '/media/files/yp/rbm/yinchuansanqu/2016/'  # 时相2图像的路径
    # label_dir = '/media/files/yp/rbm/pic_div/label/'  # 标签图像的路径
    predictions_dir = '/media/files/yp/rbm/yinchuansanqu/predictions/'  # 存/放预测结果的路径
    files_num = check_num_in_path(img1_dir, img2_dir)
    if files_num != 0:
        # 如果三个路径下的 png 图像数量匹配的话，则制作标签图像的二进制文件
        # print "==============开始制作label============="
        # make_label(label_dir, files_num)
        print "==============开始制作dataset============="
        make_dataset(img1_dir, img2_dir, files_num)
        print "==============开始读取模型并批量进行预测============="
        FLAGS = init_flags()
        do_predict(FLAGS, predictions_dir, files_num)
        print "==============将二进制预测结果转化为图像============="
        read_predictions(img2_dir, predictions_dir, files_num)
    else:
        print "给出的路径下图片数量不匹配，请检查！"
