# coding=utf-8
import numpy as np
import tensorflow as tf
import os

from yadlt.models.autoencoders import stacked_denoising_autoencoder
from yadlt.utils import datasets, utilities

os.environ["CUDA_VISIBLE_DEVICES"] = '1'   # 指定第二块GPU可用

# #################### #
#   Flags definition   #
# #################### #
flags = tf.app.flags
FLAGS = flags.FLAGS

# 全局配置
flags.DEFINE_string('dataset', 'custom', '用哪个数据集. ["mnist", "cifar10", "custom"]')
flags.DEFINE_string('train_dataset', '/media/files/yp/rbm/fangwu/dataset/train_RGB.npy', '训练集 .npy 文件的路径.')
flags.DEFINE_string('train_labels', '/media/files/yp/rbm/fangwu/label/train_01.npy', '训练标签 .npy 文件的路径.')
flags.DEFINE_string('valid_dataset', '/media/files/yp/rbm/fangwu/dataset/valid_RGB.npy', '验证集 .npy 文件的路径.')
flags.DEFINE_string('valid_labels', '/media/files/yp/rbm/fangwu/label/valid_01.npy', '验证标签 .npy 文件的路径.')
flags.DEFINE_string('test_dataset', '/media/files/yp/rbm/fangwu/dataset/test201602_RGB.npy', '测试集 .npy 文件的路径.')
flags.DEFINE_string('test_labels', '/media/files/yp/rbm/fangwu/label/label02.npy', '测试标签 .npy 文件的路径.')
flags.DEFINE_string('cifar_dir', '', ' cifar 10 数据集目录路径.')
flags.DEFINE_boolean('do_pretrain', True, '是否使用无监督预训练网络.')
flags.DEFINE_string('save_predictions', '/media/files/yp/rbm/output/predictions/predictions201602.npy', '保存模型预测结果的 .npy '
                                                                                                  '文件的路径.')
flags.DEFINE_string('save_layers_output_test', '', '保存模型各层对测试集输出的 .npy 文件的路径.')
flags.DEFINE_string('save_layers_output_train', '', '保存模型各层对训练集输出的 .npy 文件的路径.')
flags.DEFINE_integer('seed', -1, '随机发生器的种子（> = 0）. 适用于测试超参数.')
flags.DEFINE_string('name', 'sdae_buildings', '模型的名称.')
flags.DEFINE_float('momentum', 0.5, '动量参数.')

# 有监督的微调的参数
flags.DEFINE_string('finetune_loss_func', 'softmax_cross_entropy', '损失函数. ["softmax_cross_entropy", "mse"]')
flags.DEFINE_integer('finetune_num_epochs', 30, ' epochs 数量.')
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
flags.DEFINE_string('dae_num_epochs', '10,', ' epochs 数量.')
flags.DEFINE_string('dae_batch_size', '10,', '每个 mini-batch 的大小.')
flags.DEFINE_string('dae_corr_type', 'none,', '输入干扰的类型. ["none", "masking", "salt_and_pepper"]')
flags.DEFINE_string('dae_corr_frac', '0.0,', '输入干扰的占比.')

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

if __name__ == '__main__':

    utilities.random_seed_np_tf(FLAGS.seed)

    if FLAGS.dataset == 'mnist':

        # ################# #
        #   MNIST Dataset   #
        # ################# #

        trX, trY, vlX, vlY, teX, teY = datasets.load_mnist_dataset(mode='supervised')

    elif FLAGS.dataset == 'cifar10':

        # ################### #
        #   Cifar10 Dataset   #
        # ################### #

        trX, trY, teX, teY = datasets.load_cifar10_dataset(FLAGS.cifar_dir, mode='supervised')
        # Validation set is the first half of the test set
        vlX = teX[:5000]
        vlY = teY[:5000]

    elif FLAGS.dataset == 'custom':

        # ################## #
        #   Custom Dataset   #
        # ################## #

        def load_from_np(dataset_path):
            if dataset_path != '':
                return np.load(dataset_path)
            else:
                return None


        trX, trY = load_from_np(FLAGS.train_dataset), load_from_np(FLAGS.train_labels)
        vlX, vlY = load_from_np(FLAGS.valid_dataset), load_from_np(FLAGS.valid_labels)
        teX, teY = load_from_np(FLAGS.test_dataset), load_from_np(FLAGS.test_labels)

    else:
        trX = None
        trY = None
        vlX = None
        vlY = None
        teX = None
        teY = None

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

    # 有监督微调
    sdae.fit(trX, trY, vlX, vlY)

    # 计算模型在测试集上的准确率
    print('Test set accuracy: {}'.format(sdae.score(teX, teY)))

    # 保存模型的预测
    if FLAGS.save_predictions:
        print('Saving the predictions for the test set...')
        np.save(FLAGS.save_predictions, sdae.predict(teX))


    def save_layers_output(which_set):
        if which_set == 'train':
            trout = sdae.get_layers_output(trX)
            for i, o in enumerate(trout):
                np.save(FLAGS.save_layers_output_train + '-layer-' + str(i + 1) + '-train', o)

        elif which_set == 'test':
            teout = sdae.get_layers_output(teX)
            for i, o in enumerate(teout):
                np.save(FLAGS.save_layers_output_test + '-layer-' + str(i + 1) + '-test', o)

    # 保存模型每一层对测试集的输出
    if FLAGS.save_layers_output_test:
        print('Saving the output of each layer for the test set')
        save_layers_output('test')

    # 保存模型每一层对训练集的输出
    if FLAGS.save_layers_output_train:
        print('Saving the output of each layer for the train set')
        save_layers_output('train')
