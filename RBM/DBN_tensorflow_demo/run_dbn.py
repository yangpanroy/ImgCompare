# --*-- coding=utf-8 --*--
import numpy as np
import tensorflow as tf

from yadlt.models.boltzmann import dbn
from yadlt.utils import datasets, utilities

# #################### #
#   Flags definition   #
# #################### #
flags = tf.app.flags
FLAGS = flags.FLAGS

# 全局配置
flags.DEFINE_string('dataset', 'custom', '用哪个数据集. ["mnist", "cifar10", "custom"]')
flags.DEFINE_string('train_dataset', '/media/files/yp/rbm/train03.npy', '训练集 .npy 文件的路径.')
flags.DEFINE_string('train_labels', '/media/files/yp/rbm/train_label03.npy', '训练标签 .npy 文件的路径.')
flags.DEFINE_string('valid_dataset', '/media/files/yp/rbm/valid03.npy', '验证集 .npy 文件的路径.')
flags.DEFINE_string('valid_labels', '/media/files/yp/rbm/valid_label03.npy', '验证标签 .npy 文件的路径.')
flags.DEFINE_string('test_dataset', '/media/files/yp/rbm/dataset03.npy', '测试集 .npy 文件的路径.')
flags.DEFINE_string('test_labels', '/media/files/yp/rbm/label03.npy', '测试标签 .npy 文件的路径.')
flags.DEFINE_string('cifar_dir', '', ' cifar 10 数据集目录路径.')
flags.DEFINE_string('name', 'dbn', '模型的名称.')
flags.DEFINE_string('save_predictions', '/media/files/yp/rbm/predictions.npy', '保存模型预测结果的 .npy 文件的路径.')
flags.DEFINE_string('save_layers_output_test', '/media/files/yp/rbm/output/layers_output/', '保存模型各层对测试集输出的 .npy 文件的路径.')
flags.DEFINE_string('save_layers_output_train', '/media/files/yp/rbm/output/layers_output/', '保存模型各层对训练集输出的 .npy 文件的路径.')
flags.DEFINE_boolean('do_pretrain', True, '是否预训练网络.')
flags.DEFINE_integer('seed', -1, '随机发生器的种子（> = 0）。 适用于测试超参数.')
flags.DEFINE_float('momentum', 0.5, '动量参数.')

# RBMs层具体参数
flags.DEFINE_string('rbm_layers', '250,150,100', '将每层的节点数用逗号分隔开.')
flags.DEFINE_boolean('rbm_gauss_visible', False, '是否将高斯单元用于可见层.')
flags.DEFINE_float('rbm_stddev', 0.1, '高斯可见单元的标准差.')
flags.DEFINE_string('rbm_learning_rate', '0.001,', '初始学习率.')
flags.DEFINE_string('rbm_num_epochs', '10,', ' epochs 数量.')
flags.DEFINE_string('rbm_batch_size', '32,', '每个 mini-batch 的大小.')
flags.DEFINE_string('rbm_gibbs_k', '1,', '吉布斯采样步长.')

# 有监督的微调的参数
flags.DEFINE_string('finetune_act_func', 'relu', '激活函数.')
flags.DEFINE_float('finetune_learning_rate', 0.01, '学习率.')
flags.DEFINE_float('finetune_momentum', 0.9, '动量参数.')
flags.DEFINE_integer('finetune_num_epochs', 10, ' epochs 数量.')
flags.DEFINE_integer('finetune_batch_size', 32, '每个 mini-batch 的大小.')
flags.DEFINE_string('finetune_opt', 'momentum', '["sgd", "ada_grad", "momentum", "adam"]')
flags.DEFINE_string('finetune_loss_func', 'softmax_cross_entropy', '损失函数. ["mse", "softmax_cross_entropy"]')
flags.DEFINE_float('finetune_dropout', 1, 'Dropout 参数.')

# 将自动编码器层参数从字符串转换为其特定类型
rbm_layers = utilities.flag_to_list(FLAGS.rbm_layers, 'int')
rbm_learning_rate = utilities.flag_to_list(FLAGS.rbm_learning_rate, 'float')
rbm_num_epochs = utilities.flag_to_list(FLAGS.rbm_num_epochs, 'int')
rbm_batch_size = utilities.flag_to_list(FLAGS.rbm_batch_size, 'int')
rbm_gibbs_k = utilities.flag_to_list(FLAGS.rbm_gibbs_k, 'int')

# 检查参数
assert FLAGS.dataset in ['mnist', 'cifar10', 'custom']
assert FLAGS.finetune_act_func in ['sigmoid', 'tanh', 'relu']
assert len(rbm_layers) > 0

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
        vlX = teX[:5000]  # Validation set is the first half of the test set
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
        trX, trY, vlX, vlY, teX, teY = None, None, None, None, None, None

    # 生成对象
    # 生成激活函数对象
    finetune_act_func = utilities.str2actfunc(FLAGS.finetune_act_func)

    # 生成DBN网络模型对象
    srbm = dbn.DeepBeliefNetwork(
        name=FLAGS.name, do_pretrain=FLAGS.do_pretrain,
        rbm_layers=rbm_layers,
        finetune_act_func=finetune_act_func, rbm_learning_rate=rbm_learning_rate,
        rbm_num_epochs=rbm_num_epochs, rbm_gibbs_k=rbm_gibbs_k,
        rbm_gauss_visible=FLAGS.rbm_gauss_visible, rbm_stddev=FLAGS.rbm_stddev,
        momentum=FLAGS.momentum, rbm_batch_size=rbm_batch_size, finetune_learning_rate=FLAGS.finetune_learning_rate,
        finetune_num_epochs=FLAGS.finetune_num_epochs, finetune_batch_size=FLAGS.finetune_batch_size,
        finetune_opt=FLAGS.finetune_opt, finetune_loss_func=FLAGS.finetune_loss_func,
        finetune_dropout=FLAGS.finetune_dropout)

    # 训练模型 (无监督预训练)
    if FLAGS.do_pretrain:
        srbm.pretrain(trX, vlX)

    # 微调
    print('Start deep belief net finetuning...')
    srbm.fit(trX, trY, vlX, vlY)

    # 测试模型
    print('Test set accuracy: {}'.format(srbm.score(teX, teY)))

    # 保存模型的预测
    if FLAGS.save_predictions:
        print('Saving the predictions for the test set...')
        np.save(FLAGS.save_predictions, srbm.predict(teX))


    def save_layers_output(which_set):
        if which_set == 'train':
            trout = srbm.get_layers_output(trX)
            for i, o in enumerate(trout):
                np.save(FLAGS.save_layers_output_train + '-layer-' + str(i + 1) + '-train', o)

        elif which_set == 'test':
            teout = srbm.get_layers_output(teX)
            for i, o in enumerate(teout):
                np.save(FLAGS.save_layers_output_test + '-layer-' + str(i + 1) + '-test', o)


    # 保存模型每一层的输出
    if FLAGS.save_layers_output_test:
        print('Saving the output of each layer for the test set')
        save_layers_output('test')

    # 保存模型每一层的输出
    if FLAGS.save_layers_output_train:
        print('Saving the output of each layer for the train set')
        save_layers_output('train')
