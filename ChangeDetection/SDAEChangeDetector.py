# coding=utf-8
import os
from collections import defaultdict

import cv2
import gdal
import numpy as np
import ogr
import shapely.affinity
import shapely.geometry
import shutil
import tensorflow as tf
from shapely.geometry import MultiPolygon, Polygon
from yadlt.models.autoencoders import stacked_denoising_autoencoder
from yadlt.utils import utilities


class SDAEChangeDetector(object):
    def __init__(self,
                 img1Path="E:/ypTest/2015/20150.TIF",
                 img2Path="E:/ypTest/2016/yinchuan20000.TIF",
                 outPath="E:/ypTest/out/",
                 workspace="E:/ypTest/"):
        """
        初始化函数。初始化了一些路径
        :param img1Path: 字符串类型。时相 1 图像路径
        :param img2Path: 字符串类型。时相 2 图像路径
        :param outPath: 字符串类型。输出路径
        :param workspace: 字符串类型。工作区路径（用于存放切割的图像块和；临时文件夹）
        """
        self.img1Path = img1Path
        self.img2Path = img2Path
        self.outPath = outPath
        self.workspace = workspace
        self.img1Dir = self.workspace + "phase1/"  # 时相 1 图像块目录，用于存放时相 1 图像切割的图像块
        self.img2Dir = self.workspace + "phase2/"  # 时相 2 图像块目录，用于存放时相 2 图像切割的图像块

    def convertRGB(self, img):
        b = cv2.split(img)[0]
        g = cv2.split(img)[1]
        r = cv2.split(img)[2]
        return cv2.merge((r, g, b))

    def splitDualPhaseImage(self):
        """
        切割两时相图像。用固定参数生成 CMD 指令字符串，通过调用 arcpy 切分
        【注意】
        调用前应确保 commandLine 中的 arcpy 环境路径和切分脚本 splitImage.py 路径正确
        :return:
        """
        status = True
        imgSuffixName = "TIFF"  # 切分后要生成的图像块的格式
        img1PrefixName = "img1"  # 时相 1 切分后图像块的公共前缀
        img2PrefixName = "img2"  # 时相 2 切分后图像块的公共前缀
        gridPath = "E:/ypTest/grid/yc1km.shp"  # 用于切分的 1km 银川三区网格
        # 检测盛放切分后图像块的目录是否存在，若不存在则创建
        if not os.path.exists(self.img1Dir):
            os.makedirs(self.img1Dir)
        if not os.path.exists(self.img2Dir):
            os.makedirs(self.img2Dir)
        img1, img2 = self.readImage(self.img1Path, self.img2Path)
        if img1.shape[0] > 1001 or img1.shape[1] > 1001:
            print("开始切分图像...")
            # 切分时相 1 图像
            print("正在切分时相 1 图像...")
            commandLine = "C:/Python27/ArcGIS10.2/python.exe D:/new_code/ypTest/splitImage.py {0} {1} {2} {3} {4}".format(
                self.img1Path, self.img1Dir, img1PrefixName, imgSuffixName, gridPath)
            sta = os.system(commandLine)
            if sta != 0:
                print("切分时相 1 图像出错！")
                status = False
                return status
            # 切分时相 2 图像
            print("正在切分时相 2 图像...")
            commandLine = "C:/Python27/ArcGIS10.2/python.exe D:/new_code/ypTest/splitImage.py {0} {1} {2} {3} {4}".format(
                self.img2Path, self.img2Dir, img2PrefixName, imgSuffixName, gridPath)
            sta = os.system(commandLine)
            if sta != 0:
                print("切分时相 2 图像出错！")
                status = False
                return status
        else:
            tfwSrcPath1 = os.path.split(self.img1Path)[0] + "\\" + os.path.splitext(os.path.split(self.img1Path)[1])[0] + ".tfw"
            tfwSrcPath2 = os.path.split(self.img2Path)[0] + "\\" + os.path.splitext(os.path.split(self.img2Path)[1])[0] + ".tfw"
            if os.path.exists(tfwSrcPath1) and os.path.exists(tfwSrcPath2):
                tfwDstPath1 = self.img1Dir + img1PrefixName + "1.tfw"
                tfwDstPath2 = self.img2Dir + img2PrefixName + "1.tfw"
                tifDstPath1 = self.img1Dir + img1PrefixName + "1.TIF"
                tifDstPath2 = self.img2Dir + img2PrefixName + "1.TIF"
                shutil.copy(tfwSrcPath1, tfwDstPath1)
                shutil.copy(tfwSrcPath2, tfwDstPath2)
                shutil.copy(self.img1Path, tifDstPath1)
                shutil.copy(self.img2Path, tifDstPath2)
            else:
                print("未找到与切分图像匹配的 TFW 文件！请检查！")
                status = False
        if status:
            print("切分图像完毕！")
        return status

    @staticmethod
    def initFlags(predictMode):
        """
        初始化堆栈降噪自动编码器的各项参数，其中通过 predictMode 选择要调用的模型
        :param predictMode: 字符串类型。用于确定要调用的网络模型，默认为"all"。
                            其中"all"代表检测所有变化的模型，其他字符串代表检测建筑变化的模型
        :return: tensorflow.app.flags.FLAGS 类型。包含了定义的网络模型的各项参数
        """
        flags = tf.app.flags
        FLAGS = flags.FLAGS
        # 全局配置
        flags.DEFINE_string('dataset', 'custom', '用哪个数据集. ["mnist", "cifar10", "custom"]')
        flags.DEFINE_boolean('do_pretrain', False, '是否使用无监督预训练网络.')
        flags.DEFINE_integer('seed', -1, '随机发生器的种子（> = 0）. 适用于测试超参数.')
        # 根据定义的模式，选择模型，默认为 all
        if predictMode == "all":
            flags.DEFINE_string('name', 'change_detection_sdae', '模型的名称.')
        else:
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
        return FLAGS

    @staticmethod
    def convertUint8(img):
        """
        将深度为16位的图像转化为8位深度
        :param img: ndarray 类型。需要转化深度的图像
        :return: ndarray类型。转化完成的8位深度的图像
        """
        if img.dtype == "uint16":
            maxValue = np.amax(img)
            img8 = cv2.convertScaleAbs(img, alpha=(255.0 / maxValue))
        else:
            img8 = img
        return img8

    def batchPredict(self, batchSize=1000000, predictMode="all", imgType=None):
        """
        读取盛放两时相切分好的图像块的目录，组成图像对，批量进行数据处理和预测，最后生成矢量文件
        :param batchSize: int 类型。代表将每个图像块分批次处理预测的批次大小，主要用于减小显存要求。默认为1000,000。
                        图像块尺寸默认为1000*1000大小， 即每对图像块数据量为 1000,000 组数据，若此规模的数据量相对于显存来说过大（运
                        行出现 OOM ERROR），可适量调小 batchSize 为一个小于 1000,000 的整型数;若硬件条件允许，可保持本默认参数不变。
        :param predictMode: 字符串类型。用于确定要调用的网络模型，默认为"all"。其中"all"代表检测所有变化的模型，其他字符串代表检测建筑
                            变化的模型
        :param imgType: list 类型。用于指明两时相图像通道顺序，默认为 ["RGB", "RGB"]。若有的图像的红蓝通道颠倒，则为 "BGR"
        :return:
        """
        status = True
        os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # 指定第二块GPU可用
        if imgType is None:
            imgType = ["BGR", "BGR"]
        tempDir = self.workspace + 'temp/'  # 定义临时目录的路径
        if not os.path.exists(tempDir):
            os.makedirs(tempDir)
        imgSuffixName = ".TIF"
        img1PrefixName = "img1"
        img2PrefixName = "img2"
        outType = "shp"
        print("正在准备模型...")
        FLAGS = self.initFlags(predictMode)
        # 检测两时相图像块目录，匹配成对的图像块，将成对的图像块批量进行处理
        fileList1 = os.listdir(self.img1Dir)
        fileList2 = os.listdir(self.img2Dir)
        for fileName in fileList1:
            if os.path.splitext(fileName)[1] == imgSuffixName:
                fileId = fileName.split(img1PrefixName)[-1].split('.')[0]
                anotherFileName = img2PrefixName + fileId + imgSuffixName
                if anotherFileName in fileList2:
                    img1Path = self.img1Dir + fileName
                    img2Path = self.img2Dir + anotherFileName
                    # 读取成对的两时相图像
                    img1, img2 = self.readImage(img1Path, img2Path)
                    if imgType[0] == "BGR":
                        img1 = self.convertRGB(img1)
                    if imgType[1] == "BGR":
                        img2 = self.convertRGB(img2)
                    # 预测两时相图像变化检测结果
                    print("开始预测两时相图像变化检测结果...")
                    prediction = self.predictSingle(img1, img2, FLAGS, tempDir, batchSize)
                    # 将预测结果可视化为二值图像
                    predictionImg = self.readPrediction(prediction, img1.shape)
                    print("将预测结果转化为图像，并生成矢量文件")
                    # 在临时文件夹中存储二值图像
                    cv2.imwrite(tempDir + anotherFileName.split('.')[0] + '.jpg', predictionImg)
                    os.remove(tempDir + "0.npy")
                    del prediction
                    del predictionImg
        if outType == "shp":
            # 读取临时文件夹中所有的二值图像，生成矢量文件
            self.generateShp(tempDir)
            # 清空临时文件夹和用于盛放两时相图像块的文件夹
            self.clearDir(tempDir)
            self.clearDir(self.img1Dir)
            self.clearDir(self.img2Dir)
        return status

    def readImage(self, img1Path, img2Path):
        """
        从给定的路径读取图像文件
        :param img1Path: 字符串类型。时相 1 图像块的路径
        :param img2Path: 字符串类型。时相 2 图像块的路径
        :return: ndarray 类型。时相 1 和时相 2 图像（已匹配）
        """
        # 读取图像
        img1 = cv2.imread(img1Path)
        img2 = cv2.imread(img2Path)
        # 确保图像深度为 uint8 类型
        img1 = self.convertUint8(img1)
        img2 = self.convertUint8(img2)
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
        return img1, img2

    def predictSingle(self, img1, img2, FLAGS, tempDir, batchSize=1000000):
        """
        给定两时相图像，预测它们的变化检测结果并返回
        :param img1: ndarray 类型。时相 1 图像
        :param img2: ndarray 类型。时相 2 图像
        :param FLAGS: tensorflow.app.flags.FLAGS 类型。包含了定义的网络模型的各项参数
        :param tempDir: 字符串类型。临时文件夹路径
        :param batchSize: int 类型。代表将每个图像块分批次处理预测的批次大小，主要用于减小显存要求。默认为1000,000。
                        图像块尺寸默认为1000*1000大小， 即每对图像块数据量为 1000,000 组数据，若此规模的数据量相对于显存来说过大（运
                        行出现 OOM ERROR），可适量调小 batchSize 为一个小于 1000,000 的整型数;若硬件条件允许，可保持本默认参数不变。
        :return: ndarray 类型。变化检测结果
        """
        row, col, dim = img1.shape
        step = int((row * col) / batchSize)
        res = (row * col) % batchSize
        predictions = np.zeros((row, col))
        if res != 0:
            step = step + 1
        for i in range(step):
            print("开始制作第{0}/{1}批数据".format((i + 1), step))
            dataset = self.make_dataset(img1, img2, batchSize, kernel_width=5, offset=i)
            print("==============开始读取模型并批量进行预测=============")
            predictions = self.do_predict(FLAGS, dataset)
            del dataset
            predictions = predictions.astype(np.float16)
            np.save(tempDir + str(i) + '.npy', predictions)
        if step == 1:
            return predictions
        elif step >= 2:
            return self.concludeFromTempDir(tempDir, step)

    @staticmethod
    def make_dataset(img1, img2, batchSize=1000000, kernel_width=5, offset=0):
        """
        从配对的图像块制作测试输入数据
        :param img1: ndarray 类型。时相 1 图像
        :param img2: ndarray 类型。时相 2 图像
        :param batchSize: int 类型。代表将每个图像块分批次处理预测的批次大小，主要用于减小显存要求。默认为1000,000。
                        图像块尺寸默认为1000*1000大小， 即每对图像块数据量为 1000,000 组数据，若此规模的数据量相对于显存来说过大（运
                        行出现 OOM ERROR），可适量调小 batchSize 为一个小于 1000,000 的整型数;若硬件条件允许，可保持本默认参数不变。
        :param kernel_width: int 类型。默认为 5。表示邻域矩阵边长
        :param offset: int 类型。用于控制分批次生成数据文件
        :return: ndarray 类型。制作好的测试数据
        """
        offset = offset * batchSize  # 偏移量，从偏移量开始读取本批数据，分批读取减少显存要求，每批批次大小默认为1000000
        length = 0

        content_R = []
        content_G = []
        content_B = []
        for dim in range(3):
            # 分离通道 0：B 1：G 2：R
            length = 0
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
                edgeSpace = int(t / 2)
                img1_expand[edgeSpace:height + edgeSpace, edgeSpace:width + edgeSpace] = img1_gray
                img2_expand[edgeSpace:height + edgeSpace, edgeSpace:width + edgeSpace] = img2_gray
                content = []  # content用于存储特征，他的维度是[height*width, 50]
                for i in range(edgeSpace, height + edgeSpace):
                    for j in range(edgeSpace, width + edgeSpace):
                        # 提取每个时相每个像素的5×5邻域像素特征，并将两时相特征拼接
                        count = count + 1
                        if count >= offset:
                            window1 = img1_expand[i - edgeSpace:i + edgeSpace + 1, j - edgeSpace:j + edgeSpace + 1]
                            window2 = img2_expand[i - edgeSpace:i + edgeSpace + 1, j - edgeSpace:j + edgeSpace + 1]
                            window1 = np.reshape(window1, (1, kernel_width * kernel_width))
                            window2 = np.reshape(window2, (1, kernel_width * kernel_width))
                            content.append(np.column_stack((window1, window2)).tolist())
                            length = len(content)
                            # print(length)
                            if length == batchSize:
                                break
                    if length == batchSize:
                        break
                if dim == 0:
                    content_B = np.array(content)
                if dim == 1:
                    content_G = np.array(content)
                if dim == 2:
                    content_R = np.array(content)
            else:
                print("时相1和时相2的图像尺寸不匹配!")
        content_RGB = np.column_stack((content_R, content_G, content_B))
        content_RGB = np.reshape(content_RGB, (length, kernel_width * kernel_width * 6))
        return content_RGB

    @staticmethod
    def do_predict(FLAGS, datasets):
        """
        调用堆栈降噪自动编码器进行预测
        :param FLAGS: tensorflow.app.flags.FLAGS 类型。包含了定义的网络模型的各项参数
        :param datasets: ndarray 类型。制作好的测试数据
        :return: ndarray 类型。预测结果
        """
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

        # 创建编码、解码、微调函数和网络模型对象
        dae_enc_act_func = [utilities.str2actfunc(af) for af in dae_enc_act_func]
        dae_dec_act_func = [utilities.str2actfunc(af) for af in dae_dec_act_func]
        finetune_act_func = utilities.str2actfunc(FLAGS.finetune_act_func)

        sdae = stacked_denoising_autoencoder.StackedDenoisingAutoencoder(
            do_pretrain=FLAGS.do_pretrain, name=FLAGS.name,
            layers=dae_layers, finetune_loss_func=FLAGS.finetune_loss_func,
            finetune_learning_rate=FLAGS.finetune_learning_rate,
            finetune_num_epochs=FLAGS.finetune_num_epochs,
            finetune_opt=FLAGS.finetune_opt, finetune_batch_size=FLAGS.finetune_batch_size,
            finetune_dropout=FLAGS.finetune_dropout,
            enc_act_func=dae_enc_act_func, dec_act_func=dae_dec_act_func,
            corr_type=dae_corr_type, corr_frac=dae_corr_frac, regcoef=dae_regcoef,
            loss_func=dae_loss_func, opt=dae_opt,
            learning_rate=dae_learning_rate, momentum=FLAGS.momentum,
            num_epochs=dae_num_epochs, batch_size=dae_batch_size,
            finetune_act_func=finetune_act_func)

        # 训练模型 (无监督预训练)
        # if FLAGS.do_pretrain:
        #     encoded_X, encoded_vX = sdae.pretrain(trX, vlX)

        teX = datasets
        # print('Saving the predictions for the test set...')
        internal_predictions = sdae.predict(teX)
        return internal_predictions

    @staticmethod
    def concludeFromTempDir(tempDir, step):
        """
        从临时文件夹读取分批次生成的中间预测结果，汇总得到当前成对图像块的预测结果
        :param tempDir: 字符串类型。临时文件夹路径
        :param step: int 类型。批次数量
        :return: ndarray 类型。成对图像块的预测结果
        """
        npy1 = np.load(tempDir + "0.npy")
        npy2 = np.load(tempDir + "1.npy")
        combineNpy = np.row_stack((npy1, npy2))
        del npy1, npy2
        os.remove(tempDir + "0.npy")
        os.remove(tempDir + "1.npy")
        for i in range(2, step):
            # print(i)
            npy = np.load(tempDir + str(i) + ".npy")
            combineNpy = np.row_stack((combineNpy, npy))
            del npy
            os.remove(tempDir + str(i) + ".npy")
        conclusion = np.array(combineNpy).astype(np.float16)
        np.save(tempDir + "0.npy", conclusion)
        return conclusion

    @staticmethod
    def readPrediction(prediction, imgShape):
        """
        将预测结果二值化，转变为可视化结果
        :param prediction: ndarray 类型。成对图像块的预测结果
        :param imgShape: tuple 类型。图像块的尺寸
        :return: ndarray 类型。二值化的预测结果
        """
        length, classes = np.shape(prediction)
        predictionLabel = []
        for ind in range(length):
            if prediction[ind][0] > prediction[ind][1]:
                predictionLabel.append(255)
            else:
                predictionLabel.append(0)
        height = imgShape[0]
        width = imgShape[1]
        if length != height * width:
            predictionLabel = predictionLabel[:-1]
        predictionLabel = np.array(predictionLabel)
        predictionImg = predictionLabel.reshape(height, width)
        return predictionImg

    def generateShp(self, tempDir):
        """
        读取临时文件夹，将生成的二值化预测结果图像，汇总成为矢量文件
        :param tempDir: 字符串类型。临时文件夹路径
        :return:
        """
        sumPolygons = []
        for filename in os.listdir(tempDir):
            newMask = cv2.imread(tempDir + "{}".format(filename), 0)
            newMask = newMask / 255
            new_pre2 = self.mask2Polygons(newMask)
            import fileinput

            list2 = []
            for line in fileinput.input(self.img2Dir + "{}.tfw".format(filename.split(".")[0])):
                list2.append(float(line.split("\n")[0]))
            matrix = tuple(list2)
            scaledPredPolygons = shapely.affinity.affine_transform(
                new_pre2, matrix=matrix)
            for ploy in scaledPredPolygons:
                sumPolygons.append(ploy)
        self.writeVectorFile(sumPolygons)

    def clearDir(self, dirPath):
        """
        清空文件夹
        :param dirPath: 要清空的文件夹路径
        :return:
        """
        ls = os.listdir(dirPath)
        for i in ls:
            tempPath = os.path.join(dirPath, i)
            if os.path.isdir(tempPath):
                self.clearDir(tempPath)
            else:
                os.remove(tempPath)

    @staticmethod
    def mask2Polygons(mask):
        """
        将二值化图像转为多边形对象列表
        :param mask: ndarray 类型。二值化预测结果
        :return: list 类型。多边形对象列表
        """
        epsilon = 2
        # first, find contours with cv2: it's much faster than shapely
        image, contours, hierarchy = cv2.findContours(((mask == 1) * 255).astype(np.uint8), cv2.RETR_CCOMP,
                                                      cv2.CHAIN_APPROX_TC89_KCOS)
        # create approximate contours to have reasonable submission size
        approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                           for cnt in contours]
        if not contours:
            return MultiPolygon()
        # now messy stuff to associate parent and child contours
        cnt_children = defaultdict(list)
        child_contours = set()
        assert hierarchy.shape[0] == 1
        # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
        for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
            if parent_idx != -1:
                child_contours.add(idx)
                cnt_children[parent_idx].append(approx_contours[idx])
        # create actual polygons filtering by area (removes artifacts)
        all_polygons = []
        for idx, cnt in enumerate(approx_contours):
            if idx not in child_contours and cv2.contourArea(cnt) >= 1.:
                assert cnt.shape[1] == 1
                poly = Polygon(
                    shell=cnt[:, 0, :],
                    holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                           if cv2.contourArea(c) >= 1.])
                all_polygons.append(poly)

        all_polygons = MultiPolygon(all_polygons)

        if not all_polygons.is_valid:
            # return all_polygons.buffer(0)
            all_polygons = all_polygons.buffer(0)

            if all_polygons.type == 'Polygon':
                all_polygons = MultiPolygon([all_polygons])

        return all_polygons

    def writeVectorFile(self, scaledPredPolygons):
        """
        将带有地理信息的多边形对象列表，转化为矢量文件并保存
        :param scaledPredPolygons: list 类型。带有地理信息的多边形对象列表
        :return:
        """
        gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")
        gdal.SetConfigOption("SHAPE_ENCODING", "")

        strVectorFile = self.outPath

        ogr.RegisterAll()

        strDriverName = "ESRI Shapefile"
        oDriver = ogr.GetDriverByName(strDriverName)
        if oDriver is None:
            print("%s 驱动不可用！\n", strDriverName)

        oDS = oDriver.CreateDataSource(strVectorFile)
        if oDS is None:
            print("创建文件【%s】失败！", strVectorFile)

        papszLCO = []
        oLayer = oDS.CreateLayer("TestPolygon", None, ogr.wkbPolygon, papszLCO)
        if oLayer is None:
            print("图层创建失败！\n")

        oFieldID = ogr.FieldDefn("FieldID", ogr.OFTInteger)
        oLayer.CreateField(oFieldID, 1)

        oFieldName = ogr.FieldDefn("FieldName", ogr.OFTString)
        oFieldName.SetWidth(100)
        oLayer.CreateField(oFieldName, 1)

        oDefn = oLayer.GetLayerDefn()

        i = 0
        for poly in scaledPredPolygons:
            oFeatureTriangle = ogr.Feature(oDefn)
            oFeatureTriangle.SetField(0, i)
            oFeatureTriangle.SetField(1, "bianhua")
            geomTriangle = ogr.CreateGeometryFromWkt(poly.to_wkt())
            oFeatureTriangle.SetGeometry(geomTriangle)
            oLayer.CreateFeature(oFeatureTriangle)
            i = i + 1

        oDS.Destroy()
        print("数据集创建完成！")


if __name__ == '__main__':
    img1Path = "E:/ypTest/2015/2015gf222.TIF"
    img2Path = "E:/ypTest/2016/201702gf222.TIF"
    # img1Dir = "E:/result/SupervisedModel/2015/test/"
    # img2Dir = "E:/result/SupervisedModel/2016/test/"
    outDir = "E:/ypTest/out/"
    workspace = "E:/ypTest/"
    sdaeChangeDetector = SDAEChangeDetector(img1Path, img2Path, outDir, workspace)
    sdaeChangeDetector.splitDualPhaseImage()
    sdaeChangeDetector.batchPredict(batchSize=50000, predictMode="build")
