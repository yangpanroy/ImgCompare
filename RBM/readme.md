# Readme
本脚本使用 SDAE 来实现变化检测。本方法将两张图片对应像素的邻域矩阵展开并拼接到一起作为特征输入，使用 SDAE 进行编码和解码，通过无监督预训练降低重构损失，使得编码得到的降维特征更能体现图片特征。然后将编码特征加标签用于训练 softmax 进行有监督微调。
## 环境配置
请确保你已经正确安装了 opencv-python, numpy, tensorflow, yadlt 和他们的依赖包。

注意: yadlt.core.supervised_model 中只在 fit() 方法中初始化了 tf 的参数，而在 predict() 方法中并未初始化，所以需要将 predict() 方法改为以下代码：
```python
    def predict(self, test_X):
        """Predict the labels for the test set.

        Parameters
        ----------

        test_X : array_like, shape (n_samples, n_features)
            Test data.

        Returns
        -------

        array_like, shape (n_samples,) : predicted labels.
        """
        with self.tf_graph.as_default():
            # Build model
            self.build_model(test_X.shape[1], 2)
            with tf.Session() as self.tf_session:
                # Initialize tf stuff
                summary_objs = tf_utils.init_tf_ops(self.tf_session)
                self.tf_merged_summaries = summary_objs[0]
                self.tf_summary_writer = summary_objs[1]
                self.tf_saver = summary_objs[2]
                # Restore the model
                self.tf_saver.restore(self.tf_session, self.model_path)
                feed = {
                    self.input_data: test_X,
                    self.keep_prob: 1
                }
                return self.mod_y.eval(feed)
```
这样才可以单独调用 predict() 进行预测。

## 训练
使用 batch_training.py 脚本进行批量训练。在训练前要将时相1、时相2、标签文件的路径进行赋值：
```python
    img1_dir = '/media/files/yp/rbm/pic_div/2015/'  # 时相1图像的路径
    img2_dir = '/media/files/yp/rbm/pic_div/2016/'  # 时相2图像的路径
    label_dir = '/media/files/yp/rbm/pic_div/label/'  # 存放标签的路径
```
注意：请确保时相1图像和时相2图像对应的文件名、图像尺寸应一致，且图像经过配准等预处理。 

在 init_flags() 方法中更改模型的名称（之后预测需要通过模型名称读取模型参数并预测）。
```python
flags.DEFINE_string('name', 'change_detection_sdae_urban_upgrade', '模型的名称.')
```
其他网络参数视情况修改。

## 预测
使用 batch_testing_upgrade.py 脚本进行批量预测。和训练时一样，预测前请确保时相1、时相2、输出预测图路径已经被赋值；确保读取的模型名称与训练的模型名称一致。