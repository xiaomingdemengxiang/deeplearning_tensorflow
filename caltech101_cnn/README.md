使用卷积神经网络对caltech101数据集进行图片分类

@author:xubaochuan
@date:2016/9/24

测试正确率：53%

文件说明：
loaddata.py:将图片数据转为numpy.ndarray并保存到文件中。
caltech.py:网络模型，两层卷积池化层，一层全连接层。

运行说明：
1.下载caltech101数据集
2.在当前目录下创建四个.npy文件保存numpy.ndarray
3.python loaddata.py
4.caltch.py

遇到的问题：
1.在tensorflow中使用各种图片转numpy数组的方法，都失败了，所以单独将图片数据转化numpy数组保存下来
2.caltech101每张图片的大小大学在300*300左右，首先实验的时候将图片resize为128*128，使用4层卷积网络进行训练效果奇差，故模仿mnist_cnn的方法
将图片resize为32*32，使用2层卷积网络进行训练
3.mnist_cnn中使用cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))来计算交叉熵，但是当y_conv=0时，网络的权值会变成Nan，导致训练失败。
解决办法是cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y_conv, 1e-10, 1.0)))，为y_conv设置最小值和最大值

初次使用tensorflow，若代码存在问题，请见谅，并希望能够反馈给我
