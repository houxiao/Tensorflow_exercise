"""
建立卷积神经网络 提高手写识别准确率
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# 将单张图片从784维向量还原为28*28的矩阵图片
x_image = tf.reshape(x, [-1, 28, 28, 1])


# 返回一个给定形状的变量并自动以截断正态分布初始化
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 返回一个给定形状的变量，初始化所有值是0.1
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


"""
weight_variable创建卷积的核(kernel)
bias_variable创建卷积的偏置
"""


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 第一层卷积，由一个卷积接一个maxpooling完成，卷积在每个
# 5x5的patch中算出32个特征。
# 卷积的权重张量形状是[5, 5, 1, 32]，前两个维度是patch的大小，
# 接着是输入的通道数目，最后是输出的通道数目。
# 而对于每一个输出通道都有一个对应的偏置量。
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

'''
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
真正进行卷积计算。卷积计算后选用ReLU作为激活函数

h_pool1 = max_pool_2x2(h_conv1)
进行池化操作

卷积，激活函数，池化 是一个卷积层的标配。
'''

# 对第一次卷积后产生的h_pool1再做一次卷积计算
# 每个5x5的patch会得到64个特征
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 两层卷积层之后是全连接层
# 全连接层 输出位1024维的向量
# 图片尺寸变为7x7，加入有1024个神经元的全连接层，
# 把池化层输出张量reshape成向量乘上权重矩阵，加上偏置，然后进行ReLU
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
# 使用Dropout, keep_prob 是一个占位符，训练时为0.5，测试时为1
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

'''
在全连接层加入Dropout，是一种防止神经网络过拟合的方法
在每一步训练时，以一定概率去掉网络中的某些连接，
但不是永久的，只是在当前步骤中去除，并且每一步去除的连接都是随机选择的
'''

# 最后在加入一层全连接，把上一步得到的h_fc1_drop转换为10个类别的打分
# 把1024维的向量转成10维，对应10个类别
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

'''
y_conv相当于softmax中的logit
可以使用softmax函数转化为10个类别的概率然后计算交叉熵损失

但是Tensorflow提供了更直接的tf.nn.softmax_cross_entropy_with_logits函数，
可以直接对logit定义交叉熵损失
'''

'''  softmax 版本
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))  #计算交叉熵
'''

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
# 同样定义 train_step
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 定义测试的准确率
correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

'''
训练与softmax.py相似
不同在于这次会和外在验证集上计算模型的准确率并输出，方便监控训练的进度，也可据此调整模型参数
'''

# 创建session， 初始化参数
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# 训练2000步
for i in range(20000):
    batch = mnist.train.next_batch(50)
    # 没100步报告一次在验证集上的准确率
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, trai"
              "ning accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# 训练结束后报告在测试集上的准确率
# print("test accuracy %g" % accuracy.eval(session=sess,feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

'''
ResourceExhaustedError的原因及解决方式
这个矩阵的尺寸是10000x28x28x32，每个元素是一个float64占用8字节，
所以这单个对象就需要占用1.87g的存储空间，还不算其他的系统运行内存和其他变量。肯定崩

解决办法
把test数据也分了10份，每份batchsize=1000，跑10次结果取accuracy的平均

'''
import numpy as np

accuResult = list(range(10))
for i in range(10):
    batch = mnist.test.next_batch(1000)
    accuResult[i] = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
print("Test accuracy:", np.mean(accuResult))
