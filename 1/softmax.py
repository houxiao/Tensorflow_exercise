# 导入tensorflow
import tensorflow as tf
# 导入MNIST模块
from tensorflow.examples.tutorials.mnist import input_data
# 读入MNIST数据
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# x占位符 代表待识别的图片
x = tf.placeholder(tf.float32, [None, 784])

'''
Softmax
y = Softmax(W^Tx + b)
'''
W = tf.Variable(tf.zeros([784, 10]))  # tf中，模型的参数用tf.Variable表示
b = tf.Variable(tf.zeros([10]))
# y表示模型的输出
y = tf.nn.softmax(tf.matmul(x, W) + b)
# y实际上定义了回归模型

# y_是实际的图片标签，同样用占位符表示
y_ = tf.placeholder(tf.float32, [None, 10])

'''
占位符不依赖于其他tensor，值有用户自行传递，通常用来存储样本数据和标签
变量指在计算过程中可以改变的值，每次计算变量的值会被保存下来，通常用变量表示模型的参数
'''

# 衡量模型的输出y和实际的图片标签y_， 用交叉熵 模型损失
cross_entropy = tf.reduce_mean(-tf.reduce_sum((y_ * tf.log(y))))
# 用梯度下降方法优化损失
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 建立会话
sess = tf.InteractiveSession()
# 运行之前初始化所有变量，分配内存
tf.global_variables_initializer().run()
# 进行1000步梯度下降
for _ in range(1000):
    # 在mnist.train中取100个数据训练，batch_xs和batch_ys对应x和y_
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # 在session中运行train_step,运行时传入x和y_
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 正确预测结果
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# 计算预测准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# 最重模型准确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

# .9161