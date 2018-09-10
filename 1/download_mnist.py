import os
import scipy.misc

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# print(mnist.train.images.shape)  # (55000, 784)
# print(mnist.train.labels.shape)  # (55000, 10)
#
# print(mnist.validation.images.shape)  # (5000, 784)
# print(mnist.validation.labels.shape)  # (5000, 10)
#
# print(mnist.test.images.shape)  # (10000, 784)
# print(mnist.test.labels.shape)  # (10000, 10)

# print(mnist.train.images[0, :])  # print 1st img in vector

# save_dir = "MNIST_data/raw/"
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
#
# # save first 20 imgs
# for i in range(20):
#     image_array = mnist.train.images[i, :].reshape(28, 28)
#     filename = save_dir + "mnist_train_%d.jpg" % i
#     scipy.misc.toimage(image_array, cmin=0.0, cmax=1.0).save(filename)

print(mnist.train.labels[0, :])


