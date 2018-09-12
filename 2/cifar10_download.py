import cifar10
import tensorflow as tf

# tf.app.flags.FLAGS是tf内部一个全局变量存储器，同时可以用于命令行参数的处理
FLAG = tf.app.flags.FLAGS
FLAG.data_dir = 'cifar10_data/'

cifar10.maybe_download_and_extract()