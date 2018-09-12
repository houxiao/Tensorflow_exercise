"""
tf中的读写机制
文件名队列 + 内存队列 双队列形式”
很好管理epoch

创建文件名队列： tf.train.string_input_producer()
    参数：
        文件名list
        epoch数
        shuffle

内存对立不需要自己建立，只需要使用reader(tf.WholeFileReader())对象从文件名队列读取数据就可以了
"""

"""
在使用tf.train.string_input_producer()创建文件名队列后，
整个系统处于'停滞'状态，文件名并没有加入队列中。
需要使用tf.train.start_queue_runners，启动填充队列的线程。
"""

''' example: A.jpg, B.jpg, C.jpg 5 epochs
with tf.Session() as sess:
    filename = ['A.jpg', 'B.jpg', 'C.jpg']
    file_name_queue = tf.train.string_input_producer(filename, suffle=False, num_epoch=5)
    reader = tf.WholeFileReader()
    key, value = reader.read(file_name_queue)
    tf.local_variables_initializer().run()
    threads = tf.train.start_queue_runners(sess=sess)
    i = 0
    while True:
        i += 1
        image_data = sess.run(value)
        with open('test_%d.jpg‘ % i, 'wb') as f:
            f.write(image_data)
    
    ......
    
'''

'''
cifar10 每个图片3073字节 <1 x label><3072 x pixel>

读取cifar10 数据：
1. 用tf.train.string_input_producer建立队列
2. 通过reader.read读数据，这里因为我们不是读取整个文件，因此reader不用tf.WholeFileReader()
    而是用tf.FixedLengthRecordReader()
3. 调用tf.train.start_queue_runners
4. 通过sess.run()取出图片
'''

import os
import scipy.misc
import tensorflow as tf

import cifar10_input


with tf.Session() as sess:
    dir_name = "cifar10_data/cifar-10-batches-bin"
    file_names = [os.path.join(dir_name, 'data_batch_%d.bin' % i) for i in range(1, 6)]

    file_queue = tf.train.string_input_producer(file_names)
    # cifar10_input.read_cifar10是预先写好的函数，可以从queue读取文件
    # 返回的read_input类型是uint8image也就是图像的Tensor
    read_input = cifar10_input.read_cifar10(file_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    threads = tf.train.start_queue_runners(sess=sess)
    sess.run(tf.global_variables_initializer())

    # 建立文件夹存图片
    if not os.path.exists('cifar10_data/raw/'):
        os.makedirs('cifar10_data/raw/')

    # 保存30张图片
    for i in range(30):
        # 每次sess.run(reshaped_image)都会取出一张照片
        image = sess.run(reshaped_image)
        scipy.misc.toimage(image).save('cifar10_data/raw/%d.jpg' % i)
