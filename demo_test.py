import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np

inputs = tf.placeholder(tf.float32, shape=[None, None, None, 3])

conv1 = slim.conv2d(inputs, num_outputs=20, kernel_size=3, stride=4)

de_weight = tf.get_variable('de_weight', shape=[3, 3, 3, 20])

deconv1 = tf.nn.conv2d_transpose(conv1, filter=de_weight, output_shape=tf.shape(inputs),
                                 strides=[1, 4, 4, 1], padding='SAME')

loss = deconv1 - inputs
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(10):
        data_in = np.random.normal(size=[3, 97, 93, 3])
        _, los_ = sess.run([train_op, loss], feed_dict={inputs: data_in})
        print(los_)