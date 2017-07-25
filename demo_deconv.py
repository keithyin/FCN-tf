import tensorflow as tf

shape = tf.constant([1, 9, 9, 3], dtype=tf.int32)
kernel = tf.get_variable('kernel', shape=[3, 3, 3, 10])
data = tf.Variable(tf.random_normal(shape=[1, 5, 5, 10]))

res = tf.nn.conv2d_transpose(value=data, filter=kernel, output_shape=shape, padding="SAME", strides=[1, 2, 2, 1])

print(res)