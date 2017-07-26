import tensorflow as tf
import matplotlib.pyplot as plt
from data.gene_label import COLOR_MAPS
from data.gene_label import label2img
from tensorflow.python.training import queue_runner_impl
import numpy as np

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94

RGB_MEAN = [_R_MEAN, _G_MEAN, _B_MEAN]


def read_data(file_names, batch_size=1):
    if isinstance(file_names, str):
        file_names = [file_names]
    assert isinstance(file_names, list)

    with tf.name_scope("InputPipeLine"):
        file_name_queue = tf.train.string_input_producer(file_names, num_epochs=10000, shuffle=True)

        # prepare reader
        reader = tf.TFRecordReader()
        key, record_string = reader.read(file_name_queue)
        features = tf.parse_single_example(record_string, features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.string),
            'image_raw': tf.FixedLenFeature([], tf.string)})
        img = tf.decode_raw(features['image_raw'], tf.uint8)
        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)
        # depth = tf.cast(features['depth'], tf.int32)
        label = tf.decode_raw(features['label'], tf.uint8)

        ####
        img_shape = tf.stack([256, 256, 3])
        label_shape = tf.stack([256, 256])

        # img [height, width, channel]
        img_reshaped = tf.cast(tf.reshape(img, img_shape), tf.float32)

        # label [height, width]
        label_reshaped = tf.reshape(label, label_shape)
        # img_reshaped = tf.expand_dims(img_reshaped, axis=0)
        # label_reshaped = tf.expand_dims(label_reshaped, axis=0)
        # one_hot_label  [batch_size, height, width, 22]
        # one_hot_label = tf.one_hot(label_reshaped, depth=22, axis=-1, dtype=tf.uint8)
        img_reshaped, one_hot_label, _ = preprocess_data(img_reshaped, label_reshaped)
        # concated_data = tf.concat([img_reshaped, one_hot_label], axis=-1)
        #
        # data_shuffle_queue = tf.RandomShuffleQueue(capacity=100, min_after_dequeue=80, dtypes=tf.float32)
        #
        # enqueue_op = data_shuffle_queue.enqueue(concated_data)
        # qr = tf.train.QueueRunner(data_shuffle_queue, enqueue_ops=[enqueue_op] * 4)
        # queue_runner_impl.add_queue_runner(qr)
        #
        # get_concated_data = data_shuffle_queue.dequeue()
        #
        # # the first is img , the shape is [1, height, width, 3]
        # # the second is one hot label, the shape is [1, height, width, num_classes]
        # img = get_concated_data[:, :, :, :3]
        # one_hot_label = get_concated_data[:, :, :, 3:]
        # img_shape = tf.shape(img)
        # img = tf.reshape(img, [img_shape[0], img_shape[1], img_shape[2], 3])
        # one_hot_label_shape = tf.shape(one_hot_label)
        #
        # one_hot_label = tf.reshape(one_hot_label,
        #                            [one_hot_label_shape[0], one_hot_label_shape[1], one_hot_label_shape[2], 22])

        bat_img, batch_label = tf.train.shuffle_batch(tensors=[img_reshaped, one_hot_label],
                                                      batch_size=batch_size, capacity=100,
                                                      min_after_dequeue=80,
                                                      num_threads=4)

        return bat_img, batch_label, None


def preprocess_data(img, label):
    with tf.name_scope('preprocess_data'):
        rgb_mean = tf.constant(RGB_MEAN, dtype=tf.float32)
        img = tf.subtract(img, rgb_mean)

        one_hot_label = tf.one_hot(label, depth=21, axis=-1)
    return img, one_hot_label, label


def main():
    # /media/fanyang/workspace/DataSet/VOCdevkit/VOC2012/semantic_2012_train.tfrecords
    data_files = '/media/fanyang/workspace/DataSet/VOCdevkit/VOC2012/semantic_2012_train.tfrecords'
    img, label, _ = read_data(data_files)
    print(tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS))
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        img, label = sess.run([img, label])

        plt.imshow(label2img(np.argmax(label[0], axis=-1)))

        plt.show()
        coord.request_stop()
        coord.join(threads=threads)


if __name__ == '__main__':
    main()
