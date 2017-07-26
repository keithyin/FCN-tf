import tensorflow as tf
import nets.vgg as vgg
from tensorflow.contrib import slim


def img_softmax_loss(logits, labels):
    """
    softmax loss for image
    :param logits: shape [batch_size, height, width, num_classes]
    :param labels: shape [batch_size, height, width, num_classes]
    :return: mean loss value, pixel-wise
    """
    with tf.name_scope("img_softmax_loss"):
        # balance the sample
        class_weights = tf.constant([1.] * 21, dtype=tf.float32)
        dis_logits = tf.nn.softmax(logits=logits, dim=-1)
        cross_entropy = tf.reduce_mean(-labels * tf.log(dis_logits + 1e-8) * class_weights)
        loss = cross_entropy * (logits.get_shape().as_list()[-1])
    return loss


class FCN(object):
    def __init__(self):
        self.has_restore_vgg16_saver = False
        self.has_fcn_saver = False
        self.fcn_saver = None

    def feedforward(self, inputs, is_training=True):
        """
        feedforward procedure
        :param inputs: net's input , the shape is [1, height, width, 3]
        :return: logits, the shape is [1, height, width, 22]
        """
        with slim.arg_scope(vgg.vgg_arg_scope()):
            logits = vgg.vgg_16(inputs=inputs, num_classes=21, is_training=is_training)
        return logits

    @staticmethod
    def get_loss(logits, labels, regulizer=True):
        with tf.name_scope('get_loss'):
            # tf.assert_equal(tf.shape(logits), tf.shape(labels))
            loss = img_softmax_loss(logits=logits, labels=labels)
            regu_loss = 0.
            if regulizer:
                var_list = tf.trainable_variables()
                regulizer_obj = slim.l2_regularizer(0.0001)
                regu_loss = slim.apply_regularization(regulizer_obj, weights_list=var_list)
            loss = tf.add(loss, regu_loss, name='total_loss')
        return loss

    @staticmethod
    def get_opt(loss):
        with tf.name_scope('get_opt'):
            opt = tf.train.RMSPropOptimizer(learning_rate=0.00005).minimize(loss)
        return opt

    def restore_vgg16_ckpt(self, ckpt_file):
        sess = tf.get_default_session()
        if sess is None:
            raise ValueError("using with tf.Session() as sess before call this method")
        if not self.has_restore_vgg16_saver:
            restore_var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg_16')
            saver = tf.train.Saver(var_list=restore_var_list)
            saver.restore(sess=sess, save_path=ckpt_file)
            self.has_restore_vgg16_saver = True
        else:
            raise ValueError("you have restored once, shouldn't do it again")

    def restore_fcn_ckpt(self, ckpt_dir):
        sess = tf.get_default_session()
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if sess is None:
            raise ValueError("using with tf.Session() as sess before call this method")
        if not self.has_fcn_saver:
            saver = tf.train.Saver()
            saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)
            self.has_fcn_saver = True
            self.fcn_saver = saver
        else:
            raise ValueError("you have restored once, shouldn't do it again")
        print("successfully restored from fcn ckpt! ")

    def save_fcn_ckpt(self, ckpt_path, global_step):
        sess = tf.get_default_session()
        if sess is None:
            raise ValueError("using with tf.Session() as sess before call this method")
        if not self.has_fcn_saver:
            saver = tf.train.Saver()
            saver.save(sess=sess, save_path=ckpt_path, global_step=global_step, write_meta_graph=False)
            self.has_fcn_saver = True
            self.fcn_saver = saver
        else:
            self.fcn_saver.save(sess=sess, save_path=ckpt_path, global_step=global_step, write_meta_graph=False)
