from nets import model
import tensorflow as tf
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from data.read_data import RGB_MEAN
import os
import pydensecrf.densecrf as dcrf
from tools import image_visual_tools
from pydensecrf.utils import create_pairwise_bilateral, \
    create_pairwise_gaussian, unary_from_softmax


def crf(prob, image):
    # [height, width, class] --> [class, height, width]
    prob = np.transpose(prob, axes=(2, 0, 1))
    unary = unary_from_softmax(prob)
    unary = np.ascontiguousarray(unary)
    d = dcrf.DenseCRF2D(image.shape[0], image.shape[1], 21)
    d.setUnaryEnergy(unary)
    feats = create_pairwise_gaussian(sdims=(3, 3), shape=image.shape[:2])

    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    feats = create_pairwise_bilateral(sdims=(80, 80), schan=(20, 20, 20),
                                      img=image, chdim=2)

    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(5)
    res = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))
    return res


def inference_net(inputs, net, phrase='TEST'):
    logits = net.feedforward(inputs, is_training=False)
    prob = tf.nn.softmax(logits, dim=-1)
    labels = tf.argmax(logits, axis=-1)
    labels = tf.squeeze(labels)
    return labels, tf.squeeze(prob)


def do_inference(label_tensor, prob_tensor, img):
    img = img.astype(np.float32)
    img = img - np.array(RGB_MEAN)
    img = np.expand_dims(img, axis=0)
    sess = tf.get_default_session()
    if sess is None:
        raise ValueError("using with tf.Session() as sess before call the function")
    label, prob = sess.run([label_tensor, prob_tensor], feed_dict={'inputs:0': img})
    # label is 2-D
    return label, prob


def main():
    #######################
    img_dir = '/media/fanyang/workspace/DataSet/VOCdevkit/VOC2012/JPEGImages/'
    result_dir = '/media/fanyang/workspace/DataSet/VOCdevkit/VOC2012/SemanticResult/'
    imgs_name = os.listdir(img_dir)
    imgs_abs_path = [os.path.join(img_dir, img) for img in imgs_name]
    #######################

    net = model.FCN()
    inputs = tf.placeholder(dtype=tf.float32, shape=[1, None, None, 3], name='inputs')
    labels, prob_t = inference_net(inputs, net)

    index = np.arange(len(imgs_abs_path))
    np.random.shuffle(index)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        net.restore_fcn_ckpt("./ckpt")
        for i in index[:10]:
            img_path = imgs_abs_path[i]
            img_name = imgs_name[i]
            img = imread(img_path)
            label, prob = do_inference(label_tensor=labels, prob_tensor=prob_t, img=img)
            fig = plt.figure()
            axes1 = fig.add_subplot(1, 3, 1)
            axes2 = fig.add_subplot(1, 3, 2)
            axes3 = fig.add_subplot(1, 3, 3)
            semantic_img = image_visual_tools.semantic_image(label, max_class_id=21)
            crf_img = image_visual_tools.semantic_image(crf(prob, img), max_class_id=21)
            axes1.imshow(semantic_img)
            axes2.imshow(crf_img)
            axes3.imshow(img)
            fig.savefig(os.path.join(result_dir, img_name))


if __name__ == '__main__':
    main()
