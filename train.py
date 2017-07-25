import tensorflow as tf
import time
from data import read_data
from nets import model
import progressbar

GLOBAL_STEP = 46


def train_num_iteration(train_op, loss_op, num_iteration=2913):
    global GLOBAL_STEP
    sess = tf.get_default_session()

    ####
    widgets = ["processing: ", progressbar.Percentage(),
               " ", progressbar.ETA(),
               " ", progressbar.FileTransferSpeed(),
               ]
    bar = progressbar.ProgressBar(widgets=widgets, max_value=num_iteration).start()
    ####

    if sess is None:
        raise ValueError("using with tf.Session() to create a sess")
    total_loss = 0.
    max_loss = 1.
    min_loss = 1.
    max_step = 0
    min_step = 0
    begin = time.time()
    for i in range(num_iteration):
        bar.update(i)
        _, loss = sess.run([train_op, loss_op])
        total_loss += loss
        if loss > max_loss:
            max_loss = loss
            max_step = i
        if loss < min_loss:
            min_loss = loss
            min_step = i
    bar.finish()
    GLOBAL_STEP += 1
    end = time.time() - begin
    print("GLOBAL_STEP: %d, elapsed time %.1f(s), %.5f(s) per sample,  mean loss is %.6f , "
          "max loss is %.6f in step %d, min loss is %.6f in step %d" % (GLOBAL_STEP, end, end / num_iteration,
                                                                        total_loss / num_iteration, max_loss, max_step,
                                                                        min_loss, min_step))


def main():
    file_names = '/media/fanyang/workspace/DataSet/VOCdevkit/VOC2012/semantic_2012_train.tfrecords'
    img, one_hot_label, _ = read_data.read_data(file_names=file_names)
    print(img)
    fcn = model.FCN()
    ###############debug#######
    # img = tf.random_normal(shape=[1, 574, 374, 3])
    # one_hot_label = tf.random_normal([1, 574, 374, 22])
    ##################
    logits = fcn.feedforward(img)
    loss = fcn.get_loss(logits=logits, labels=one_hot_label, regulizer=False)
    train_op = fcn.get_opt(loss)

    with tf.Session() as sess:
        tf.summary.FileWriter(logdir='./ckpt/', graph=sess.graph)
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        # fcn.restore_vgg16_ckpt("./ckpt/vgg_16.ckpt")
        fcn.restore_fcn_ckpt('./ckpt')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                train_num_iteration(train_op=train_op, loss_op=loss)
                fcn.save_fcn_ckpt(ckpt_path="./ckpt/fcn.ckpt", global_step=GLOBAL_STEP)
        except tf.errors.OutOfRangeError:
            print("Done training----epoch limit reached")
        except KeyboardInterrupt:
            print("KeboardInterrupt Exception")
        finally:
            coord.request_stop()
            coord.join(threads)
            print("all threads closed successfully")


if __name__ == '__main__':
    main()
