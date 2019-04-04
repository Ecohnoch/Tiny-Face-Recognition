import tensorflow as tf 
import numpy as np 
import argparse
import os


from resnet50 import resnet50
from loss import arcface_loss

def parse_function(example_proto):
	features = {'image_raw': tf.FixedLenFeature([], tf.string),
		'label': tf.FixedLenFeature([], tf.int64)}
	features = tf.parse_single_example(example_proto, features)

	img = tf.image.decode_jpeg(features['image_raw'])
	img = tf.reshape(img, shape=(112, 112, 3))
	r, g, b = tf.split(img, num_or_size_splits=3, axis=-1)
	img = tf.concat([b, g, r], axis=-1)
	img = tf.cast(img, dtype=tf.float32)
	img = tf.subtract(img, 127.5)
	img = tf.multiply(img, 0.0078125)
	img = tf.image.random_flip_left_right(img)
	label = tf.cast(features['label'], tf.int64)
	return img, label

def train(args):
    num_classes = args.num_classes   # 85164
    batch_size  = args.batch_size
    ckpt_save_dir = args.ckpt_save_dir
    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
    tfr = args.tfrecords
    print('-------------Training Args-----------------')
    print('--tfrecords        :  ', tfr)
    print('--batch_size       :  ', batch_size)
    print('--num_classes      :  ', num_classes)
    print('--ckpt_save_dir    :  ', ckpt_save_dir)
    print('--lr               :  ', args.lr)
    print('-------------------------------------------')


    dataset = tf.data.TFRecordDataset(tfr)
    dataset = dataset.map(parse_function)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    images = tf.placeholder(tf.float32, [None, 112, 112, 3], name='image_inputs')
    labels = tf.placeholder(tf.int64,   [None, ], name='labels_inputs')

    emb = resnet50(images, is_training=True)

    logit = arcface_loss(embedding=emb, labels=labels, w_init=w_init_method, out_num=num_classes)
    inference_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels))

    p = int(512.0/batch_size)
    global_step = tf.Variable(name='global_step', initial_value=0, trainable=False)
    inc_op = tf.assign_add(global_step, 1, name='increment_global_step')
    lr_steps = [p*val for val in [40000, 60000, 80000]]
    lr = tf.train.piecewise_constant(global_step, boundaries=lr_steps, values=args.lr, name='lr_schedule')
    opt = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9)

    grads = opt.compute_gradients(inference_loss)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.apply_gradients(grads, global_step=global_step)
    pred = tf.nn.softmax(logit)
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, axis=1), labels), dtype=tf.float32))

    saver = tf.train.Saver(max_to_keep=3)
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    counter = 0
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # saver.restore(sess, '/data/ChuyuanXiong/backup/face_real330_ckpt/Face_vox_iter_271800.ckpt')
        for i in range(args.epoch):
            sess.run(iterator.initializer)

            while True:
                try:
                    image_train, label_train = sess.run(next_element)
                    # print(image_train.shape, label_train.shape) 
                    # print(label_train)
                    feed_dict = {images: image_train, labels: label_train}
                    _, loss_val, acc_val, _ = sess.run([train_op, inference_loss, acc, inc_op], feed_dict=feed_dict)

                    counter += 1
                    # print('counter: ', counter, 'loss_val', loss_val, 'acc: ', acc_val)
                    if counter % 100 == 0:
                        print('counter: ', counter, 'loss_val', loss_val, 'acc: ', acc_val)
                        filename = 'Face_vox_iter_{:d}'.format(counter) + '.ckpt'
                        filename = os.path.join(ckpt_save_dir, filename)
                        saver.save(sess, filename)
                except tf.errors.OutOfRangeError:
                    print('End of epoch %d', i)
               	    break

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser()

    # Train
    parser.add_argument('--tfrecords', default='/data/ChuyuanXiong/up/face/tfrecords/tran.tfrecords', required=True)
    parser.add_argument('--batch_size', default=64, type=int, required=True)
    parser.add_argument('--num_classes', default=85742, type=int, required=True)
    parser.add_argument('--lr', default=[0.001, 0.0005, 0.0003, 0.0001], required=True)
    parser.add_argument('--ckpt_save_dir', default='/data/ChuyuanXiong/backup/face_real403_ckpt', required=True)
    parser.add_argument('--epoch', default=10000, type=int, required=False)
    parser.set_defaults(func=train)
    
    opt = parser.parse_args()
    opt.func(opt)


