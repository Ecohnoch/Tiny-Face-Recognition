import tensorflow as tf 
import numpy as np 
import mxnet as mx

import argparse
import pickle
import cv2
import os
import io

from loss import arcface_loss
from resnet50 import resnet50
from verification import ver_test


def load_bin(db_name, image_size):
    bins, issame_list = pickle.load(open(db_name, 'rb'), encoding='bytes')
    data_list = []
    for _ in [0,1]:
        data = np.empty((len(issame_list)*2, image_size[0], image_size[1], 3))
        data_list.append(data)
    for i in range(len(issame_list)*2):
        _bin = bins[i]
        img = mx.image.imdecode(_bin).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # print(img.shape)
        #
        img = cv2.resize(img, (112, 112))
        #
        for flip in [0,1]:
            if flip == 1:
                img = np.fliplr(img)
            data_list[flip][i, ...] = img
        i += 1
        if i % 1000 == 0:
            print('loading bin', i)
    print(data_list[0].shape)
    return data_list, issame_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Infer
    parser.add_argument('--datasets', default='/data/ChuyuanXiong/up/face/faces_emore/cfp_fp.bin', required=True)
    parser.add_argument('--dataset_name', default='cfp_fp', required=True)
    parser.add_argument('--num_classes', default=85742, type=int, required=True)
    parser.add_argument('--ckpt_restore_dir', default='/data/ChuyuanXiong/backup/face_ckpt/Face_vox_iter_78900.ckpt', required=True)
    opt = parser.parse_args()


    
    ver_list = []
    ver_name_list = []
    data_set = load_bin(opt.datasets, [112,112])
    ver_list.append(data_set)
    ver_name_list.append(opt.dataset_name)

    images = tf.placeholder(tf.float32, [None, 112, 112, 3], name='image_inputs')
    labels = tf.placeholder(tf.int64,   [None, ], name='labels_inputs')
    w_init_method = tf.contrib.layers.xavier_initializer(uniform=False)
    num_classes = opt.num_classes

    emb = resnet50(images, is_training=True)
    emb = tf.contrib.layers.flatten(emb)
    logit = arcface_loss(embedding=emb, labels=labels, w_init=w_init_method, out_num=num_classes)
    # inference_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=labels))

    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    counter = 0
    saver = tf.train.Saver(var_list=var_list)
    with tf.Session(config=config) as sess:
    	saver.restore(sess, opt.face_ckpt)
    	feed_dict = {}

    	results = ver_test(ver_list=ver_list, ver_name_list=ver_name_list, nbatch=0, sess=sess, embedding_tensor=emb, 
    		batch_size=32, feed_dict=feed_dict, input_placeholder=images)
    	print(results)