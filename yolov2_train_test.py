# -*- coding: utf-8 -*-
"""
    yolov2_tensorflow.yolov2_input_test
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Created on 2016-12-27 09:49
    @author : huangguoxiong
    copyright: (c) 2016 by huangguoxiong.
    license: Apache license, see LICENSE for more details.
"""
import cv2
import numpy as np
import tensorflow as tf
import config.yolov2_config as cfg
import utils.pascal_voc as voc
import yolov2_train as train
from utils.timer import Timer
slim = tf.contrib.slim
#voc.set_config(cfg)
#voc.print_config()

# Load the images and labels.
#voc._get_train_paths()
images,labels = voc.get_next_batch()

## Create the model.
#scene_predictions, depth_predictions, pose_predictions = CreateMultiTaskModel(images)
#
## Define the loss functions and get the total loss.
#classification_loss = slim.losses.softmax_cross_entropy(scene_predictions, scene_labels)
#sum_of_squares_loss = slim.losses.sum_of_squares(depth_predictions, depth_labels)
#pose_loss = MyCustomLossFunction(pose_predictions, pose_labels)
#slim.losses.add_loss(pose_loss) # Letting TF-Slim know about the additional loss.
#
## The following two ways to compute the total loss are equivalent:
#regularization_loss = tf.add_n(slim.losses.get_regularization_losses())
#total_loss1 = classification_loss + sum_of_squares_loss + pose_loss + regularization_loss
#
## (Regularization Loss is included in the total loss by default).
#total_loss2 = losses.get_total_loss()
img = cv2.imread(cfg.test_img,1)
inputs = cv2.resize(img, (cfg.image_size, cfg.image_size)).astype(np.float32)
inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
inputs = (inputs / 255.0) 
inputs = np.reshape(inputs, (1, cfg.image_size, cfg.image_size, 3))
inputs = images
#print labels[np.isnan(labels)]
#net = train.tf_post_process(inputs,images[0])
train_imgs = tf.placeholder(dtype=tf.float32,shape=images.shape)
train_lbls = tf.placeholder(dtype=tf.float32,shape=labels.shape)
print train_imgs
iou = train._train(train_imgs,train_lbls)
init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = False
init_T = Timer()
init_T.tic()
with tf.Session(config=config) as sess:
    sess.run(init)
    saver = tf.train.Saver(slim.get_model_variables())
    #saver.save(sess,OUT_FILE)
    
    init_T.toc()
    print init_T.average_time
    restore_T = Timer()
    restore_T.tic()
    saver.restore(sess, cfg.out_file)
    print "weights restore."
    restore_T.toc()
    print restore_T.average_time
    run_T = Timer()
    run_T.tic()
    #boxiou = sess.run(iou)
    train = sess.run(iou)
    run_T.toc()
    #print boxiou.shape
    ##net_output = sess.run(net)
    #out = net_output[...,5:] * net_output[...,4:5]
    ##print net_output[...,5:]
    ##print net_output[...,4:5]
    #out[out<cfg.threshold] =0
    #out_m = np.max(out,4)    
    #out_i = np.argmax(out,4)
    ##print out_m
    #print "net_output:",out_i [out_m > cfg.threshold]
    #print "net_output:",out_m [out_m > cfg.threshold]
    #run_T.toc()
    #print run_T.average_time

