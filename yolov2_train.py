# -*- coding: utf-8 -*-
"""
    yolov2_tensorflow.yolov2_train
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Created on 2016-12-26 17:55
    @author : huangguoxiong
    copyright: (c) 2016 by huangguoxiong.
    license: Apache license, see LICENSE for more details.
"""

import tensorflow as tf
import config.yolov2_config as cfg
import utils.pascal_voc as voc
import nets.yolov2 as yolo
from utils.timer import Timer
slim = tf.contrib.slim
voc.set_config(cfg)


# Load the images and labels.
#images, labels = voc.get_next_batch()
#print images.shape[0]
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
def box_iou():
    pass

def tf_post_precess(images,batch):
    predicts = yolo.yolo_net(images,images.shape[0])
    # 1. x,y,w,h处理
    # 2. scale处理
    # 3. class处理
    t_coords = predicts[...,:5]
    t_classes = predicts[...,5:]
    t_x = predicts[...,0:1]
    t_y = predicts[...,1:2]
    t_w = predicts[...,2]
    t_h = predicts[...,3]
    t_c = predicts[...,4:5] #scale
    
    t_index = tf.where(t_x>-99999)
    t_row = tf.reshape(tf.to_float(t_index[:,1:2]),t_x.get_shape())
    t_col = tf.reshape(tf.to_float(t_index[:,2:3]),t_x.get_shape())
    #t_b = tf.reshape(t_index[:,3:4],t_x.get_shape())

    t_x = (t_col + tf.sigmoid(t_x)) /cfg.cell_size
    t_y = (t_row + tf.sigmoid(t_y)) /cfg.cell_size
    
    t_c = tf.sigmoid(t_c)
    #print tf.tile(cfg.anchors[::2],[13*13]) 
    t_w = tf.exp(t_w) * cfg.anchors[::2]/ cfg.cell_size
    t_h = tf.exp(t_h) * cfg.anchors[1::2]/ cfg.cell_size
    t_w = tf.expand_dims(t_w,-1)
    t_h = tf.expand_dims(t_h,-1)

    t_cls_max = tf.reduce_max(t_classes,-1,True)
    t_classes = t_classes - t_cls_max
    t_classes = tf.nn.softmax(t_classes)

    #t_probs = t_classes * t_c
    print t_w
    print t_h
    return tf.concat(4,[t_x,t_y,t_w,t_h,t_c,t_classes])

