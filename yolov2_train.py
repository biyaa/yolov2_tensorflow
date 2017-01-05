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
import utils.stat as stat
import nets.yolov2 as yolo
from utils.timer import Timer
slim = tf.contrib.slim
voc.set_config(cfg)


# Load the images and labels.
#images, labels = voc.gep_nexp_batch()
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
#regularization_loss = tf.add_n(slim.losses.gep_regularization_losses())
#total_loss1 = classification_loss + sum_of_squares_loss + pose_loss + regularization_loss
#
## (Regularization Loss is included in the total loss by default).
#total_loss2 = losses.gep_total_loss()
def box_iou_by_pred(preds,truths):
    truths = tf.expand_dims(truths,1)
    truths = tf.expand_dims(truths,1)
    truths = tf.expand_dims(truths,1)
    # coords
    p_x = preds[...,0:1]
    p_y = preds[...,1:2]
    p_w = preds[...,2:3]
    p_h = preds[...,3:4]

    p_l = p_x - p_w/2
    p_r = p_x + p_w/2
    p_t = p_y - p_h/2
    p_b = p_y + p_h/2

    t_x = truths[...,0]
    t_y = truths[...,1]
    t_w = truths[...,2]
    t_h = truths[...,3]

    t_l = t_x - t_w/2
    t_r = t_x + t_w/2
    t_t = t_y - t_h/2
    t_b = t_y + t_h/2

    # box intersection
    x_w = tf.minimum(t_r , p_r) - tf.maximum(t_l , p_l)
    x_h = tf.minimum(t_b , p_b) - tf.maximum(t_t , p_t)
    area_intersction = x_w * x_h
    area_intersction = tf.maximum(area_intersction,0)
    
    # box union
    area_p = p_w * p_h
    area_t = t_w * t_h
    area_union = area_p + area_t - area_intersction
    return area_intersction/area_union

def tf_post_process(predicts):
    # 1. x,y,w,h处理
    # 2. scale处理
    # 3. class处理
    p_coords = predicts[...,:5]
    p_classes = predicts[...,5:]
    p_x = predicts[...,0:1]
    p_y = predicts[...,1:2]
    p_w = predicts[...,2]
    p_h = predicts[...,3]
    p_c = predicts[...,4:5] #scale
    
    p_index = tf.where(p_x>-99999)
    p_row = tf.reshape(tf.to_float(p_index[:,1:2]),p_x.get_shape())
    p_col = tf.reshape(tf.to_float(p_index[:,2:3]),p_x.get_shape())
    #p_b = tf.reshape(p_index[:,3:4],p_x.get_shape())

    p_x = (p_col + tf.sigmoid(p_x)) /cfg.cell_size
    p_y = (p_row + tf.sigmoid(p_y)) /cfg.cell_size
    
    p_c = tf.sigmoid(p_c)
    #print tf.tile(cfg.anchors[::2],[13*13]) 
    p_w = tf.exp(p_w) * cfg.anchors[::2]/ cfg.cell_size
    p_h = tf.exp(p_h) * cfg.anchors[1::2]/ cfg.cell_size
    p_w = tf.expand_dims(p_w,-1)
    p_h = tf.expand_dims(p_h,-1)

    p_cls_max = tf.reduce_max(p_classes,-1,True)
    p_classes = p_classes - p_cls_max
    p_classes = tf.nn.softmax(p_classes)

    #p_probs = p_classes * p_c
    return tf.concat(4,[p_x,p_y,p_w,p_h,p_c,p_classes])

def sigmoid_gradient(x):
    return (1-x)*x

def delta_obj_scales(pred_scales,cfg_scale,iou):
    d_scales = cfg_scale * ((iou - pred_scales) * sigmoid_gradient(pred_scales))
    return d_scales

def delta_noobj_scales(pred_scales,cfg_scale):
    d_scales = cfg_scale * ((0 - pred_scales) * sigmoid_gradient(pred_scales))
    return d_scales

def loss(net,labels):
    net_out = tf_post_process(net)
    delta = tf.Variable(tf.zeros(net_out.get_shape()))

    # 1. init delta
    delta_scales = delta[...,4:5]
    delta_scales = delta_noobj_scales(net_out[...,4:5],cfg.noobject_scale)

    # 2. compute avg_anyobj
    stat.avg_anyobj = tf.reduce_sum(net_out[...,4:5])
    # 3. compute best_iou
    iou_by_pred = box_iou_by_pred(net_out,labels)
    best_iou_by_pred = tf.reduce_max(iou_by_pred,-1,keep_dims=True)
    # 4. in delta_scales, select best_iou>threshold to set 0
    cond = best_iou_by_pred > cfg.threshold
    delta_scales = tf.where(cond,delta_scales*0,delta_scales)
    
    truths = labels[...,:4]

    t_x = truths[...,0:1]
    t_y = truths[...,1:2]
    t_x = t_x * cfg.cell_size
    t_y = t_y * cfg.cell_size

    t_shift_x = t_x * 0
    t_shift_y = t_y * 0

    truths_shift = tf.concat(2,[t_shift_x,t_shift_y,truths[...,2:4]])
    preds = net_out[...,:4]
    print preds

    p_x = preds[...,0:1]
    p_y = preds[...,1:2]
    p_shift_x = p_x * 0
    p_shift_y = p_y * 0

    preds_shift = tf.concat(4,[p_shift_x,p_shift_y,preds[...,2:4]])

    shift_iou = box_iou_by_pred(preds_shift,truths_shift)
    print delta
    return delta


def _train(images,labels):
    return loss(images,labels)

def train():
    log_dir = cfg.train_log_path

    images,labels = voc.get_next_batch()
    train_imgs = tf.placeholder(dtype=tf.float32,shape=images.shape)
    train_lbls = tf.placeholder(dtype=tf.float32,shape=labels.shape)
    net = yolo.yolo_net(train_imgs,images.shape[0],trainable=True)
    t_loss = loss(net,train_lbls)
    train_op = tf.train.MomentumOptimizer(cfg.learning_rate,cfg.momentum).minimize(t_loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    writer = tf.summary.FileWriter("logs/",sess.graph)
    sess.run(init)

    for i in xrange(cfg.max_steps):
        with sess.as_default():
            print stat.avg_iou.eval()
            if i % 10 == 0:  # Record summaries and test-set accuracy
                print('at step %s' % (i))
            else:  # Record train set summaries, and train
                 _ = sess.run(train_op, feed_dict={train_imgs: images, train_lbls: labels})
        images,labels = voc.get_next_batch()

    train_writer.add_summary(i)
    sess.close()

train()


    
