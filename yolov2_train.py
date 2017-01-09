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
import numpy as np
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
def box_iou(pred,truth):
    p_x = pred[...,0:1]
    p_y = pred[...,1:2]
    p_w = pred[...,2:3]
    p_h = pred[...,3:4]

    p_l = p_x - p_w/2
    p_r = p_x + p_w/2
    p_t = p_y - p_h/2
    p_b = p_y + p_h/2

    t_x = truth[...,0:1]
    t_y = truth[...,1:2]
    t_w = truth[...,2:3]
    t_h = truth[...,3:4]

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

def box_iou_by_truths(preds,truths,b,i,j,t):
    preds00 = preds[b,j,i,:,:]
    truths00 = truths[b,t,:]
    #print preds00,truths00
    # coords
    p_x = preds00[...,0]
    p_y = preds00[...,1]
    p_w = preds00[...,2]
    p_h = preds00[...,3]

    p_l = p_x - p_w/2
    p_r = p_x + p_w/2
    p_t = p_y - p_h/2
    p_b = p_y + p_h/2

    t_x = truths00[...,0:1]
    t_y = truths00[...,1:2]
    t_w = truths00[...,2:3]
    t_h = truths00[...,3:4]

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
    # 4. nan 处理
    predicts = tf.clip_by_value(predicts,1e-10,1e10)
    p_coords = predicts[...,:5]
    p_classes = predicts[...,5:]
    p_x = predicts[...,0:1]
    p_y = predicts[...,1:2]
    p_w = predicts[...,2]
    p_h = predicts[...,3]
    p_c = predicts[...,4:5] #scale
    
    p_zero = tf.zeros_like(p_x)
    p_index = tf.where(p_zero > -1)
    print p_index, p_x
    p_row = tf.reshape(tf.to_float(p_index[...,1:2]),p_x.get_shape())
    p_col = tf.reshape(tf.to_float(p_index[...,2:3]),p_x.get_shape())
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

    #p_classes = p_classes * p_c
    #p_probs = p_classes * p_c
    net_out = tf.concat(4,[p_x,p_y,p_w,p_h,p_c,p_classes])
    net_out = tf.clip_by_value(net_out,1e-10,1e10)
    return net_out 

def sigmoid_gradient(x):
    return (1-x)*x

def _delta_obj_scales(pred_scales,cfg_scale,iou):
    d_scales = cfg_scale * ((iou - pred_scales) * sigmoid_gradient(pred_scales))
    return d_scales

def _delta_noobj_scales(pred_scales,cfg_scale):
    d_scales = cfg_scale * ((0 - pred_scales) * sigmoid_gradient(pred_scales))
    return d_scales


def do_assign(ref,v,value):
    return tf.assign(ref,v.assign(value))

def _delta_region_box(net,truths_in_net):
    #print pre[4]

    t_x = truths_in_net[...,0:1]
    t_y = truths_in_net[...,1:2]
    t_w = truths_in_net[...,2:3]
    t_h = truths_in_net[...,3:4]

    delta_x = cfg.coord_scale * (t_x - tf.sigmoid(net[...,0:1])) * sigmoid_gradient(tf.sigmoid(net[...,0:1]))
    delta_y = cfg.coord_scale * (t_y - tf.sigmoid(net[...,1:2])) * sigmoid_gradient(tf.sigmoid(net[...,1:2]))
    delta_w = cfg.coord_scale * (t_w - net[...,2:3])
    delta_h = cfg.coord_scale * (t_h - net[...,3:4])
#    print delta_x



    return tf.concat(4,[delta_x,delta_y,delta_w,delta_h])


def _delta_region_class(net_out,truths_in_net):
    delta_region_class = cfg.class_scale * (truths_in_net[...,5:] - net_out[...,5:])
    return delta_region_class

#def _delta_by_truths(truths,net,preds,delta):
#    t_x = truths[...,0:1]
#    t_y = truths[...,1:2]
#    t_x = t_x * cfg.cell_size
#    t_y = t_y * cfg.cell_size
#
#    t_shift_x = t_x * 0
#    t_shift_y = t_y * 0
#
#    truths_shift = tf.concat(2,[t_shift_x,t_shift_y,truths[...,2:4]])
#
#    p_x = preds[...,0:1]
#    p_y = preds[...,1:2]
#    p_shift_x = p_x * 0
#    p_shift_y = p_y * 0
#    #print preds[:,1,1,:,:]
#    #print truths
#
#    preds_shift = tf.concat(4,[p_shift_x,p_shift_y,preds[...,2:4]])
#
#    # 8. compute shift_iou and best_shift_iou_by_pred
#    #shift_iou = box_iou_by_pred(preds_shift,truths_shift)
#    #print shift_iou
#    #best_shift_iou_by_pred = tf.reduce_max(shift_iou,-1,keep_dims=True)
#    #print best_shift_iou_by_pred
#    
#    #print t_x
#    for b in xrange(truths.get_shape()[0]):
#        for t in xrange(30):
#            i = t_x[b,t,0]
#            j = t_y[b,t,0]
#            i =tf.to_int32(i)
#            j =tf.to_int32(j)
#            shift_iou = box_iou_by_truths(preds_shift,truths_shift,b,i,j,t)
#            best_shift_iou = tf.reduce_max(shift_iou,-1,keep_dims=True)
#            #print shift_iou
#            best_shift_n = tf.arg_max(shift_iou,0)
#            best_shift_n =tf.to_int32(best_shift_n)
#        
#            #print best_shift_n
#            #delta_box = _delta_region_box(truths,net,preds,cfg.anchors,cfg.coord_scale,b,i,j,best_shift_n,t,delta)
#            #print delta_box
#
#    return delta_box
#



def loss(net,labels,delta_mask,truths_in_net):
    net_out = tf_post_process(net)
    #delta = tf.Variable(tf.zeros(net_out.get_shape()))

    # 1. init delta
    #delta_scales = delta[...,4:5]
    #delta_region_box = delta[...,:4]
    #print delta_mask

    


    stat.set_zero()
    # 2. compute delta_scales
    delta_scales = _delta_noobj_scales(net_out[...,4:5],cfg.noobject_scale)
    

    # 3. compute avg_anyobj
    stat.avg_anyobj = tf.reduce_sum(net_out[...,4:5])
    # 4. compute best_iou
    iou_by_pred = box_iou_by_pred(net_out,labels)
    best_iou_by_pred = tf.reduce_max(iou_by_pred,-1,keep_dims=True)
    #print best_iou_by_pred
    # 5. in delta_scales, select best_iou>threshold to set 0
    cond = best_iou_by_pred > cfg.threshold
    delta_scales = tf.where(cond,delta_scales*0,delta_scales)
    
    # 6. compose of truths_shift
    truths = labels[...,:4]
    t_x = truths[...,0:1]
    t_y = truths[...,1:2]
    t_x = t_x * cfg.cell_size
    t_y = t_y * cfg.cell_size

    t_shift_x = t_x * 0
    t_shift_y = t_y * 0

    truths_shift = tf.concat(2,[t_shift_x,t_shift_y,truths[...,2:4]])



    # 7. compose of preds_shift
    preds = net_out[...,:4]
    #print preds


    p_x = preds[...,0:1]
    p_y = preds[...,1:2]
    p_shift_x = p_x * 0
    p_shift_y = p_y * 0
    #print preds[:,1,1,:,:]
    #print truths

    preds_shift = tf.concat(4,[p_shift_x,p_shift_y,preds[...,2:4]])

    iou_shift_by_pred = box_iou_by_pred(preds_shift,truths_shift)
    #print iou_shift_by_pred
    best_iou_shift_boxes = tf.reduce_max(iou_shift_by_pred,-1,keep_dims=True)
    best_iou_shift_n = tf.reduce_max(best_iou_shift_boxes,-2,keep_dims=True)
    best_iou_shift_boxes = best_iou_shift_boxes - best_iou_shift_n

    best_iou_mask = best_iou_shift_boxes>=0
    #stat.count = tf.count_nonzero(delta_mask[...,4:5])
    delta_mask = tf.logical_and(delta_mask, best_iou_mask)
    #stat.count = tf.count_nonzero(delta_mask[...,4:5])

    #print delta_mask
    mask_iou = tf.where(delta_mask[...,0:1], best_iou_by_pred, best_iou_by_pred*0)
    # print avg_iou
    stat.avg_iou = tf.reduce_sum(mask_iou)
    stat.recall = tf.count_nonzero(mask_iou>0.5) 
    delta_obj_scales = _delta_obj_scales(net_out[...,4:5], cfg.object_scale, mask_iou)
    delta_scales = tf.where(delta_mask[...,4:5], delta_obj_scales, delta_scales)
    mask_obj = tf.where(delta_mask[...,4:5], delta_scales,delta_scales*0)
    #print avg_obj
    #print mask_obj
    
    stat.avg_obj = tf.reduce_sum(mask_obj)
    stat.count = tf.count_nonzero(mask_obj)
    print delta_scales
    #print best_iou_mask





    # 9. compute delta_region_box
    delta_region_box = _delta_region_box(net, truths_in_net)
    delta_region_box = tf.where(delta_mask[...,0:4], delta_region_box, delta_region_box * 0)
    #delta = _delta_by_truths(truths,net,preds,delta)
    print delta_region_box
    
    # 10. compute delta_region_class
    delta_region_class = _delta_region_class(net_out,truths_in_net)
    delta_region_class = tf.where(delta_mask[...,5:],delta_region_class, delta_region_class * 0)
    stat.avg_cat = tf.reduce_sum(delta_region_class)

    delta = tf.concat(4, [delta_region_box,delta_scales,delta_region_class])
    print delta
    stat.cost = tf.reduce_sum(tf.pow(delta,2))
    return delta

def _delta_mask(labels):
    shape = labels.shape
    batch = shape[0]
    box_num = shape[1]
    # init mask
    delta_shape = (labels.shape[0],cfg.cell_size,cfg.cell_size,5,85)
    delta_mask = np.zeros(delta_shape,dtype=np.bool)
    delta_mask[...] = False
    x = 0
    y = 0
    c = 0
    # compute mask
    for b in xrange(batch):
        for n in xrange(box_num):
            if labels[b,n,0] > 0.00001:
                x = np.int(labels[b,n,0] * cfg.cell_size)
                y = np.int(labels[b,n,1] * cfg.cell_size)
                c = np.int(labels[b,n,4])
                delta_mask[b,y,x,:,5+c] = True
                delta_mask[b,y,x,:,0:5] = True
                #print delta_mask[b,y,x,:,:]
                
    #print "label count:%s" %(np.count_nonzero(delta_mask)/30)
    return delta_mask

def _truths_in_net(labels):
    shape = labels.shape
    batch = shape[0]
    box_num = shape[1]
    # init mask
    net_shape = (labels.shape[0],cfg.cell_size,cfg.cell_size,5,85)
    truths_in_net = np.zeros(net_shape,dtype=np.float32)
    x = 0
    y = 0
    c = 0
    # compute truths_in_net
    for b in xrange(batch):
        for n in xrange(box_num):
            if labels[b,n,0] > 0.00001:
                x = np.int(labels[b,n,0] * cfg.cell_size)
                y = np.int(labels[b,n,1] * cfg.cell_size)
                c = np.int(labels[b,n,4])
                truths_in_net[b,y,x,:,5+c] = 1.0
                truths_in_net[b,y,x,:,0:1] = labels[b,n,0:1] * cfg.cell_size - x
                truths_in_net[b,y,x,:,1:2] = labels[b,n,1:2] * cfg.cell_size - y
                for bp in xrange(cfg.boxes_per_cell):
                    truths_in_net[b,y,x,bp,2:3] = np.log(labels[b,n,2:3] * cfg.cell_size / cfg.anchors[2*bp])
                    truths_in_net[b,y,x,bp,3:4] = np.log(labels[b,n,3:4] * cfg.cell_size / cfg.anchors[2*bp+1])

    return truths_in_net




    
def _train(images,labels):
    return loss(images,labels)

def train():
    log_dir = cfg.train_log_path

    images,labels = voc.get_next_batch()
    delta_mask = _delta_mask(labels)
    truths_in_net = _truths_in_net(labels)

    train_imgs = tf.placeholder(dtype=tf.float32,shape=images.shape)
    train_lbls = tf.placeholder(dtype=tf.float32,shape=labels.shape)
    train_mask = tf.placeholder(dtype=tf.bool,shape=delta_mask.shape)
    train_truthinnet = tf.placeholder(dtype=tf.float32,shape=truths_in_net.shape)

    global_step = tf.Variable(0, trainable=False, name='global_step')

    net = yolo.yolo_net(train_imgs,images.shape[0],trainable=True)
    t_loss = loss(net,train_lbls,train_mask,train_truthinnet)
    train_op = tf.train.MomentumOptimizer(cfg.learning_rate,cfg.momentum).minimize(t_loss, global_step=global_step)




    saver = tf.train.Saver(yolo.slim.get_model_variables())
    saver1 = tf.train.Saver()#yolo.slim.get_model_variables())
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    saver.restore(sess,cfg.out_file)

    print "Weights restored."
    writer = tf.summary.FileWriter("logs/",sess.graph)

    avg_iou = 0
    avg_obj = 0
    avg_noobj = 0
    count = 0
    recall = 0
    avg_cat = 0
    cost = 0

    for i in xrange(cfg.max_steps):
        with sess.as_default():
            feed_dict = {train_imgs: images, train_lbls: labels,train_mask:delta_mask,train_truthinnet:truths_in_net}
            train_ops = [train_op,stat.avg_iou,stat.avg_obj,stat.avg_anyobj,stat.count,stat.recall,stat.avg_cat,stat.cost]
            _,avg_iou,avg_obj,avg_noobj,count,recall,avg_cat,cost= sess.run(train_ops, feed_dict=feed_dict)

            if i % 10 == 0:
                print "label count:%s" %(np.count_nonzero(delta_mask)/(30))
                print('step:%s,cost:%s,avg_iou:%s,avg_obj:%s,avg_noobj:%s,count:%s,recall:%s,avg_class:%s' % (i,cost,avg_iou,avg_obj,avg_noobj,count,recall,avg_cat))

            if i % 500 == 0:
                print('global_step: %s' % tf.train.global_step(sess, global_step))
                saver1.save(sess,'ckpt/yolo.ckpt', global_step=global_step)

        
        images,labels = voc.get_next_batch()
        delta_mask = _delta_mask(labels)
        truths_in_net = _truths_in_net(labels)

    train_writer.add_summary(i)
    sess.close()
if __name__ == "__main__":
    train()


    
