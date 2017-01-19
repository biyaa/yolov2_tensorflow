# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 18:07:21 2016

@author: huangguoxiong
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import nets.yolov2 as yolo
import config.yolov2_config as cfg
import utils.box as box
from utils.timer import Timer
slim = tf.contrib.slim

def get_region_box(boxes,i,j,n,anchors):
    box=np.zeros((4),dtype=np.float32)
    #box[0] = (i + boxes[i,j,n,0])/cfg.cell_size
    #box[1] = (j + boxes[i,j,n,1])/cfg.cell_size
    #box[2] = (anchors[2*n] * boxes[i,j,n,2])/cfg.cell_size
    #box[3] = (anchors[2*n+1] * boxes[i,j,n,3])/cfg.cell_size
    
    box[0] = boxes[i,j,n,0]
    box[1] = boxes[i,j,n,1]
    box[2] = anchors[2*n] * boxes[i,j,n,2]
    box[3] = anchors[2*n+1] * boxes[i,j,n,3]
    return box
    
    
def interpret_output(output):
    #output = np.transpose(output,(0,2,1))
    probs = np.zeros((cfg.cell_size, cfg.cell_size,
                      cfg.boxes_per_cell, cfg.num_class))
    #print output.shape
    info_num = cfg.coords + cfg.scale + cfg.num_class
    #output = np.reshape(output,(cfg.cell_size,cfg.cell_size,cfg.boxes_per_cell,info_num))
    class_probs = np.reshape(output[:,:,:,cfg.coords+1:], (cfg.cell_size, cfg.cell_size,cfg.boxes_per_cell, cfg.num_class))
    #scales = np.reshape(output[:,:,:,cfg.coords], ( cfg.cell_size, cfg.cell_size,cfg.boxes_per_cell, cfg.scale))
    scales = np.zeros((cfg.cell_size, cfg.cell_size,cfg.boxes_per_cell),dtype=np.float32)
    scales = output[:,:,:,cfg.coords]
    print "probs",probs.shape

    probs = class_probs * scales[:,:,:,np.newaxis]
    print probs[probs>0.5]
    print np.where(probs>0.5)

    #offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell), (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

    boxes = output[:,:,:,:cfg.coords]

    for row in xrange(cfg.cell_size):
        for col in xrange(cfg.cell_size):
            for n in xrange(cfg.boxes_per_cell):
                boxes[row,col,n] = get_region_box(boxes,col,row,n,cfg.anchors)

    filter_mat_probs = np.array(probs>=cfg.threshold,dtype='bool')
    filter_mat_boxes = np.nonzero(filter_mat_probs)
    boxes_filtered = boxes[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]]
    probs_filtered = probs[filter_mat_probs]
    classes_num_filtered = np.argmax(filter_mat_probs,axis=3)[filter_mat_boxes[0],filter_mat_boxes[1],filter_mat_boxes[2]] 

    argsort = np.array(np.argsort(probs_filtered))[::-1]
    boxes_filtered = boxes_filtered[argsort]
    probs_filtered = probs_filtered[argsort]
    classes_num_filtered = classes_num_filtered[argsort]
    
    for i in range(len(boxes_filtered)):
        if probs_filtered[i] == 0 : continue
        for j in range(i+1,len(boxes_filtered)):
            if box.box_iou(boxes_filtered[i],boxes_filtered[j]) > cfg.iou_threshold : 
                probs_filtered[j] = 0.0
    
    filter_iou = np.array(probs_filtered>0.0,dtype='bool')
    boxes_filtered = boxes_filtered[filter_iou]
    probs_filtered = probs_filtered[filter_iou]
    classes_num_filtered = classes_num_filtered[filter_iou]
    result = []
    for i in range(len(boxes_filtered)):
        result.append([cfg.cls[classes_num_filtered[i]],boxes_filtered[i][0],boxes_filtered[i][1],boxes_filtered[i][2],boxes_filtered[i][3],probs_filtered[i]])

    return result
    #print "probs",probs.shape
    #probs[probs < cfg.threshold] =0
    #probs_max = np.max(probs,3)
    #probs_max_index = np.argmax(probs,3)

    #probs_mask = np.where(probs_max >= cfg.threshold)
    #probs_masked = probs_max[probs_mask]
    #probs_index_masked = probs_max_index[probs_mask]
    #boxes_masked = boxes[probs_mask]
    #print probs_mask

    ## 按类排序
    #

    #result = []
    #result_num = len(probs_mask[0]) if len(probs_mask)>0 else 0
    #print "num",result_num

    #for i in range(result_num):
    #    result.append([cfg.cls[probs_index_masked[i]], boxes_masked[i][0], boxes_masked[
    #                  i][1], boxes_masked[i][2], boxes_masked[i][3], probs_masked[i]*100])

    #return result

def detect_from_cvmat(inputs):
    #print "inputs:",inputs
    net = yolo.yolo_net(inputs,1,False)
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
        #saver.restore(sess, cfg.out_file)
        saver.restore(sess, "ckpt/yolo.ckpt-4602")
        print "weights restore."
        restore_T.toc()
        print restore_T.average_time
        run_T = Timer()
        run_T.tic()
        net_output = sess.run(net)
        print "net_output:",net_output.shape[0]
        run_T.toc()
        print run_T.average_time
    results = []
    itp_T = Timer()
    itp_T.tic()
    for i in range(net_output.shape[0]):
        results.append(interpret_output(net_output[i]))

    itp_T.toc()
    print itp_T.average_time
    return results

def detect(img):
    #print img
    img_h, img_w, _ = img.shape
    inputs = cv2.resize(img, (cfg.image_size, cfg.image_size)).astype(np.float32)

    #inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
    inputs = (inputs / 255.0) 
    inputs = np.reshape(inputs, (1, cfg.image_size, cfg.image_size, 3))
    #inputs = np.transpose(inputs,(0,3,2,1))

    result = detect_from_cvmat(inputs)[0]

    for i in range(len(result)):
        left = (result[i][1] - result[i][3]/2)*img_w
        right = (result[i][1] + result[i][3]/2)*img_w
        top = (result[i][2] - result[i][4]/2)*img_h
        bot = (result[i][2] + result[i][4]/2)*img_h
        result[i][1] = left if left>0 else 0
        result[i][2] = right if right<img_w-1 else img_w-1
        result[i][3] = top if top>0 else 0
        result[i][4] = bot if bot<img_h-1 else img_h-1

    print "result:", result
    return result

def draw_result(img, result):
    for i in range(len(result)):
        left = int(result[i][1])
        right = int(result[i][2])
        top = int(result[i][3])
        bot = int(result[i][4])
        c = i%3
        color = 200*(c==0), 200*(c==1), 200*(c==2)
        cv2.rectangle(img, (left, top), (right, bot), (color), 5)
        cv2.rectangle(img, (left, top + 20),
                      (right, top+1), (color), -1)
        cv2.putText(img, result[i][0] + ' : %.2f' % result[i][5], (left+ 15, top -7 + 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.CV_AA)

def image_detector(imname, wait=0):
    print imname
    detect_timer = Timer()
    image = cv2.imread(imname,1)
    print image.shape

    detect_timer.tic()
    result = detect(image)
    detect_timer.toc()
    print 'Average detecting time: {:.3f}s'.format(detect_timer.average_time)

    draw_result(image, result)
    cv2.imwrite('prediction.jpg', image)
    cv2.imshow('Image', image)
    cv2.waitKey(wait)


image_detector(cfg.test_img)
