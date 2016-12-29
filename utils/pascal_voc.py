# -*- coding: utf-8 -*-
"""
    yolov2_tensorflow.pascal_voc
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Created on 2016-12-26 17:59
    @author : huangguoxiong
    copyright: (c) 2016 by huangguoxiong.
    license: Apache license, see LICENSE for more details.
"""

import os
import random as rand
import numpy as np
import cv2
import tensorflow as tf 

cfg = "no config be set"

def set_config(config):
    global cfg 
    cfg = config

def print_config():
    print cfg.cell_size

def _get_train_paths():
    with open(cfg.train_data_path,'r') as f:
        train_img_paths=f.readlines()

    global train_paths
    train_paths=[]
    for img in train_img_paths:
        img = img.replace('\n','')
        label = img.replace('/JPEGImages/','/labels/')
        label = label[:label.rindex('.')] + '.txt'
        train_paths.append([img,label])

    #print len(train_paths)
    return train_paths 

def _get_rand_one_path():
    i = rand.randint(0,len(train_paths)-1)
    #print i
    return train_paths[i]

def read_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (cfg.image_size, cfg.image_size))
    image = image /255.0
    return image


def read_label(path):
    print path
    label = np.loadtxt(path,dtype=np.float32)
    if len(label.shape)<2:
        label = label[np.newaxis,...]
    label = np.transpose(label,(1,0))
    print label,label.shape

    return label


def get_next_batch():
    images = np.zeros((cfg.batch_size,cfg.image_size,cfg.image_size,3),dtype=np.float32)
    labels = np.zeros((cfg.batch_size,5,30),dtype=np.float32)
    # load paths from a file
    _get_train_paths()

    for i in xrange(cfg.batch_size):
        path = _get_rand_one_path()
        images[i] = read_image(path[0])

        label= read_label(path[1])
        box_num = label.shape[-1]
        labels[i,:,0:box_num] = label
        #print labels[i].shape

    return images,labels

