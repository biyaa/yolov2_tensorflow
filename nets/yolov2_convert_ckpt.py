# -*- coding: utf-8 -*-
"""
  @author: huangguoxiong
"""


import tensorflow as tf
import numpy as np
slim =  tf.contrib.slim
import yolov2
yolo = yolov2

WEIGHT_DIR ='weights/'
OUT_FILE = 'ckpt/yolo_hgx.ckpt'
DATA_IN_MEMORY = {}
DARK_TO_TF ={
        'weights':'weights',
        'BatchNorm/moving_mean':'rolling_mean',
        'BatchNorm/moving_variance':'rolling_variance',
        'BatchNorm/scales':'scales',
        'BatchNorm/beta':'beta',
        'biases':'biases'
        }

def read_txt_data(path,filename):
  f = open(path + filename ,'r')
  
  lines = f.readlines()
  f.close()
  data ={}
  for l in lines:
      d = l.split(",")
      data[d[0]] =d[1:-1]
  
  return data

def get_from_name(full_name):
    name = full_name.split(':')[0]
    names = name.split('/')
    id = int(names[0].replace('conv',''))
    varname = '/'.join(names[1:])
    return id,varname
    
def get_data_from_txt(id,darknet_varname,shape):
    #print id,darknet_varname,shape
    val = ''
    if not DATA_IN_MEMORY.has_key(id):
        DATA_IN_MEMORY[id] = read_txt_data(WEIGHT_DIR , str(id) + '_conv.txt')

    data = DATA_IN_MEMORY[id]
    if darknet_varname == "weights":
        val = np.array(data['weights']).astype('float32')
        print "o-val:",shape
        val = np.reshape(val,(shape[3],shape[2],shape[0],shape[1]))
        print "val:",val.shape
        val = np.transpose(val,(2,3,1,0))
        print "val:",val.shape

    if darknet_varname == "biases":
        val = np.array(data['biases']).astype('float32')

    if darknet_varname == "rolling_mean":
        val = np.array(data['rolling_mean']).astype('float32')
    
    if darknet_varname == "rolling_variance":
        val = np.array(data['rolling_variance']).astype('float32')

    if darknet_varname == "scales":
        val = np.array(data['scales']).astype('float32')
    
    if darknet_varname == "beta":
        val = np.zeros(shape,dtype=np.float32)
    val = np.reshape(val,shape)
    print "val:",val.shape
    return val
        


#init = tf.global_variables_initializer
inputs = np.ones((1,416,416,3),dtype=np.float32)
net = yolo.yolo_net(inputs)
#print net
all_var = yolo.slim.get_model_variables()
with tf.Session() as sess:
    #sess.run(init)
    for v in all_var:
        id,varname = get_from_name(v.name)
        #print id,varname
        varname = DARK_TO_TF[varname]
        shape = v.get_shape()
        value = tf.constant(get_data_from_txt(id,varname,shape))
        sess.run(v.assign(value))
        print v

    saver = tf.train.Saver(tf.global_variables())
    saver.save(sess,OUT_FILE)
    print 'It has convertted from txt to tf-ckpt file!'

