# -*- coding: utf-8 -*-
"""
  @author: huangguoxiong
"""


import tensorflow as tf
import numpy as np
import os.path 
slim =  tf.contrib.slim
# 1. 确定PAD类型为SAME
# 2. 确定weights的shape [size,size,channels,filters]
# 3. 确定卷积顺序 conv -> BatchNorm -> scale -> add bias -> leaky_relu 
# 4. 解决 reorg 问题 自定义op实现重组逻辑
DATA_FORMAT_NCHW = 'NCHW'
DATA_FORMAT_NHWC = 'NHWC'
DEVICE ='/CPU:0'
ALPHA = 0.1
#_reorg_module = tf.load_op_library(
            #os.path.join(tf.resource_loader.get_data_files_path(),
                                #'core/re_org.so'))
#reorg_func = _reorg_module.re_org
#def cond_cls_region(
def yolo_net(inputs,batch_size,trainable):
        layer0 = conv2d(inputs, 32, [3, 3], 1, padding='SAME', scope='conv0',trainable=trainable)
        layer1 = slim.max_pool2d(layer0, [2, 2], scope='pool1')
        layer2 = conv2d(layer1, 64, [3, 3], 1, padding='SAME', scope='conv2',trainable=trainable)
        layer3 = slim.max_pool2d(layer2, [2, 2], scope='pool3')
        layer4 = conv2d(layer3, 128, [3, 3], 1, padding='SAME', scope='conv4',trainable=trainable)
        layer5 = conv2d(layer4, 64, [1, 1], 1, padding='SAME', scope='conv5',trainable=trainable)
        layer6 = conv2d(layer5, 128, [3, 3], 1, padding='SAME', scope='conv6',trainable=trainable)
        layer7 = slim.max_pool2d(layer6, [2, 2], scope='pool7')
        layer8 = conv2d(layer7, 256, [3, 3], 1, padding='SAME', scope='conv8',trainable=trainable)
        layer9 = conv2d(layer8, 128, [1, 1], 1, padding='SAME', scope='conv9',trainable=trainable)
        layer10 = conv2d(layer9, 256, [3, 3], 1, padding='SAME', scope='conv10',trainable=trainable)
        layer11 = slim.max_pool2d(layer10, [2, 2], scope='pool11')
        layer12 = conv2d(layer11, 512, [3, 3], 1, padding='SAME', scope='conv12',trainable=trainable)
        layer13 = conv2d(layer12, 256, [1, 1], 1, padding='SAME', scope='conv13',trainable=trainable)
        layer14 = conv2d(layer13, 512, [3, 3], 1, padding='SAME', scope='conv14',trainable=trainable)
        layer15 = conv2d(layer14, 256, [1, 1], 1, padding='SAME', scope='conv15',trainable=trainable)
        layer16 = conv2d(layer15, 512, [3, 3], 1, padding='SAME', scope='conv16',trainable=trainable)
        layer17 = slim.max_pool2d(layer16, [2, 2], scope='pool17')
        layer18 = conv2d(layer17, 1024, [3, 3], 1, padding='SAME', scope='conv18',trainable=trainable)
        layer19 = conv2d(layer18, 512, [1, 1], 1, padding='SAME', scope='conv19',trainable=trainable)
        layer20 = conv2d(layer19, 1024, [3, 3], 1, padding='SAME', scope='conv20',trainable=trainable)
        layer21 = conv2d(layer20, 512, [1, 1], 1, padding='SAME', scope='conv21',trainable=trainable)
        layer22 = conv2d(layer21, 1024, [3, 3], 1, padding='SAME', scope='conv22',trainable=trainable)
        layer23 = conv2d(layer22, 1024, [3, 3], 1, padding='SAME', scope='conv23',trainable=trainable)
        layer24 = conv2d(layer23, 1024, [3, 3], 1, padding='SAME', scope='conv24',trainable=trainable)
        layer25 = layer16
        #layer26 = reorg(layer25,(batch_size,13,13,2048))
        layer26 = slim.array_ops.space_to_depth(layer25,2)
        layer27 = tf.concat([layer26,layer24],3)
        #layer27 = slim.array_ops.concat(3,[layer26,layer24])
        layer28 = conv2d(layer27, 1024, [3, 3], 1, padding='SAME', scope='conv28',trainable=trainable)
        layer29 = conv2d_with_linear(layer28, 425, [1, 1], 1, padding='SAME', scope='conv29',trainable=trainable)
        layer29 = slim.array_ops.reshape(layer29,[batch_size,13,13,5,85])
        #print layer29


#        layer30 = slim.nn.sigmoid(layer29[:,:,:,:,0:2],name="coords_xy")
#        # (x + col)/w
#        layer31 = xy_add_cr_div_size(layer30[:,:,:,:,0:1],2,13,name="x_add_col_div_size")
#        # (y + row)/h
#        layer32 = xy_add_cr_div_size(layer30[:,:,:,:,1:2],1,13,name="x_add_row_div_size")
#        layer33 = slim.math_ops.exp(layer29[:,:,:,:,2:4],name="coords_wh")
#        layer33 = layer33 / 13
#        layer34 = slim.nn.sigmoid(layer29[:,:,:,:,4:5],name="scale")
#
#        layer40 = tf.reduce_max(layer29[:,:,:,:,5:],4,name="max_class")
#        layer41 = slim.array_ops.expand_dims(layer40,-1,name="scale")
#        layer42 = slim.math_ops.sub(layer29[:,:,:,:,5:],layer41,name="sub_max")
#        layer43 = slim.nn.softmax(layer42,name="class_softmax")
#
#        layer50 = slim.array_ops.concat(4,[layer31,layer32,layer33,layer34,layer43])
#        print layer34
#        print layer43
#        
        return layer29
def xy_add_cr_div_size(inputs,cr,cell_size,name):
    shape = inputs.get_shape()
    #print shape
    addn = slim.array_ops.where(inputs>-999999)
    addn = addn[...,cr]
    addn = slim.array_ops.expand_dims(addn,-1)
    addn = tf.to_float(addn)
    #print addn
    inputs = slim.array_ops.reshape(inputs,[shape[0].value,-1,1])
    inputs = inputs + addn 
    inputs = inputs / cell_size 
    #print inputs
    return slim.array_ops.reshape(inputs,shape)


#def wh_mul_anchor_of_n(inputs,n,anchors,name):
#    shape = inputs.get_shape()
#    #print shape
#    nn = slim.array_ops.where(inputs>-999999)
#    nn = nn[...,n]
#    nn = slim.array_ops.expand_dims(nn,-1)
#    nn = tf.to_float(nn)
#    #print nn
#    inputs = slim.array_ops.reshape(inputs,[shape[0].value,-1,1)
#    inputs = inputs * anchors(2*nnk
#    inputs = inputs / cell_size 
#    #print inputs
#
def reorg_bak(inputs,shape):
    #print inputs
    #body = lambda x:
    inputs = slim.array_ops.transpose(inputs,perm=[0,3,1,2])

    print(inputs)
    output = reorg_func(inputs)
    output = slim.array_ops.transpose(output,perm=[0,2,3,1])
    #print output
    return slim.array_ops.reshape(output,shape)
    

def reorg(inputs,shape):
    #output = tf.Variable(tf.zeros(shape,tf.float32))
    c = shape[-1]

    output1 = inputs[:,0:4:4,0:4:4,0:6:2]
    output2 = inputs[:,0:4:4,::4,1::2]
    output3 = inputs[:,::4,2::4,::2]
    output4 = inputs[:,::4,2::4,1::2]
    #output = output[:,:,:,0:s].assign(inputs[:,::2,::2,:])
    #output = output[:,:,:,s:2*s].assign(inputs[:,::2,1::2,:])
    #output = output[:,:,:,2*s:3*s].assign(inputs[:,1::2,::2,:])
    #output = output[:,:,:,3*s:4*s].assign(inputs[:,1::2,1::2,:])
    #print output

    return tf.concat(1,[output1,output2,output3,output4])


def conv2d(inputs,filters,kernel_size,stride,padding,scope,trainable):
    with slim.arg_scope([slim.conv2d],
                      activation_fn=None,
                      trainable=trainable,
                      normalizer_fn=slim.batch_norm,
                      normalizer_params={
                          'scale':True,
                          'center':False,
                          'epsilon':.0000001
                          },
                      weights_initializer=tf.constant_initializer(0)):
        part1 = slim.conv2d(inputs, filters, kernel_size, stride, padding='SAME', scope=scope)
        #part2 = scale_bias(part1,scope=scope)
        part3 = slim.bias_add(part1,scope=scope)
        part4 = leaky_relu(part3)
        return part4

def conv2d_with_linear(inputs,filters,kernel_size,stride,padding,scope,trainable):
    with slim.arg_scope([slim.conv2d],
                      activation_fn=None,
                      normalizer_fn=None,
                      trainable=trainable,
                      weights_initializer=tf.constant_initializer(0),
                      biases_initializer=tf.constant_initializer(0)):
        part1 = slim.conv2d(inputs, filters, kernel_size, stride, padding='SAME', scope=scope)
        return part1
#data_format="NCHW",
def scale_bias(inputs,data_format=DATA_FORMAT_NHWC,scope='BatchNorm'):
    with tf.variable_scope(scope +'/BatchNorm') as scope:
        inputs_shape = inputs.get_shape()
        axis = 1 if data_format==DATA_FORMAT_NCHW else -1
        num_features = inputs_shape[axis].value
        if num_features is None:
          raise ValueError('`C` dimension must be known but is None')
        scales = slim.model_variable('scales',
                          shape=[num_features,],
                          initializer=tf.constant_initializer(0),
                          device=DEVICE)
        return slim.math_ops.mul(inputs,scales)

def leaky_relu(inputs):
    return slim.math_ops.maximum(inputs,ALPHA*inputs,"leaky_relu")
