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
DEVICE ='/GPU:0'
ALPHA = 0.1
#_reorg_module = tf.load_op_library(
            #os.path.join(tf.resource_loader.get_data_files_path(),
                                #'core/re_org.so'))
#reorg_func = _reorg_module.re_org
#def cond_cls_region(
def yolo_net(inputs,batch_size):
        layer0 = conv2d(inputs, 32, [3, 3], 1, padding='SAME', scope='conv0')
        layer1 = slim.max_pool2d(layer0, [2, 2], scope='pool1')
        layer2 = conv2d(layer1, 64, [3, 3], 1, padding='SAME', scope='conv2')
        layer3 = slim.max_pool2d(layer2, [2, 2], scope='pool3')
        layer4 = conv2d(layer3, 128, [3, 3], 1, padding='SAME', scope='conv4')
        layer5 = conv2d(layer4, 64, [1, 1], 1, padding='SAME', scope='conv5')
        layer6 = conv2d(layer5, 128, [3, 3], 1, padding='SAME', scope='conv6')
        layer7 = slim.max_pool2d(layer6, [2, 2], scope='pool7')
        layer8 = conv2d(layer7, 256, [3, 3], 1, padding='SAME', scope='conv8')
        layer9 = conv2d(layer8, 128, [1, 1], 1, padding='SAME', scope='conv9')
        layer10 = conv2d(layer9, 256, [3, 3], 1, padding='SAME', scope='conv10')
        layer11 = slim.max_pool2d(layer10, [2, 2], scope='pool11')
        layer12 = conv2d(layer11, 512, [3, 3], 1, padding='SAME', scope='conv12')
        layer13 = conv2d(layer12, 256, [1, 1], 1, padding='SAME', scope='conv13')
        layer14 = conv2d(layer13, 512, [3, 3], 1, padding='SAME', scope='conv14')
        layer15 = conv2d(layer14, 256, [1, 1], 1, padding='SAME', scope='conv15')
        layer16 = conv2d(layer15, 512, [3, 3], 1, padding='SAME', scope='conv16')
        layer17 = slim.max_pool2d(layer16, [2, 2], scope='pool17')
        layer18 = conv2d(layer17, 1024, [3, 3], 1, padding='SAME', scope='conv18')
        layer19 = conv2d(layer18, 512, [1, 1], 1, padding='SAME', scope='conv19')
        layer20 = conv2d(layer19, 1024, [3, 3], 1, padding='SAME', scope='conv20')
        layer21 = conv2d(layer20, 512, [1, 1], 1, padding='SAME', scope='conv21')
        layer22 = conv2d(layer21, 1024, [3, 3], 1, padding='SAME', scope='conv22')
        layer23 = conv2d(layer22, 1024, [3, 3], 1, padding='SAME', scope='conv23')
        layer24 = conv2d(layer23, 1024, [3, 3], 1, padding='SAME', scope='conv24')
        layer25 = layer16
        layer26 = reorg(layer25,(batch_size,13,13,2048))
        layer27 = slim.array_ops.concat(3,[layer26,layer24])
        layer28 = conv2d(layer27, 1024, [3, 3], 1, padding='SAME', scope='conv28')
        layer29 = conv2d_with_linear(layer28, 425, [1, 1], 1, padding='SAME', scope='conv29')
        layer29 = slim.array_ops.reshape(layer29,[batch_size,13,13,5,85])
        #print layer29


        layer30 = slim.nn.sigmoid(layer29[:,:,:,:,0:2],name="coords_xy")
        # (x + col)/w
        layer31 = xy_add_cr_div_size(layer29[:,:,:,:,0:1],2,13,name="x_add_col_div_size")
        # (y + row)/h
        layer32 = xy_add_cr_div_size(layer29[:,:,:,:,1:2],1,13,name="x_add_row_div_size")
        layer33 = slim.math_ops.exp(layer29[:,:,:,:,2:4],name="coords_wh")
        layer33 = layer33 / 13
        layer34 = slim.nn.sigmoid(layer29[:,:,:,:,4:5],name="scale")

        layer40 = tf.reduce_max(layer29[:,:,:,:,5:],4,name="max_class")
        layer41 = slim.array_ops.expand_dims(layer40,-1,name="scale")
        layer42 = slim.math_ops.sub(layer29[:,:,:,:,5:],layer41,name="sub_max")
        layer43 = slim.nn.softmax(layer42,name="class_softmax")

        layer50 = slim.array_ops.concat(4,[layer31,layer32,layer33,layer34,layer43])
        print layer34
        print layer43
        
        return layer50

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



def reorg_bak(inputs,shape):
    print inputs
    #body = lambda x:
    inputs = slim.array_ops.transpose(inputs,perm=[0,3,1,2])

    print inputs
    output = reorg_func(inputs)
    output = slim.array_ops.transpose(output,perm=[0,2,3,1])
    print output
    return slim.array_ops.reshape(output,shape)
    

def reorg(inputs,shape):
    output = tf.Variable(tf.zeros([shape[0],52,52,128],tf.float32))

    output[:,::2,::2,:].assign(inputs[:,:,:,0:128])
    output[:,::2,1::2,:].assign(inputs[:,:,:,128:256])
    output[:,1::2,::2,:].assign(inputs[:,:,:,256:384])
    output[:,1::2,1::2,:].assign(inputs[:,:,:,384:512])
    #print output

    return slim.array_ops.reshape(output,shape)


def conv2d(inputs,filters,kernel_size,stride,padding,scope):
    with slim.arg_scope([slim.conv2d],
                      activation_fn=None,
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

def conv2d_with_linear(inputs,filters,kernel_size,stride,padding,scope):
    with slim.arg_scope([slim.conv2d],
                      activation_fn=None,
                      normalizer_fn=None,
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
    return slim.math_ops.maximum(inputs,ALPHA*inputs)
