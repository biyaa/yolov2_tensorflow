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
ALPHA = 0.1
_reorg_module = tf.load_op_library(
            os.path.join(tf.resource_loader.get_data_files_path(),
                                'core/re_org.so'))
reorg_func = _reorg_module.re_org
def yolo_net(inputs):
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
        #layer26 = slim.array_ops.reshape(layer25,shape=(1,13,13,2048))
        layer26 = reorg(layer25,(1,13,13,2048))
        layer27 = slim.array_ops.concat(3,[layer26,layer24])
        layer28 = conv2d(layer27, 1024, [3, 3], 1, padding='SAME', scope='conv28')
        layer29 = conv2d_with_linear(layer28, 425, [1, 1], 1, padding='SAME', scope='conv29')
        
        return layer29

def reorg(inputs,shape):
    print inputs.get_shape()
    #body = lambda x:

    output = reorg_func(inputs)
    output = slim.array_ops.transpose(output,perm=[0,2,3,1])
    print output
    return slim.array_ops.reshape(output,shape)
    



def conv2d(inputs,filters,kernel_size,stride,padding,scope):
    with slim.arg_scope([slim.conv2d],
                      activation_fn=None,
                      normalizer_fn=slim.batch_norm,
                      normalizer_params={
                          'decay':0.0005,
                          },
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005)):
        part1 = slim.conv2d(inputs, filters, kernel_size, stride, padding='SAME', scope=scope)
        part2 = scale_bias(part1,scope=scope)
        part3 = slim.bias_add(part2,scope=scope)
        part4 = leaky_relu(part3)
        return part4

def conv2d_with_linear(inputs,filters,kernel_size,stride,padding,scope):
    with slim.arg_scope([slim.conv2d],
                      activation_fn=None,
                      normalizer_fn=None,
                      weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      weights_regularizer=slim.l2_regularizer(0.0005),
                      biases_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                      biases_regularizer=slim.l2_regularizer(0.0005)):
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
                          initializer=tf.truncated_normal_initializer(stddev=0.1),
                          regularizer=slim.l2_regularizer(0.05),
                          device='/CPU:0')
        return slim.math_ops.mul(inputs,scales)

def leaky_relu(inputs):
    return slim.math_ops.maximum(inputs,ALPHA*inputs)
