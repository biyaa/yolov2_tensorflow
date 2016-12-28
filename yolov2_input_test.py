# -*- coding: utf-8 -*-
"""
    yolov2_tensorflow.yolov2_input_test
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Created on 2016-12-27 09:49
    @author : huangguoxiong
    copyright: (c) 2016 by huangguoxiong.
    license: Apache license, see LICENSE for more details.
"""

import tensorflow as tf
import config.yolov2_config as cfg
import utils.pascal_voc as voc

voc.set_config(cfg)
#voc.print_config()

# Load the images and labels.
#voc._get_train_paths()
voc.get_next_batch()
#voc.read_file()
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
