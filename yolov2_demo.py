import tensorflow as tf
import numpy as np
import nets.yolov2 as yolo
import os
import cv2
import config.config as cfg
from utils.timer import Timer
slim = tf.contrib.slim
def logistic_activate(x):
    return 1./(1. + np.exp(-x))

def get_region_box(boxes,i,j,n,anchors):
    box=np.zeros((4),dtype=np.float32)
    box[0] = (i + boxes[i,j,n,0])/cfg.cell_size
    box[1] = (j + boxes[i,j,n,1])/cfg.cell_size
    #box[2] = (0.5 * logistic_activate(boxes[i,j,n,2]))/cfg.cell_size
    #box[3] = (0.5 * logistic_activate(boxes[i,j,n,3]))/cfg.cell_size
    box[2] = (anchors[2*n] * boxes[i,j,n,2])/cfg.cell_size
    box[3] = (anchors[2*n] * boxes[i,j,n,3])/cfg.cell_size
    return box

def iou(box1, box2):
    tb = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
        max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
    lr = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
        max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
    if tb < 0 or lr < 0:
        intersection = 0
    else:
        intersection = tb * lr
    return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)
    
    
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

    #offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell), (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

    boxes = output[:,:,:,:cfg.coords]

    for i in xrange(cfg.cell_size):
        for j in xrange(cfg.cell_size):
            for n in xrange(cfg.boxes_per_cell):
                boxes[i,j,n] = get_region_box(boxes,i,j,n,cfg.anchors)

    #boxes *= cfg.image_size

    for i in range(cfg.boxes_per_cell):
        for j in range(cfg.num_class):
            probs[:, :, i, j] = np.multiply(
                class_probs[:, :, i,j], scales[:, :, i])

    filter_mat_probs = np.array(probs >= cfg.threshold, dtype='bool')
    filter_mat_boxes = np.nonzero(filter_mat_probs)
    boxes_filtered = boxes[filter_mat_boxes[0],
                           filter_mat_boxes[1], filter_mat_boxes[2]]
    probs_filtered = probs[filter_mat_probs]
    classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[filter_mat_boxes[
        0], filter_mat_boxes[1], filter_mat_boxes[2]]

    argsort = np.array(np.argsort(probs_filtered))[::-1]
    boxes_filtered = boxes_filtered[argsort]
    probs_filtered = probs_filtered[argsort]
    classes_num_filtered = classes_num_filtered[argsort]

    for i in range(len(boxes_filtered)):
        if probs_filtered[i] == 0:
            continue
        for j in range(i + 1, len(boxes_filtered)):
            if iou(boxes_filtered[i], boxes_filtered[j]) > cfg.iou_threshold:
                probs_filtered[j] = 0.0

    filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
    boxes_filtered = boxes_filtered[filter_iou]
    probs_filtered = probs_filtered[filter_iou]
    classes_num_filtered = classes_num_filtered[filter_iou]

    result = []
    for i in range(len(boxes_filtered)):
        result.append([cfg.cls[classes_num_filtered[i]], boxes_filtered[i][0], boxes_filtered[
                      i][1], boxes_filtered[i][2], boxes_filtered[i][3], probs_filtered[i]])

    return result

def detect_from_cvmat(inputs):
    print "inputs:",inputs
    net = yolo.yolo_net(inputs)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver(tf.global_variables())
        #saver.save(sess,OUT_FILE)
        saver.restore(sess, cfg.out_file)
        print "weights restore."
        net_output = sess.run(net)
        print "net_output:",net_output.shape
        print "net_output:",net_output.shape[0]
    results = []
    for i in range(net_output.shape[0]):
        results.append(interpret_output(net_output[i]))

    return results

def detect(img):
    #print img
    img_h, img_w, _ = img.shape
    inputs = cv2.resize(img, (cfg.image_size, cfg.image_size))
    inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
    inputs = (inputs / 255.0) 
    inputs = np.reshape(inputs, (1, cfg.image_size, cfg.image_size, 3))
    #inputs = np.transpose(inputs,(0,3,2,1))

    result = detect_from_cvmat(inputs)[0]

    for i in range(len(result)):
        result[i][1] *= (1.0 * img_w / cfg.image_size)
        result[i][2] *= (1.0 * img_h / cfg.image_size)
        result[i][3] *= (1.0 * img_w / cfg.image_size)
        result[i][4] *= (1.0 * img_h / cfg.image_size)

    print "result:", result
    return result

def draw_result(img, result):
    for i in range(len(result)):
        x = int(result[i][1])
        y = int(result[i][2])
        w = int(result[i][3] / 2)
        h = int(result[i][4] / 2)
        cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
        cv2.rectangle(img, (x - w, y - h - 20),
                      (x + w, y - h), (125, 125, 125), -1)
        cv2.putText(img, result[i][0] + ' : %.2f' % result[i][5], (x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.CV_AA)
def image_detector(imname, wait=0):
    print imname
    detect_timer = Timer()
    image = cv2.imread(imname)

    detect_timer.tic()
    result = detect(image)
    detect_timer.toc()
    print 'Average detecting time: {:.3f}s'.format(detect_timer.average_time)

    draw_result(image, result)
    cv2.imshow('Image', image)
    cv2.waitKey(wait)


image_detector(cfg.test_img)
