import tensorflow as tf
import numpy as np
import nets.yolov2 as yolo
import os
import cv2
import config.config as cfg
from utils.timer import Timer
slim = tf.contrib.slim

def get_region_box(boxes,i,j,n,anchors):
    box=np.zeros((4),dtype=np.float32)
    box[0] = (i + boxes[i,j,n,0])/cfg.cell_size
    box[1] = (j + boxes[i,j,n,1])/cfg.cell_size
    #box[2] = (0.5 * logistic_activate(boxes[i,j,n,2]))/cfg.cell_size
    #box[3] = (0.5 * logistic_activate(boxes[i,j,n,3]))/cfg.cell_size
    box[2] = (anchors[2*n] * boxes[i,j,n,2])/cfg.cell_size
    box[3] = (anchors[2*n+1] * boxes[i,j,n,3])/cfg.cell_size
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

    #offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell), (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

    boxes = output[:,:,:,:cfg.coords]

    for row in xrange(cfg.cell_size):
        for col in xrange(cfg.cell_size):
            for n in xrange(cfg.boxes_per_cell):
                boxes[row,col,n] = get_region_box(boxes,col,row,n,cfg.anchors)

    #boxes *= cfg.image_size
    print "box",boxes.shape

    probs = class_probs * scales[:,:,:,np.newaxis]

    print "probs",probs.shape
    probs[probs < cfg.threshold] =0
    probs_max = np.max(probs,3)
    probs_max_index = np.argmax(probs,3)

    #filter_mat_probs = np.array(probs >= cfg.threshold, dtype='bool')
    probs_mask = np.where(probs_max >= cfg.threshold)
    #filter_mat_boxes = np.nonzero(filter_mat_probs)
    probs_masked = probs_max[probs_mask]
    probs_index_masked = probs_max_index[probs_mask]
    boxes_masked = boxes[probs_mask]
    print probs_mask
    

    result = []
    result_num = len(probs_mask[0]) if len(probs_mask)>0 else 0
    print "num",result_num

    for i in range(result_num):
        result.append([cfg.cls[probs_index_masked[i]], boxes_masked[i][0], boxes_masked[
                      i][1], boxes_masked[i][2], boxes_masked[i][3], probs_masked[i]*100])

    return result

def detect_from_cvmat(inputs):
    #print "inputs:",inputs
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
    image = cv2.imread(imname)

    detect_timer.tic()
    result = detect(image)
    detect_timer.toc()
    print 'Average detecting time: {:.3f}s'.format(detect_timer.average_time)

    draw_result(image, result)
    cv2.imwrite('prediction.jpg', image)
    cv2.imshow('Image', image)
    cv2.waitKey(wait)


image_detector(cfg.test_img)
