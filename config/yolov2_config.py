import os
import numpy as np
#
# path and dataset parameter
#

yolo_home = ''
train_data_path='/mnt/disk1/tim/train.txt'
train_log_path=os.path.join(yolo_home,'ckpt/logs')
train_ckpt_path=os.path.join(yolo_home,'ckpt/')
#mnt/disk4/nn/tf-env/tim/yolov2_tensorflow/ckpt/yolo.ckpt-49501
object_scale=5
noobject_scale=1
class_scale=1
coord_scale=1
image_size = 416
batch_size = 8
momentum=0.9
learning_rate = 0.0000001
max_steps = 150000

test_path = 'test'
test_img = os.path.join(test_path, 'dog.jpg')
cell_size = 13
num_class = 80
boxes_per_cell = 5
coords = 4
scale = 1
threshold = 0.6
anchors = np.array([0.738768,0.874946,  2.42204,2.65704,  4.30971,7.04493,  10.246,4.59428,  12.6868,11.8741],dtype=np.float32)
weight_dir ='nets/weights/'
out_file = 'nets/ckpt/yolo_hgx.ckpt'
iou_threshold = 0.400000006
cls=['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
