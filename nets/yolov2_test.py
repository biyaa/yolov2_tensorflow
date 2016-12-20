import numpy as np
import tensorflow as tf
import yolov2
yolo = yolov2
def read_txt_data(path,filename):
  f = open(path + filename ,'r')
  
  lines = f.readlines()
  f.close()
  data =[]
  for l in lines:
      row = []
      for r in xrange(416):
          d = l.split(",")
          row.append(d)
      
      data.append(row)
  return data

data = read_txt_data('./','img.txt')
inputs = np.array(data,dtype=np.float32)
inputs = np.reshape(1,416,416,3),dtype=np.float32)
net = yolo.yolo_net(inputs)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print sess.run(net)
all_var = yolo.slim.get_model_variables()
for v in all_var:                        
       print  v.name ,v.get_shape()

