import numpy as np
import tensorflow as tf
import yolo
OUT_FILE = 'ckpt/yolo_hgx.ckpt'
inputs = np.ones((1,416,416,3),dtype=np.float32)
net = yolo.yolo_net(inputs)
all_var = yolo.slim.get_model_variables()

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
saver = tf.train.Saver(tf.global_variables())
saver.restore(sess,OUT_FILE)
for v in all_var:                        
       if "conv0/weight" in v.name:
           print v
           print sess.run(v)

sess.close()
