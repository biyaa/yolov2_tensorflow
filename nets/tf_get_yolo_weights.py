import tensorflow as tf 
import numpy as np
WEIGHT_DIR ='weights/'
OUT_FILE = 'ckpt/yolo_hgx.ckpt'
ALPHA = 0.1
BN_EPSILON = .000001
def read_txt_data(path,filename):
  f = open(path + filename ,'r')
  
  lines = f.readlines()
  f.close()
  data ={}
  for l in lines:
      d = l.split(",")
      data[d[0]] =d[1:-1]
  
  return data

def apply_bn(x,mean,variance,scale):
    x = tf.nn.batch_normalization(x, mean, variance, None, scale, BN_EPSILON)
    return x

def conv_layer(idx,inputs,filters,size,stride):
    channels = inputs.get_shape()[3]
    #f_w = open(self.weights_dir + str(idx) + '_conv.txt','r')
    data = read_txt_data(WEIGHT_DIR , str(idx) + '_conv.txt')
    l_w = np.array(data['weights']).astype('float32')	
    w = np.zeros((size,size,channels,filters),dtype='float32')
    ci = int(channels)
    filter_step = ci*size*size
    channel_step = size*size
    for i in range(filters):
    	for j in range(ci):
    		for k in range(size):
    			for l in range(size):
    				w[k,l,j,i] = l_w[i*filter_step + j*channel_step + k*size + l]
    
    weight = tf.Variable(w)
    l_b = np.array(data['biases']).astype('float32')	
    biases = tf.Variable(l_b.reshape((filters)))
    
    pad_size = size//2
    pad_mat = np.array([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
    inputs_pad = tf.pad(inputs,pad_mat)
    
    conv = tf.nn.conv2d(inputs_pad, weight, strides=[1, stride, stride, 1], padding='VALID',name=str(idx)+'_conv')	

    l_m = np.array(data['rolling_mean']).astype('float32')	
    rolling_mean = tf.Variable(l_m.reshape((filters)))

    l_v = np.array(data['rolling_variance']).astype('float32')	
    rolling_variance = tf.Variable(l_v.reshape((filters)))

    l_s = np.array(data['scales']).astype('float32')	
    scales = tf.Variable(l_s.reshape((filters)))

    conv_bn = apply_bn(conv,rolling_mean,rolling_variance,scales)
    conv_biased = tf.add(conv_bn,biases,name=str(idx)+'_conv_biased')	
    print 'Loaded ' + str(idx) + ' : conv     from ' + WEIGHT_DIR + str(idx) + '_conv.txt'
    return tf.maximum(ALPHA*conv_biased,conv_biased,name=str(idx)+'_leaky_relu')
  
def pooling_layer(idx,inputs,size,stride):
    print 'Create ' + str(idx) + ' : pool'
    return tf.nn.max_pool(inputs, ksize=[1, size, size, 1],strides=[1, stride, stride, 1], padding='SAME',name=str(idx)+'_pool')

def build_networks():
    x = tf.placeholder('float32',[None,416,416,3])
    conv_0 = conv_layer(0,x,32,3,1)
    pool_1 = pooling_layer(1,conv_0,2,2)
    conv_2 = conv_layer(2,pool_1,64,3,1)
    pool_3 = pooling_layer(3,conv_2,2,2)
    conv_4 = conv_layer(4,pool_3,128,3,1)
    conv_5 = conv_layer(5,conv_4,64,1,1)
    conv_6 = conv_layer(6,conv_5,128,3,1)
    pool_7 = pooling_layer(7,conv_6,2,2)
    conv_8 = conv_layer(8,pool_7,256,3,1)
    conv_9 = conv_layer(9,conv_8,128,1,1)
    conv_10 = conv_layer(10,conv_9,256,3,1)
    pool_11 = pooling_layer(11,conv_10,2,2)
    conv_12 = conv_layer(12,pool_11,512,3,1)
    conv_13 = conv_layer(13,conv_12,256,1,1)
    conv_14 = conv_layer(14,conv_13,512,3,1)
    conv_15 = conv_layer(15,conv_14,256,1,1)
    conv_16 = conv_layer(16,conv_15,512,3,1)
    pool_17 = pooling_layer(17,conv_16,2,2)
    conv_18 = conv_layer(18,pool_17,1024,3,1)
    conv_19 = conv_layer(19,conv_18,512,1,1)
    conv_20 = conv_layer(20,conv_19,1024,3,1)
    conv_21 = conv_layer(21,conv_20,512,1,1)
    conv_22 = conv_layer(22,conv_21,1024,3,1)
    conv_23 = conv_layer(23,conv_22,1024,3,1)
    conv_24 = conv_layer(24,conv_23,1024,3,1)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print conv_24
    saver = tf.train.Saver(tf.global_variables())
    saver.save(sess,OUT_FILE)

def main():
	yolo = build_networks()


if __name__=='__main__':
	main()
