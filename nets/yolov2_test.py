import numpy as np
import tensorflow as tf
import yolov2
slim = tf.contrib.slim
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

#inputs = np.array(data,dtype=np.float32)
#inputs = np.ones((1,416,416,3),dtype=np.float32)
#net = yolo.yolo_net(inputs,1)
#print net
#init = tf.global_variables_initializer()
#with tf.Session() as sess:
#    sess.run(init)
#    sess.run(net)
#all_var = yolo.slim.get_model_variables()
#for v in all_var:                        
#       print  v.name ,v.get_shape()

inp =np.array([111.,112.,113.,114.,115.,116., 
        121.,122.,123.,124.,125.,126., 
        131.,132.,133.,134.,135.,136., 
        141.,142.,143.,144.,145.,146., 
                                       
        211.,212.,213.,214.,215.,216., 
        221.,222.,223.,224.,225.,226., 
        231.,232.,233.,234.,235.,236., 
        241.,242.,243.,244.,245.,246., 
                                       
        311.,312.,313.,314.,315.,316., 
        321.,322.,323.,324.,325.,326., 
        331.,332.,333.,334.,335.,336., 
        341.,342.,343.,344.,345.,346., 
                                       
        411.,412.,413.,414.,415.,416., 
        421.,422.,423.,424.,425.,426., 
        431.,432.,433.,434.,435.,436., 
        441.,442.,443.,444.,445.,446   
        ],dtype = np.float32)
inp = np.reshape(inp,(1,4,4,6))
print inp
#inp = np.transpose(inp,(0,2,3,1))
t= yolo.reorg(inp,(1,16,2,3))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    #s =  sess.run(slim.array_ops.space_to_depth(inp,2))
    s =  sess.run(t)
    #sess.run(v)
    #s = np.transpose(s,(0,3,1,2))
    print s
