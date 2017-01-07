import tensorflow as tf 
avg_anyobj = tf.Variable(tf.constant(0,dtype=tf.float32))
avg_obj = tf.Variable(tf.constant(0,dtype=tf.float32))
avg_cat = tf.Variable(tf.constant(0,dtype=tf.float32))
avg_iou = tf.Variable(tf.constant(0,dtype=tf.float32))
class_count = tf.Variable(tf.constant(0,dtype=tf.float32))
count = tf.Variable(tf.constant(0,dtype=tf.float32))
recall = tf.Variable(tf.constant(0,dtype=tf.float32))
cost = tf.Variable(tf.constant(0,dtype=tf.float32))
def set_zero():
    global avg_anyobj
    global avg_obj
    global avg_iou
    global avg_cat
    global class_count
    global count
    global recall
    global cost
    avg_anyobj = avg_anyobj * 0
    avg_obj = avg_obj * 0
    avg_cat = avg_cat * 0
    avg_iou = avg_iou * 0
    class_count = class_count * 0
    count = count * 0
    recall = recall * 0
    cost = cost * 0
    


