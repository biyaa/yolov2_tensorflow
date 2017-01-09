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
    avg_anyobj = _to_zero(avg_anyobj)
    avg_obj = _to_zero(avg_obj)
    avg_cat = _to_zero(avg_cat)
    avg_iou = _to_zero(avg_iou)
    class_count = _to_zero(class_count)
    count = _to_zero(count)
    recall = _to_zero(recall)
    cost = _to_zero(cost)

def _to_zero(v):
    v = tf.clip_by_value(v,1e-10,1e10)
    v = v * 0
