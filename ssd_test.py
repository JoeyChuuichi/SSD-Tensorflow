import tensorflow as tf
from nets.ssd_vgg_300 import ssd_arg_scope
from nets.ssd_vgg_300 import ssd_net
slim = tf.contrib.slim

with tf.Graph().as_default() as g:
    input_node = tf.placeholder(tf.float32, shape=(None, 300, 300, 3), name='input_node')
    with slim.arg_scope(ssd_arg_scope()):
        predictions, localisations, logits, end_points = ssd_net(input_node)

tb_file_path = './tb_visual/ssd_300'
writer = tf.summary.FileWriter(tb_file_path, g)
writer.close()


