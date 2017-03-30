import functools
import os

import numpy as np
import tensorflow as tf


def weight_variable(shape):
    n_input = functools.reduce(lambda x,y:y*x, shape[:-1])
    initial = tf.truncated_normal(shape, stddev=np.sqrt(2/n_input))
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')


def uconv2d(x, W, output_shape, stride):
    return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1, stride, stride, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)