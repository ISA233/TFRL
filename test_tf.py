# import tensorflow as tf
from chess import ChessBoard
#
# sess = tf.Session()
#
#
# def weight_variable(shape):
# 	initial = tf.truncated_normal(shape, mean=0, stddev=1)
# 	return tf.Variable(initial)
#
#
# def bias_variable(shape):
# 	initial = tf.random_uniform(shape=shape, minval=-0.3, maxval=0.3)
# 	return tf.Variable(initial)
#
#
# def conv2d(MA, MB):
# 	return tf.nn.conv2d(MA, MB, strides=[1, 1, 1, 1], padding='SAME')
#
#
# board = ChessBoard([[0, 1, 0],
# 					[-1, 0, 0],
# 					[1, -1, 0]])
#
# W_conv1 = weight_variable([1, 1, 3, 3])
# b_conv1 = bias_variable([3])
# state = tf.placeholder('float', [None, 3, 3, 3])
# probability = tf.sigmoid(conv2d(state, W_conv1) + b_conv1)
# print(probability)
#
# init = tf.global_variables_initializer()
# sess.run(init)
# print(sess.run([probability], feed_dict={state: [board.to_network_input()]}))

import numpy as np
import random


# class fuck:
# 	def __init__(self, a):
# 		self.a = a
#
#
# pool = [fuck(i) for i in range(4)]
# p = random.choice(pool)
# p.a = 23333
# for p in pool:
# 	print(p.a)

pro = [0.2, 0, 0.8, 0]
print(np.random.choice(range(4), p=pro))