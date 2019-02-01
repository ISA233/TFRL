import tensorflow as tf
import os
# from chess.chess import ChessBoard
# from vnet import Network
# from tools import version_str, load_data, unzip
# import profile


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#
# sess1 = tf.Session()
# sess2 = tf.Session()


# def weight_variable(shape):
# 	initial = tf.truncated_normal(shape, mean=0, stddev=1)
# 	return tf.Variable(initial)
#


# def pred():
# 	net = Network('cnn_vnet', bn_training=False, use_GPU=True)
# 	net.restore(path='../vnet_save', version=version_str(193))
# 	board = ChessBoard()
# 	# for i in range(10000):
# 	# 	ret = net.sess.run([net.net.vhead, net.net.phead], feed_dict={net.net.state: [board.to_network_input(-1)]})
# 	# 	print(ret)
# 	for i in range(50):
# 		ret = net.sess.run([net.net.vhead, net.net.phead], feed_dict={net.net.state: [board.to_network_input(-1)] * 200})
# 		print(ret)

sess = tf.Session()

a = tf.Variable(tf.constant(5, shape=[10]))
b = tf.Variable(tf.constant(3, shape=[10]))
c = tf.pow(a, b)
sess.run(tf.global_variables_initializer())

print(sess.run(c))

# def main():
# 	profile.run('pred()')


# if __name__ == '__main__':
# 	main()

# def bias_variable(shape, name=''):
# 	initial = tf.random_uniform(shape=shape, minval=-0.3, maxval=0.3)
# 	return tf.Variable(initial, name)
#
#
# class Network:
# 	def __init__(self, name):
# 		with tf.name_scope(name):
# 			self.p = bias_variable([5], name)
# 			self.state = tf.placeholder('float', [None, 65])
# 			layer = tf.layers.dense(
# 				inputs=self.state,
# 				units=50,
# 				activation=tf.nn.relu,
# 				kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
# 				bias_initializer=tf.constant_initializer(0.0),
# 				name=name
# 			)
# 			self.init = tf.variables_initializer([self.p])
# 		print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name))
# 		print(tf.global_variables())
# 		print('-------------------')
#
# 	def out(self):
# 		print(self.sess.run(self.p))
#
#
# a = Network('aaa')
# init = tf.global_variables_initializer()
# sess1.run(init)
# sess2.run(init)
# print(sess1.run([a.p]))
# print(sess2.run([a.p]))
# print(sess1.run([probability], feed_dict={state: [board.to_network_input()]}))
# print(sess2.run([probability], feed_dict={state: [board.to_network_input()]}))

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

# pro = [0.2, 0, 0.8, 0]
# print(np.random.choice(range(4), p=pro))

# cin = input().split(' ')
# x = int(cin[0])
# y = int(cin[1])
# print(x, y)
