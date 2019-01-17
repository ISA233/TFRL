import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from chess.chess import ChessBoard
import numpy as np
import parameter


def weight_variable(shape):
	initial = tf.truncated_normal(shape, mean=0, stddev=0.3)
	return tf.Variable(initial)


def bias_variable(shape):
	initial = tf.constant(0.0, shape=shape)
	return tf.Variable(initial)


def conv2d(MA, MB):
	return tf.nn.conv2d(MA, MB, strides=[1, 1, 1, 1], padding='SAME')


class NetStructure:
	def __init__(self, learning_rate=parameter.learning_rate):
		self.name = 'RL_zys_v'
		self.state = tf.placeholder('float', [None, 8, 8, 3])
		with tf.name_scope(self.name):
			W_conv1 = weight_variable([3, 3, 3, 32])
			b_conv1 = bias_variable([32])
			W_conv2 = weight_variable([3, 3, 32, 128])
			b_conv2 = bias_variable([128])
			W_conv3 = weight_variable([3, 3, 128, 128])
			b_conv3 = bias_variable([128])
			W_conv4 = weight_variable([3, 3, 128, 128])
			b_conv4 = bias_variable([128])
			W_conv5 = weight_variable([3, 3, 128, 128])
			b_conv5 = bias_variable([128])
			W_conv6 = weight_variable([1, 1, 128, 1])
			b_conv6 = bias_variable([1])

		h_conv1 = tf.nn.leaky_relu(conv2d(self.state, W_conv1) + b_conv1)
		h_conv2 = tf.nn.leaky_relu(conv2d(h_conv1, W_conv2) + b_conv2)
		h_conv3 = tf.nn.leaky_relu(conv2d(h_conv2, W_conv3) + b_conv3)
		h_conv4 = tf.nn.leaky_relu(conv2d(h_conv3, W_conv4) + b_conv4)
		h_conv5 = tf.nn.leaky_relu(conv2d(h_conv4, W_conv5) + b_conv5)
		h_conv6 = tf.nn.leaky_relu(conv2d(h_conv5, W_conv6) + b_conv6)

		feature = tf.reshape(h_conv6, [-1, 64])

		layer = tf.layers.dense(
			inputs=feature,
			units=128,
			activation=tf.nn.tanh,
			kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
			bias_initializer=tf.constant_initializer(0.0),
			name=self.name + '0'
		)

		self.feature_map = tf.layers.dense(
			inputs=layer,
			units=1,
			activation=None,
			kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
			bias_initializer=tf.constant_initializer(0.0),
			name=self.name + '1'
		)

		self.probability = tf.nn.sigmoid(self.feature_map)

		self.value = tf.placeholder('float', [None, 1])
		self.cost = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(logits=self.feature_map, labels=self.value))
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		self.train = self.optimizer.minimize(self.cost)
		self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
		# print(self.vars)
		# print(tf.global_variables())
		self.initializer = tf.variables_initializer(self.vars)


net = NetStructure()


def main():
	sess = tf.Session()
	sess.run(net.initializer)
	board = ChessBoard()
	print(sess.run([net.probability], feed_dict={net.state: [board.to_network_input(-1)], net.value: [[1]]})[0])
	print(sess.run([net.cost], feed_dict={net.state: [board.to_network_input(-1)], net.value: [[1]]})[0])


if __name__ == '__main__':
	main()
