import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from chess.chess import ChessBoard
import numpy as np
import parameter


def weight_variable(shape, alpha=parameter.l2_regularizer_alpha):
	initial = tf.truncated_normal(shape, mean=0, stddev=0.3)
	var = tf.Variable(initial)
	tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(alpha)(var))
	return var


def bias_variable(shape):
	initial = tf.constant(0.0, shape=shape)
	return tf.Variable(initial)


def conv2d(MA, MB):
	return tf.nn.conv2d(MA, MB, strides=[1, 1, 1, 1], padding='SAME')


class NetStructure:
	def __init__(self, learning_rate=parameter.learning_rate, bn_training=True):
		self.name = 'RL_zys_v'
		self.bn_training = bn_training
		self.state = tf.placeholder('float', [None, 8, 8, 3])
		with tf.name_scope(self.name):
			W_conv1 = weight_variable([3, 3, 3, 64])
			b_conv1 = bias_variable([64])
			W_conv2 = weight_variable([3, 3, 64, 128])
			b_conv2 = bias_variable([128])
			W_conv3 = weight_variable([3, 3, 128, 128])
			b_conv3 = bias_variable([128])
			W_conv4 = weight_variable([3, 3, 128, 128])
			b_conv4 = bias_variable([128])
			W_conv5 = weight_variable([3, 3, 128, 128])
			b_conv5 = bias_variable([128])
			W_conv6 = weight_variable([3, 3, 128, 128])
			b_conv6 = bias_variable([128])
			W_conv7 = weight_variable([1, 1, 128, 5])
			b_conv7 = bias_variable([5])

		h_conv1 = tf.nn.leaky_relu(conv2d(self.state, W_conv1) + b_conv1)
		h_conv1_bn = tf.layers.batch_normalization(h_conv1, training=bn_training, name=self.name + '_bn0')
		h_conv2 = tf.nn.leaky_relu(conv2d(h_conv1_bn, W_conv2) + b_conv2)
		h_conv2_bn = tf.layers.batch_normalization(h_conv2, training=bn_training, name=self.name + '_bn1')
		h_conv3 = tf.nn.leaky_relu(conv2d(h_conv2_bn, W_conv3) + b_conv3)
		h_conv3_bn = tf.layers.batch_normalization(h_conv3, training=bn_training, name=self.name + '_bn2')
		h_conv4 = tf.nn.leaky_relu(conv2d(h_conv3_bn, W_conv4) + b_conv4)
		h_conv4_bn = tf.layers.batch_normalization(h_conv4, training=bn_training, name=self.name + '_bn3')
		h_conv5 = tf.nn.leaky_relu(conv2d(h_conv4_bn, W_conv5) + b_conv5)
		h_conv5_bn = tf.layers.batch_normalization(h_conv5, training=bn_training, name=self.name + '_bn4')
		h_conv6 = tf.nn.leaky_relu(conv2d(h_conv5_bn, W_conv6) + b_conv6)
		h_conv6_bn = tf.layers.batch_normalization(h_conv6, training=bn_training, name=self.name + '_bn5')
		h_conv7 = tf.nn.leaky_relu(conv2d(h_conv6_bn, W_conv7) + b_conv7)
		h_conv7_bn = tf.layers.batch_normalization(h_conv7, training=bn_training, name=self.name + '_bn6')

		feature = tf.reshape(h_conv7_bn, [-1, 5 * 64])
		feature_bn = tf.layers.batch_normalization(feature, training=bn_training, name=self.name + '_bn7')

		layer1 = tf.layers.dense(
			inputs=feature_bn,
			units=330,
			activation=tf.nn.leaky_relu,
			kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
			bias_initializer=tf.constant_initializer(0.0),
			name=self.name + '0'
		)
		layer1_bn = tf.layers.batch_normalization(layer1, training=bn_training, name=self.name + '_bn8')

		layer2 = tf.layers.dense(
			inputs=layer1_bn,
			units=128,
			activation=tf.nn.leaky_relu,
			kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
			bias_initializer=tf.constant_initializer(0.0),
			name=self.name + '1'
		)
		layer2_bn = tf.layers.batch_normalization(layer2, training=bn_training, name=self.name + '_bn9')

		self.feature_map = tf.layers.dense(
			inputs=layer2_bn,
			units=1,
			activation=None,
			kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
			bias_initializer=tf.constant_initializer(0.0),
			name=self.name + '2'
		)

		self.probability = tf.nn.sigmoid(self.feature_map)

		self.value = tf.placeholder('float', [None, 1])
		self.cost = tf.reduce_sum(tf.square(self.probability - self.value))
		# self.cost = tf.reduce_mean(
		# 	tf.nn.sigmoid_cross_entropy_with_logits(logits=self.feature_map, labels=self.value))
		tf.add_to_collection('losses', self.cost)
		self.loss = tf.add_n(tf.get_collection('losses'))
		# self.optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			self.train = self.optimizer.minimize(self.loss)
		self.vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
		# print(self.vars)
		# print(tf.global_variables())
		self.initializer = tf.variables_initializer(self.vars)


train_graph = tf.Graph()
with train_graph.as_default():
	net_train = NetStructure(bn_training=True)

test_graph = tf.Graph()
with test_graph.as_default():
	net_test = NetStructure(bn_training=False)

def main():
	net = net_test
	sess = tf.Session(graph=test_graph)
	sess.run(net.initializer)
	board = ChessBoard()
	print(sess.run([net.probability], feed_dict={net.state: [board.to_network_input(-1)], net.value: [[1]]})[0])
	print(sess.run([net.cost], feed_dict={net.state: [board.to_network_input(-1)], net.value: [[1]]})[0])


if __name__ == '__main__':
	main()
