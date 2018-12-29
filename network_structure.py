import tensorflow as tf
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
		self.name = 'RL_zys'
		self.state = tf.placeholder('float', [None, 8, 8, 3])
		with tf.name_scope(self.name):
			W_conv1 = weight_variable([5, 5, 3, 16])
			b_conv1 = bias_variable([16])
			W_conv2 = weight_variable([3, 3, 16, 32])
			b_conv2 = bias_variable([32])
			W_conv3 = weight_variable([3, 3, 32, 32])
			b_conv3 = bias_variable([32])
			W_conv4 = weight_variable([3, 3, 32, 32])
			b_conv4 = bias_variable([32])
			W_conv5 = weight_variable([1, 1, 32, 1])
			b_conv5 = bias_variable([1])

		h_conv1 = tf.nn.relu(conv2d(self.state, W_conv1) + b_conv1)
		h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
		h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)
		h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
		h_conv5 = conv2d(h_conv4, W_conv5) + b_conv5
		self.feature_map = tf.reshape(h_conv5, [-1, 64])

		self.probability = tf.nn.softmax(self.feature_map)

		self.action = tf.placeholder('float', [None, 64])
		self.value = tf.placeholder('float', [None])
		self.cost = tf.reduce_mean(
			tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.feature_map, labels=self.action) * self.value)
		# self.cost = tf.reduce_mean(-tf.reduce_sum(self.action * tf.log(self.probability),
		#                                           reduction_indices=1) * self.value)
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
	print(sess.run(net.probability, feed_dict={net.state: [np.zeros([8, 8, 3])]}))


if __name__ == '__main__':
	main()
