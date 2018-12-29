import tensorflow as tf
import numpy as np
import parameter


class NetStructure:
	def __init__(self, learning_rate=parameter.learning_rate):
		self.name = 'RL_zys'
		self.state = tf.placeholder('float', [None, 65])

		layer1 = tf.layers.dense(
			inputs=self.state,
			units=50,
			activation=tf.nn.relu,
			kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
			bias_initializer=tf.constant_initializer(0.0),
			name=self.name + '0'
		)

		layer2 = tf.layers.dense(
			inputs=layer1,
			units=100,
			activation=tf.nn.relu,
			kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
			bias_initializer=tf.constant_initializer(0.0),
			name=self.name + '1'
		)

		all_act = tf.layers.dense(
			inputs=layer2,
			units=64,
			activation=None,
			kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
			bias_initializer=tf.constant_initializer(0.0),
			name=self.name + '2'
		)

		self.probability = tf.nn.softmax(all_act)

		self.action = tf.placeholder('float', [None, 64])
		self.value = tf.placeholder('float', [None])
		self.cost = tf.reduce_mean(-tf.reduce_sum(self.action * tf.log(self.probability),
		                                          reduction_indices=1) * self.value)
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		self.train = self.optimizer.minimize(self.cost)
		self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
		# print(self.vars)
		self.initializer = tf.variables_initializer(self.vars)


net = NetStructure()


def main():
	sess = tf.Session()
	sess.run(net.initializer)
	print(sess.run(net.probability, feed_dict={net.state: [np.zeros(65)]}))


if __name__ == '__main__':
	main()
