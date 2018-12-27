import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sess = tf.Session()


class Net:
	def __init__(self):
		x = tf.Variable(tf.random_uniform(shape=[1], minval=0, maxval=1))
		y = tf.Variable(tf.random_uniform(shape=[1], minval=0, maxval=1))
		self.v = x + y


if __name__ == '__main__':
	a = Net()
	b = Net()
	init = tf.global_variables_initializer()
	sess.run(init)
	print(sess.run(a.v))
	print(sess.run(a.v))
