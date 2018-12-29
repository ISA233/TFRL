import tensorflow as tf


def weight_variable(shape):
	initial = tf.truncated_normal(shape, mean=0, stddev=0.2)
	return tf.Variable(initial)


W_conv1 = weight_variable([2, 2, 3, 4])
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
print(sess.run(W_conv1))
