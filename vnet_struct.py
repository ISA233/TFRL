import tensorflow as tf
from chess.chess import ChessBoard
from config import config


def weight_variable(shape, alpha=config.l2_regularizer_alpha):
	initial = tf.truncated_normal(shape, mean=0, stddev=0.03)
	var = tf.Variable(initial)
	tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(alpha)(var))
	return var


def bias_variable(shape):
	initial = tf.constant(0.0, shape=shape)
	return tf.Variable(initial)


def conv2d(MA, MB):
	return tf.nn.conv2d(MA, MB, strides=[1, 1, 1, 1], padding='SAME')


class NetStructure:
	def get_cnt(self):
		self.cnt += 1
		return '__' + str(self.cnt)

	def build_dense_layer(self, input_tensor, shape, activation=tf.identity):
		with tf.name_scope(self.name):
			W = weight_variable([input_tensor.shape[-1].value, shape])
			b = bias_variable([shape])
		dense = tf.matmul(input_tensor, W) + b
		return activation(dense)

	def build_conv_layer(self, input_tensor, shape, bn_training):
		with tf.name_scope(self.name):
			W_conv = weight_variable(shape)
			b_conv = bias_variable([shape[-1]])
		h_conv = conv2d(input_tensor, W_conv) + b_conv
		h_conv_bn = tf.layers.batch_normalization(h_conv, training=bn_training, name=self.name + self.get_cnt())
		return tf.nn.leaky_relu(h_conv_bn)

	def build_residual_layer(self, input_tensor, shape, bn_training):
		with tf.name_scope(self.name):
			W1_conv = weight_variable(shape)
			b1_conv = bias_variable([shape[-1]])
			W2_conv = weight_variable(shape)
			b2_conv = bias_variable([shape[-1]])
		h1_conv = conv2d(input_tensor, W1_conv) + b1_conv
		h1_conv_bn = tf.layers.batch_normalization(h1_conv, training=bn_training, name=self.name + self.get_cnt())
		h1_conv_bn_relu = tf.nn.leaky_relu(h1_conv_bn)
		h2_conv = conv2d(h1_conv_bn_relu, W2_conv) + b2_conv
		h2_conv_bn = tf.layers.batch_normalization(h2_conv, training=bn_training, name=self.name + self.get_cnt())
		ADD_layer = h2_conv_bn + input_tensor
		return tf.nn.leaky_relu(ADD_layer)

	def build_body(self):
		conv_layer_shape = [3, 3, 2, 128]
		residual_layer_shape = [3, 3, 128, 128]
		residual_layer_number = 5
		body = self.build_conv_layer(self.state, conv_layer_shape, self.bn_training)
		for i in range(residual_layer_number):
			body = self.build_residual_layer(body, residual_layer_shape, self.bn_training)
		return body

	def build_vhead(self, body):
		feature_number = 1
		flatten_conv_layer_shape = [1, 1, 128, feature_number]
		vhead = self.build_conv_layer(body, flatten_conv_layer_shape, self.bn_training)
		vhead = tf.reshape(vhead, [-1, 64 * feature_number])
		dense_shapes = [128, 1]
		for shape in dense_shapes:
			activation = tf.nn.tanh if shape == 1 else tf.nn.leaky_relu
			vhead = self.build_dense_layer(vhead, shape, activation)
		return vhead

	def build_phead(self, body):
		feature_number = 2
		flatten_conv_layer_shape = [3, 3, 128, feature_number]
		phead = self.build_conv_layer(body, flatten_conv_layer_shape, self.bn_training)
		phead = tf.reshape(phead, [-1, 64 * feature_number])
		dense_shapes = [65]
		for shape in dense_shapes:
			phead = self.build_dense_layer(phead, shape)
		return phead

	def __init__(self, learning_rate=config.learning_rate, bn_training=True):
		self.name = 'RL_zys'
		self.bn_training = bn_training
		self.cnt = -1
		self.learning_rate = tf.Variable(learning_rate, dtype='float', trainable=False, name='learning_date')
		self.state = tf.placeholder('float', [None, 8, 8, 2])

		body = self.build_body()
		self.vhead = self.build_vhead(body)
		self.phead = self.build_phead(body)
		self.dist = tf.nn.softmax(self.phead)

		self.v = tf.placeholder('float', [None, 1])
		self.p = tf.placeholder('float', [None, 65])
		self.v_loss = tf.reduce_mean(tf.square(self.vhead - self.v))
		self.p_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.phead, labels=self.p))
		self.l2_loss = tf.add_n(tf.get_collection('losses'))
		self.loss = self.v_loss + self.p_loss + self.l2_loss
		self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, config.momentum)
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
	print(sess.run([net.vhead], feed_dict={net.state: [board.to_network_input(-1)]})[0])
	print(sess.run([net.phead], feed_dict={net.state: [board.to_network_input(-1)]})[0])
	print(sess.run([net.v_loss], feed_dict={net.state: [board.to_network_input(-1)], net.v: [[1]]})[0])


if __name__ == '__main__':
	main()
