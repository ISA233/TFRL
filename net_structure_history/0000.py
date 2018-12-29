import os
from chess import ChessBoard
import numpy as np
import tensorflow as tf
from tools import player_01, to_vector

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sess = tf.Session()


def weight_variable(shape):
	initial = tf.truncated_normal(shape, mean=0, stddev=0.2)
	return tf.Variable(initial)


def bias_variable(shape):
	initial = tf.constant(0.0, shape=shape)
	# initial = tf.random_uniform(shape=shape, minval=-0.3, maxval=0.3)
	return tf.Variable(initial)


def conv2d(MA, MB):
	return tf.nn.conv2d(MA, MB, strides=[1, 1, 1, 1], padding='SAME')


def arg_max(board, probability):
	board = board.reshape(9)
	for i in range(9):
		if board[i]:
			probability[i] = -1
	# print(probability)
	return np.argmax(probability)


class Network:
	def __init__(self, learning_rate=0.02):
		W_conv1 = weight_variable([2, 2, 4, 16])
		b_conv1 = bias_variable([16])
		W_conv2 = weight_variable([2, 2, 16, 16])
		b_conv2 = bias_variable([16])
		W_conv3 = weight_variable([2, 2, 16, 16])
		b_conv3 = bias_variable([16])
		W_conv4 = weight_variable([1, 1, 16, 1])
		b_conv4 = bias_variable([1])

		self.state = tf.placeholder('float', [None, 3, 3, 4])

		h_conv1 = tf.nn.relu(conv2d(self.state, W_conv1) + b_conv1)
		h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
		h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3) + b_conv3)
		h_conv4 = conv2d(h_conv3, W_conv4) + b_conv4
		feature_map = tf.reshape(h_conv4, [-1, 3 * 3])
		self.probability = tf.nn.softmax(feature_map)

		self.action = tf.placeholder('float', [None, 9])
		self.cost = tf.reduce_mean(-tf.reduce_sum(self.action * tf.log(self.probability),
		                                          reduction_indices=1))
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		self.train = self.optimizer.minimize(self.cost)

	def learn_to(self, moves, iam):
		print(moves)
		board = ChessBoard()
		boards = []
		actions = [to_vector(move) for move in moves[player_01(-iam) // 2]]
		current_player = 1
		for move in zip(moves[0], moves[1]):
			current_player = -current_player
			boards.append(board.to_network_input(current_player))
			board.move(move[player_01(current_player)], current_player)
		return sess.run(self.train,
		                feed_dict={self.state: boards, self.action: actions})

	def play(self, chessboard, player):
		probability = sess.run(self.probability,
		                       feed_dict={self.state: [chessboard.to_network_input(player)]})
		return arg_max(chessboard.board, probability[0])


def init():
	initializer = tf.global_variables_initializer()
	sess.run(initializer)


def main():
	board = ChessBoard([[0, 1, 0],
	                    [-1, 0, 0],
	                    [1, -1, 0]])
	net0 = Network()
	init()
	print(net0.play(board, -1))


if __name__ == '__main__':
	main()
