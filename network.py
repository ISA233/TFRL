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
	return np.argmax(probability)


def choice(board, probability):
	board = board.reshape(9)
	for i in range(9):
		if board[i]:
			probability[i] = 0
	sum_p = np.sum(probability)
	for i in range(9):
		probability[i] /= sum_p
	return np.random.choice(range(9), p=probability)


class Network:
	def __init__(self, learning_rate=0.04):
		self.state = tf.placeholder('float', [None, 10])

		layer = tf.layers.dense(
			inputs=self.state,
			units=50,
			activation=tf.nn.tanh,
			kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
			bias_initializer=tf.constant_initializer(0.0)
		)

		all_act = tf.layers.dense(
			inputs=layer,
			units=9,
			activation=None,
			kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
			bias_initializer=tf.constant_initializer(0.0)
		)

		self.probability = tf.nn.softmax(all_act)

		self.action = tf.placeholder('float', [None, 9])
		self.value = tf.placeholder('float', [None])
		self.cost = tf.reduce_mean(-tf.reduce_sum(self.action * tf.log(self.probability),
		                                          reduction_indices=1) * self.value)
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		self.train = self.optimizer.minimize(self.cost)

	def learn_to(self, moves, iam, delay=0.88):
		print(moves)
		board = ChessBoard()
		boards = []
		actions = [to_vector(move) for move in moves[player_01(iam)]] + [to_vector(move) for move in
		                                                                 moves[player_01(-iam)]]
		values = [-pow(delay, i) for i in range(len(moves[0]))][::-1] + [pow(delay, i) for i in range(len(moves[0]))][
		                                                                ::-1]
		# print(actions)
		# print('RT:', player_01(-iam))
		current_player = 1
		for move in zip(moves[0], moves[1]):
			current_player = -current_player
			boards.append(board.to_network_input(current_player))
			board.move(move[player_01(current_player)], current_player)
		boards = boards + boards
		# print(actions)
		# print(values)
		# print(boards)
		return sess.run(self.train,
		                feed_dict={self.state: boards, self.action: actions, self.value: values})

	def play(self, chessboard, player):
		probability = sess.run(self.probability,
		                       feed_dict={self.state: [chessboard.to_network_input(player)]})
		return choice(chessboard.board, probability[0])

	def play_max(self, chessboard, player):
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
