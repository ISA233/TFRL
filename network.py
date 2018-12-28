import os
from chess import ChessBoard
import numpy as np
import tensorflow as tf
from tools import player_01, to_vector

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sess = tf.Session()


def arg_max(board, probability, player):
	dlist = board.drop_list(player)
	for i in range(64):
		if i not in dlist:
			probability[i] = -1
	return np.argmax(probability)


def choice(board, probability, player):
	dlist = board.drop_list(player)
	for i in range(64):
		if i not in dlist:
			probability[i] = 0
	sum_p = np.sum(probability)
	for i in range(64):
		probability[i] /= sum_p
	return np.random.choice(range(64), p=probability)


class Network:
	def __init__(self, learning_rate=0.04):
		self.state = tf.placeholder('float', [None, 65])

		layer1 = tf.layers.dense(
			inputs=self.state,
			units=50,
			activation=tf.nn.relu,
			kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
			bias_initializer=tf.constant_initializer(0.0)
		)

		layer2 = tf.layers.dense(
			inputs=layer1,
			units=100,
			activation=tf.nn.relu,
			kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
			bias_initializer=tf.constant_initializer(0.0)
		)

		all_act = tf.layers.dense(
			inputs=layer2,
			units=64,
			activation=None,
			kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
			bias_initializer=tf.constant_initializer(0.0)
		)

		self.probability = tf.nn.softmax(all_act)

		self.action = tf.placeholder('float', [None, 64])
		self.value = tf.placeholder('float', [None])
		self.cost = tf.reduce_mean(-tf.reduce_sum(self.action * tf.log(self.probability),
		                                          reduction_indices=1) * self.value)
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
		self.train = self.optimizer.minimize(self.cost)

	def learn_to(self, moves, iam, delay=0.98):
		# print(moves)
		board = ChessBoard()
		boards = []
		actions0 = [to_vector(move) for move in moves[player_01(iam)] if move != -1]
		actions1 = [to_vector(move) for move in moves[player_01(-iam)] if move != -1]
		actions = actions0 + actions1
		values0 = [-pow(delay, i) for i in range(len(actions0))][::-1]
		values1 = [pow(delay, i) for i in range(len(actions1))][::-1]
		values = values0 + values1
		current_player = 1
		for move in zip(moves[0], moves[1]):
			current_player = -current_player
			if move[player_01(current_player)] == -1:
				continue
			boards.append(board.to_network_input(current_player))
			board.move(move[player_01(current_player)], current_player)
		boards = boards + boards
		print(len(boards), len(actions), len(values))
		return sess.run(self.train,
		                feed_dict={self.state: boards, self.action: actions, self.value: values})

	def play(self, chessboard, player):
		probability = sess.run(self.probability,
		                       feed_dict={self.state: [chessboard.to_network_input(player)]})
		return choice(chessboard, probability[0], player)

	def play_max(self, chessboard, player):
		probability = sess.run(self.probability,
		                       feed_dict={self.state: [chessboard.to_network_input(player)]})
		return arg_max(chessboard, probability[0], player)


def init():
	initializer = tf.global_variables_initializer()
	sess.run(initializer)


def main():
	board = np.array([[0, 1, 1, -1, 0, 0, 0, 0],
	                  [0, 0, 1, 1, -1, 0, 0, 0],
	                  [0, 1, 1, 0, 0, 0, 0, 0],
	                  [0, 1, 0, 1, 1, 0, 0, 0],
	                  [0, 1, 0, 1, -1, 0, 0, 0],
	                  [0, -1, 0, 0, 0, 0, 0, 0],
	                  [0, 0, 0, 0, 0, 0, 0, 0],
	                  [0, 0, 0, 0, 0, 0, 0, 0]])
	board = ChessBoard(board)
	net0 = Network()
	init()
	print(net0.play(board, -1))


if __name__ == '__main__':
	main()
