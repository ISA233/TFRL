import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from chess.chess import ChessBoard
import numpy as np
import tensorflow as tf
import parameter
from network_structure import net
from tools import to_vector

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
	# print(sum_p)
	if sum_p < 1e-8:
		return np.random.choice(dlist)
	for i in range(64):
		probability[i] /= sum_p
	return np.random.choice(range(64), p=probability)


def pretreatment(moves, iam, value, delay):  # single learn
	board = ChessBoard()
	boards, actions, values = [], [], []
	current_player = 1
	for move in moves:
		current_player = -current_player
		if move == -1:
			continue
		if current_player == iam:
			boards.append(board.to_network_input(current_player))
			actions.append(to_vector(move))
			if not values:
				values.append(value)
			else:
				values.append(values[-1] * delay)
		board.move(move, current_player)
	values = values[::-1]
	return boards, actions, values


class Network:
	def __init__(self, name='net'):
		self.name = name
		self.sess = tf.Session()
		self.sess.run(net.initializer)
		self.saver = tf.train.Saver(net.vars, max_to_keep=1)

	# def __del__(self):
	# 	print('del', self.name)

	def train(self, boards, actions, values):
		return self.sess.run(net.train, feed_dict={net.state: boards, net.action: actions, net.value: values})

	def learn_to(self, moves, iam, value=1, delay=parameter.delay):
		boards, actions, values = pretreatment(moves, iam, value, delay)
		# print(len(boards), len(actions), len(values))
		return self.train(boards, actions, values)

	def get_probability(self, chessboard, player):
		return self.sess.run(net.probability, feed_dict={net.state: [chessboard.to_network_input(player)]})

	def get_probability_out(self, chessboard, player):
		probability = self.get_probability(chessboard, player)[0]
		probability = probability.reshape([8, 8])
		for i in range(8):
			for j in range(8):
				print('%.2f' % (probability[i, j] * 10), end='  ')
			print()
		print('--------------------------------')

	def play(self, chessboard, player):
		probability = self.get_probability(chessboard, player)
		return choice(chessboard, probability[0], player)

	def play_max(self, chessboard, player):
		probability = self.get_probability(chessboard, player)
		return arg_max(chessboard, probability[0], player)

	def save(self):
		self.saver.save(self.sess, 'model_save/' + self.name + '/' + self.name)

	def restore(self, path='None'):
		if path == 'None':
			path = 'model_save/' + self.name + '/' + self.name
		self.saver.restore(self.sess, path)


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
	net0.get_probability_out(board, -1)
	# print(net0.play(board, -1))


if __name__ == '__main__':
	main()
