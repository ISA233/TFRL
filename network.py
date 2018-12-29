import os
from chess import ChessBoard
import numpy as np
import tensorflow as tf
import parameter
from network_structure import net
from tools import player_01, to_vector

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
	for i in range(64):
		probability[i] /= sum_p
	return np.random.choice(range(64), p=probability)


# def pretreatment(moves, iam, value, delay):  # double learn
# 	board = ChessBoard()
# 	boards = []
# 	actions0 = [to_vector(move) for move in moves[player_01(iam)] if move != -1]
# 	actions1 = [to_vector(move) for move in moves[player_01(-iam)] if move != -1]
# 	actions = actions0 + actions1
# 	values0 = [-pow(delay, i) * value for i in range(len(actions0))][::-1]
# 	values1 = [pow(delay, i) * value for i in range(len(actions1))][::-1]
# 	values = values0 + values1
# 	current_player = 1
# 	for move in zip(moves[0], moves[1]):
# 		current_player = -current_player
# 		if move[player_01(current_player)] == -1:
# 			continue
# 		boards.append(board.to_network_input(current_player))
# 		board.move(move[player_01(current_player)], current_player)
# 	boards = boards + boards
# 	return boards, actions, values

def pretreatment(moves, iam, value, delay):  # single learn
	board = ChessBoard()
	boards = []
	actions = [to_vector(move) for move in moves[player_01(-iam)] if move != -1]
	values = [pow(delay, i) * value for i in range(len(actions))][::-1]
	current_player = 1
	for move in zip(moves[0], moves[1]):
		current_player = -current_player
		if move[player_01(current_player)] == -1:
			continue
		boards.append(board.to_network_input(current_player))
		board.move(move[player_01(current_player)], current_player)
	return boards, actions, values


class Network:
	def __init__(self, name='net'):
		self.name = name
		self.sess = tf.Session()
		self.sess.run(net.initializer)
		# print(net.vars)
		self.saver = tf.train.Saver(net.vars, max_to_keep=1)

	def learn_to(self, moves, iam, value=1, delay=parameter.delay):
		# print(moves)；
		boards, actions, values = pretreatment(moves, iam, value, delay)
		# print(len(boards), len(actions), len(values))
		return self.sess.run(net.train,
		                     feed_dict={net.state: boards, net.action: actions, net.value: values})

	def get_probability(self, chessboard, player):
		return self.sess.run(net.probability, feed_dict={net.state: [chessboard.to_network_input(player)]})

	def get_probability_out(self, chessboard, player):
		probability = self.get_probability(chessboard, player)[0]
		probability = probability.reshape([8, 8])
		for i in range(8):
			for j in range(8):
				print('%.2f' % (probability[i, j] * 10), end='  ')
			print()

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
	print(net0.play(board, -1))


if __name__ == '__main__':
	main()
