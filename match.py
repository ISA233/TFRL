from chess.chess import ChessBoard
from vnet import Network
from player import Player
from tools import version_str
import numpy as np
from random import random


def match(player0, player1):
	board = ChessBoard()
	current_player = 1
	while not board.is_finish():
		current_player = -current_player
		if not board.could_drop_by(current_player):
			continue
		player = player0 if current_player == -1 else player1
		mct = player.mcts(board, current_player, 128)
		pi = mct.pi(temperature=0)
		action = np.random.choice(range(65), p=pi)
		board.move(action, current_player)
	return board


def contest(player0, player1, match_number=100):
	win_cnt = 0
	for cnt in range(match_number):
		_player0, _player1 = player0, player1
		if random() < 0.5:
			_player0, _player1 = player1, player0
		board = match(_player0, _player1)
		v = board.evaluate()
		winner = None if not v else _player0 if v < 0 else _player1
		if winner == player0:
			win_cnt += 1
		print('contest %d %.3f' % (cnt + 1, win_cnt / (cnt + 1)))
	return win_cnt / match_number


def main():
	net0 = Network('cnn_vnet')
	net0.restore(version=version_str(193))
	net1 = Network('cnn_vnet')
	net1.restore(version=version_str(193))
	player0 = Player(net0)
	player1 = Player(net1)
	match(player0, player1)


if __name__ == '__main__':
	main()
