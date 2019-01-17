from network import Network
import numpy as np
from chess.chess import ChessBoard


def test(player):
	print('------------ TESTING ------------')
	while True:
		print('choose x or o:', end=' ')
		iam = int(input())
		board = ChessBoard()
		current_player = 1
		while True:
			current_player = -current_player
			if current_player == iam:
				print('You:', end=' ')
				cin = input().split(' ')
				x, y = int(cin[0]), int(cin[1])
				board.move_xy(x, y, current_player)
			else:
				player.get_probability_out(board, current_player)
				place = player.play_max(board, current_player)
				print('AI:', place)
				board.move(place, current_player)
			board.out()
			print('---------------------------')
			if board.is_finish():
				print('Game End.')
				print('---------------------------')
				break


def test2(net):
	_, x, o = 0, -1, 1
	board = np.array([[_, o, o, o, o, o, o, _],
	                  [x, o, o, x, o, o, _, x],
	                  [x, x, o, o, x, o, o, o],
	                  [x, x, o, o, x, o, o, o],
	                  [x, x, o, o, x, o, o, o],
	                  [x, o, o, o, o, o, o, o],
	                  [x, _, o, o, x, o, _, o],
	                  [x, x, o, o, o, o, _, _]])
	board = ChessBoard(board)
	# board = ChessBoard()
	net.get_probability_out(board, -1)


def main():
	net = Network('cnn_fc_net0')
	net.restore()
	test2(net)


if __name__ == '__main__':
	main()
