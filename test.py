from network import Network
import random
from chess import ChessBoard


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
				place = player.play_max(board, current_player)
				print('AI:', place)
				board.move(place, current_player)
			board.out()
			print('---------------------------')
			if board.is_finish():
				print('Game End.')
				print('---------------------------')
				break


def main():
	net = Network()
	net.restore('model_save/net0')
	test(net)


if __name__ == '__main__':
	main()
