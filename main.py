from network import Network, init
import match
import random
from chess import ChessBoard

network_number = 5
net_pool = [Network() for i in range(network_number)]


def learning(max_epoch=150000):
	print('Learning.')
	init()
	for epoch in range(max_epoch):
		print('--------------------------')
		print('train: ', epoch)
		_player0 = random.randint(1, network_number) - 1
		_player1 = random.randint(1, network_number - 1) - 1
		if _player1 >= _player0:
			_player1 += 1
		print('player:', _player0, _player1)
		player0 = net_pool[_player0]
		player1 = net_pool[_player1]
		result = match.match(player0, player1)
		if result[0] == 1:
			player0.learn_to(result[1:], iam=-1)
		elif result[0] == -1:
			player1.learn_to(result[1:], iam=1)
		print('result:', result[0])


def test():
	print('------------ TESTING ------------')
	while True:
		print('choose AI:', end=' ')
		_player = int(input())
		print('choose x or o:', end=' ')
		player = net_pool[_player]
		iam = int(input())
		board = ChessBoard()
		current_player = 1
		while True:
			current_player = -current_player
			if current_player == iam:
				print('You:', end=' ')
				place = int(input())
			else:
				place = player.play_max(board, current_player)
				print('AI:', place)
			board.move(place, current_player)
			board.out()
			print('---------------------------')
			if board.is_finish() or not board.have_space():
				print('Game End.')
				print('---------------------------')
				break


def main():
	learning()
	test()


if __name__ == '__main__':
	main()
