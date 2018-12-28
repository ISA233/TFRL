from network import Network, init
from chess import ChessBoard


def match(player0, player1, stdout=0):
	board = ChessBoard()
	if stdout:
		board.out()
	current_player = 1
	moves0, moves1 = [], []
	while True:
		current_player = -current_player
		if not board.drop_list(current_player):
			if stdout:
				print(current_player, -1)
			moves0.append(-1)
			moves1.append(-1)
			continue
		p0 = player0.play(board, current_player)
		p1 = player1.play(board, current_player)
		moves0.append(p0)
		moves1.append(p1)
		if current_player == -1:
			move = p0
		else:
			move = p1
		board.move(move, current_player)
		if stdout:
			print(current_player, move)
			board.out()
		if board.is_finish():
			return board.evaluate(), moves0, moves1


def main():
	net0 = Network()
	net1 = Network()
	init()
	print(match(net0, net1, 1))


if __name__ == '__main__':
	main()
