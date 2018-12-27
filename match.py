from network import Network, init
from chess import ChessBoard


def match(player0, player1, stdout=0):
	board = ChessBoard()
	if stdout:
		print(board.board)
	current_player = 1
	moves0, moves1 = [], []
	while board.have_space():
		current_player = -current_player
		p0 = player0.play(board, current_player)
		p1 = player1.play(board, current_player)
		moves0.append(p0)
		moves1.append(p1)
		if current_player == -1:
			move = p0
		else:
			move = p1
		result = board.move(move, current_player)
		if stdout:
			print(move, current_player)
			print(board.board)
		if result:
			return result, moves0, moves1
	return 0, moves0, moves1


def main():
	net0 = Network()
	net1 = Network()
	init()
	print(match(net0, net1, 1))


if __name__ == '__main__':
	main()
