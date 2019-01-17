from network import Network
from chess.chess import ChessBoard


def match(player0, player1, stdout=0):
	board = ChessBoard()
	if stdout:
		board.out()
	current_player = 1
	moves = []
	while True:
		current_player = -current_player
		if not board.could_drop_by(current_player):
			if stdout:
				print(current_player, -1)
			moves.append(-1)
			continue
		if current_player == -1:
			move = player0.play(board, current_player)
		else:
			move = player1.play(board, current_player)
		moves.append(move)
		board.move(move, current_player)
		if stdout:
			print(current_player, move)
			board.out()
		if board.is_finish():
			return board.evaluate(), moves


def main():
	net0 = Network()
	net1 = Network()
	print(match(net0, net1, 1))


if __name__ == '__main__':
	main()
