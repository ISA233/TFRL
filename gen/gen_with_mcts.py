from chess.chess import ChessBoard
from vnet import Network
from MCTS import MCT


def move(chessboards_with_idx, player):
	MCTs_with_idx = []
	for i, board in chessboards_with_idx:
		if not board.is_finish():
			MCTs_with_idx.append((i, MCT(board, player)))
	for i, mct in MCTs_with_idx:
		leaf, board, player = mct.reach_leaf()



def gen(number=10000):
	chessboards_with_idx = [(i, ChessBoard()) for i in range(number)]
	player = -1
	for move_cnt in range(60):
		move(chessboards_with_idx, player)
		player = -player


def main():
	gen()


if __name__ == '__main__':
	main()
