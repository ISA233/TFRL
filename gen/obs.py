from tools import unzip, load_data
from chess.chess import ChessBoard
import numpy as np


def main():
	data = load_data('lalala.pkl')
	# np.random.shuffle(data)
	# data = data[:100]
	Xs, Vs, Ps = unzip(data)
	for X, V, P in zip(Xs, Vs, Ps):
		board = ChessBoard(X[:, :, 0])
		player = X[0, 0, 1]
		board.out()
		print('o' if player == 1 else 'x')
		print(V)
		P_pass = P[64]
		P = P[:-1].reshape([8, 8])
		for i in range(8):
			for j in range(8):
				print('%.0f' % (P[i, j] * 100), end='\t')
			print()
		print('pass: %.0f' % (P_pass * 100))
		print('--------------------------------')


if __name__ == '__main__':
	main()
