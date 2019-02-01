from vnet import Network
from chess.chess import ChessBoard
from random import sample
import numpy as np
from tools import version_str, load_data, unzip


def cost(net):
	test_data = load_data('gen/test002.pkl')
	test_X, test_Y, test_P = unzip(test_data)
	print(net.sess.run([net.net.v_loss, net.net.p_loss],
	                   feed_dict={net.net.state: test_X, net.net.v: test_Y, net.net.p: test_P}))


def test(net):
	print('test.')
	data = load_data('gen/test004.pkl')
	data = sample(data, 40)
	Xs, Vs, Ps = unzip(data)
	for X, V, P in zip(Xs, Vs, Ps):
		board = ChessBoard(X[:, :, 0])
		player = X[0, 0, 1]
		board.out()
		print('o' if player == 1 else 'x')
		print(V, net.vhead(board, player))
		net.dist_out(board, player)
		print()
		P_pass = P[64]
		P = P[:-1].reshape([8, 8])
		for i in range(8):
			for j in range(8):
				print('%.2f' % (P[i, j] * 100), end='\t')
			print()
		print('pass: %.2f' % (P_pass * 100))
		print('--------------------------------')


def test2(net):
	_, x, o = 0, -1, 1
	# board = np.array([[o, _, _, _, _, _, _, x],
	#                   [_, _, _, _, _, _, _, _],
	#                   [_, _, _, _, _, _, _, _],
	#                   [_, _, _, o, x, _, _, _],
	#                   [_, _, _, x, o, _, _, _],
	#                   [_, _, _, _, _, _, _, _],
	#                   [_, _, _, _, _, _, _, _],
	#                   [x, _, _, _, _, _, _, x]])
	# board = np.array([[x, x, o, o, o, o, x, x],
	#                   [x, o, o, o, o, o, o, x],
	#                   [x, o, o, o, o, o, o, x],
	#                   [o, o, o, o, o, o, o, o],
	#                   [o, o, o, o, o, o, o, o],
	#                   [x, o, o, o, o, o, o, x],
	#                   [x, o, o, o, o, o, o, x],
	#                   [x, x, o, o, o, o, x, x]])
	# board = ChessBoard(board)
	board = ChessBoard()
	net.dist_out(board, -1)
	print(net.vhead(board, -1))


def main():
	net = Network('train', bn_training=False, use_GPU=False)
	net.restore()
	test(net)


# cost(net)


if __name__ == '__main__':
	main()
