from vnet import Network
from chess.chess import ChessBoard
from random import sample
import numpy as np
from tools import version_str, load_data, unzip


def cost(net):
	test_data = load_data('gen/test.pkl')
	test_X, test_Y, test_P = unzip(test_data)
	print(net.sess.run([net.net.v_loss, net.net.p_loss],
	                   feed_dict={net.net.state: test_X, net.net.v: test_Y, net.net.p: test_P}))


def test(net):
	print('test.')
	data = load_data('gen/test.pkl')
	data = sample(data, 40)
	Xs, Vs, _ = unzip(data)
	for X, V in zip(Xs, Vs):
		board = ChessBoard(X[:, :, 0])
		player = X[0, 0, 1]
		board.out()
		print('o' if player == 1 else 'x')
		print(V, net.vhead(board, player))
		net.dist_out(board, player)


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
	board = np.array([[x, x, o, o, o, o, x, x],
	                  [x, o, o, o, o, o, o, x],
	                  [x, o, o, o, o, o, o, x],
	                  [o, o, o, o, o, o, o, o],
	                  [o, o, o, o, o, o, o, o],
	                  [x, o, o, o, o, o, o, x],
	                  [x, o, o, o, o, o, o, x],
	                  [x, x, o, o, o, o, x, x]])
	board = ChessBoard(board)
	# board = ChessBoard()
	net.dist_out(board, -1)
	print(net.vhead(board, -1))


def main():
	net = Network('cnn_vnet', bn_training=False, use_GPU=False)
	net.restore(name='cnn_vnet', version=version_str(193))
	test(net)


# cost(net)


if __name__ == '__main__':
	main()
