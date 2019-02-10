from vnet import Network
from chess.chess import ChessBoard
from random import sample
import numpy as np
from tools import version, load_data, unzip
from agent import Agent
from MCTS import MCT


def cost(net):
	test_data = load_data('gen/test002.pkl')
	test_X, test_Y, test_P = unzip(test_data)
	print(net.sess.run([net.net.v_loss, net.net.p_loss],
	                   feed_dict={net.net.state: test_X, net.net.v: test_Y, net.net.p: test_P}))


def test(net):
	print('test.')
	data = load_data('gen/test008.pkl')
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
	# board = np.array([[o, _, _, x, o, _, _, o],
	#                   [_, _, _, _, o, _, _, _],
	#                   [_, _, o, x, o, _, _, _],
	#                   [_, _, o, x, x, o, _, _],
	#                   [_, x, x, x, x, _, _, _],
	#                   [_, _, o, o, o, o, _, _],
	#                   [_, _, _, _, _, _, _, _],
	#                   [o, _, _, _, _, _, _, o]])
	# board = np.array([[_, _, o, o, o, o, _, _],
	#                   [_, o, o, o, o, o, _, _],
	#                   [o, o, o, o, o, x, o, o],
	#                   [x, x, x, o, x, x, o, _],
	#                   [o, o, o, o, o, o, o, _],
	#                   [o, o, o, o, o, o, o, _],
	#                   [_, _, o, o, x, o, _, _],
	#                   [_, _, o, o, o, o, _, _]])
	board = np.array([[_, _, _, _, _, _, _, _],
	                  [_, _, _, _, _, _, _, _],
	                  [_, _, o, o, o, o, _, _],
	                  [_, _, _, o, o, o, _, _],
	                  [_, _, _, x, o, o, x, _],
	                  [_, x, _, o, o, o, _, _],
	                  [_, _, o, o, o, o, _, _],
	                  [_, o, _, x, _, o, _, _]])
	# board = np.array([[x, x, x, x, x, x, x, x],
	#                   [x, o, o, o, o, o, o, x],
	#                   [x, o, o, o, o, o, o, x],
	#                   [x, o, o, o, o, o, o, x],
	#                   [x, o, o, o, o, o, o, x],
	#                   [x, o, o, o, o, o, o, x],
	#                   [x, o, o, o, o, o, o, x],
	#                   [x, x, x, x, x, x, x, x]])
	player = -1
	board = ChessBoard(board)
	# board = ChessBoard()
	net.dist_out(board, player)
	print(net.vhead(board, player))
	agent = Agent(net)
	mct = MCT(board, player)
	for i in range(5000):
		agent.simulate(mct)
	root = mct.root

	def out(_root):
		for action, P, son in _root.edges:
			print('%2d' % action, '%.8f' % P, son, son.N if son else 0, '\t%5f' % son.Q if son else None)

	out(root)
	print('----------------------')
	out(root.son(10))
	print('----------------------')
	# out(root.son(8))
	print('----------------------')
	out(root.son(10).son(11))
	print('----------------------')
	out(root.son(10).son(11).son(5))
	print('----------------------')
	out(root.son(10).son(11).son(5).son(37))


def main():
	net = Network('vnet008_11_2f', bn_training=False, use_GPU=True)
	net.restore()
	test2(net)


# cost(net)


if __name__ == '__main__':
	main()
