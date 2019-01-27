from chess.chess import ChessBoard
import numpy as np
from vnet import Network
from tools import version_str, log
import profile


class Node:
	def __init__(self, father=None):
		self.V = self.N = self.Q = 0
		self.father = father
		self.edges = []  # (action, P, son)

	def update(self, v):
		self.N += 1
		self.V += v
		self.Q = self.V / self.N

	def expand(self, drop_list, dist):
		print('expand:')
		if not drop_list:
			drop_list = [64]
		for p in drop_list:
			print('%d %d: %.4f' % (p // 8, p % 8, dist[p]))
			self.edges.append([p, dist[p], None])


class MCT:
	def __init__(self, init_board=ChessBoard(), init_player=-1):
		self.init_board = init_board.clone()
		self.init_player = init_player
		self.root = Node()

	def reach_leaf(self):
		if not self.root.edges:
			return self.root, self.init_board, self.init_player
		currentNode = self.root
		board, player = self.init_board.clone(), self.init_player
		while not board.is_finish():
			print('move_to_leaf:')
			board.out()
			MaxQU = sim_edge = None
			# if currentNode == self.root:
			# 	eps = 0.5
			# 	nu = np.random.dirichlet([0.3] * len(currentNode.edges))
			# else:
			# 	eps = 0
			# 	nu = [0] * len(currentNode.edges)
			N = currentNode.N
			for i, edge in enumerate(currentNode.edges):
				action, P, son = edge
				ni = 0 if son is None else son.N
				# U = ((1 - eps) * P + eps * nu[i]) * np.sqrt(N) / (1 + ni)
				U = P * np.sqrt(N) / (1 + ni)
				Q = 0 if son is None else -son.Q
				if MaxQU is None or Q + U > MaxQU:
					MaxQU = Q + U
					sim_edge = edge
				print('action: %d %d, Q: %.3f, U: %.3f, Q+U: %.3f' % (action // 8, action % 8, Q, U, Q + U))
			if sim_edge[0] == 64:
				print('wooooooooooooooooow')
			board.move(sim_edge[0], player)
			player = -player
			if sim_edge[2] is None:
				sim_edge[2] = Node(currentNode)
				return sim_edge[2], board, player
			currentNode = sim_edge[2]
		return currentNode, board, player

	@staticmethod
	def back_up(node, v):
		while node is not None:
			node.update(v)
			node = node.father
			v = -v

	# def simulate(self):
	# 	leaf, board, player = self.reach_leaf()
	# 	board.out()
	# 	if board.is_finish():
	# 		v = board.win(player)
	# 	else:
	# 		v, dist = self.net.pred(board, player)
	# 		leaf.expand(board.drop_list(player), dist)
	# 	print('evaluate: ', v, player)
	# 	self.back_up(leaf, v)


def main():
	# net = Network('cnn_vnet', bn_training=False, use_GPU=False)
	# net.restore(version=version_str(193))
	# _, x, o = 0, -1, 1
	# # board = np.array([[o, x, x, x, x, x, _, o],
	# #                   [_, x, o, o, x, _, o, o],
	# #                   [x, x, o, o, x, o, o, x],
	# #                   [x, o, o, o, x, o, o, o],
	# #                   [x, x, _, o, x, _, o, o],
	# #                   [x, x, x, o, x, o, _, o],
	# #                   [x, x, x, x, x, o, o, _],
	# #                   [o, o, o, o, _, x, x, o]])
	# # board = ChessBoard(board)
	# board = ChessBoard()
	# mcts = MCT(net, board, 1)
	# for i in range(50):
	# 	mcts.simulate()
	# 	print('====================================')
	# for action, _, son in mcts.root.edges:
	# 	if son is not None:
	# 		print(action, son.N, -son.Q)
	pass


if __name__ == '__main__':
	main()
# profile.run('main()', sort=1)
'''
action: 2 6, Q: 0.290, U: 0.147, Q+U: 0.437
action: 3 5, Q: 0.218, U: 0.209, Q+U: 0.427
action: 4 2, Q: 0.000, U: 0.000, Q+U: 0.000
action: 5 3, Q: 0.000, U: 0.000, Q+U: 0.000
move_to_leaf:
_ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ 
_ _ _ _ o o o _ 
_ _ _ o x _ _ _ 
_ _ _ x o _ _ _ 
_ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ 
action: 1 4, Q: -0.273, U: 0.155, Q+U: -0.119
action: 1 6, Q: -0.259, U: 0.284, Q+U: 0.025
action: 2 3, Q: 0.000, U: 0.159, Q+U: 0.159
action: 3 2, Q: -0.289, U: 0.229, Q+U: -0.060
action: 4 5, Q: 0.000, U: 0.001, Q+U: 0.001
action: 5 4, Q: -0.309, U: 0.269, Q+U: -0.040
_ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ 
_ _ _ x o o o _ 
_ _ _ x x _ _ _ 
_ _ _ x o _ _ _ 
_ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ 
_ _ _ _ _ _ _ _ 
expand:
2 2: 0.7785
4 2: 0.0003
5 2: 0.0694
evaluate:  0.2608558 1
====================================
20 24 0.24721371056511998
29 25 0.183956900537014

Process finished with exit code 0
'''