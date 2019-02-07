from chess.chess import ChessBoard
import numpy as np


class Node:
	def __init__(self, father=None):
		self.V = self.N = self.Q = 0
		self.father = father
		self.edges = []  # (action, P, son)

	def update(self, v):
		self.N += 1
		self.V += v
		self.Q = self.V / self.N

	def expand(self, board, player, dist):
		for p in range(64):
			if board.could_drop(p, player):
				self.edges.append([p, dist[p], None])
		if not self.edges:
			self.edges.append([64, dist[64], None])

	def son(self, p):
		for action, P, son in self.edges:
			if action == p:
				return Node() if son is None else son
		return Node()


class MCT:
	def __init__(self, init_board=ChessBoard(), init_player=-1, root=None):
		self.init_board = init_board.clone()
		self.init_player = init_player
		self.root = Node() if root is None else root

	def reach_leaf(self):
		if not self.root.edges:
			return self.root, self.init_board, self.init_player
		currentNode = self.root
		board, player = self.init_board.clone(), self.init_player
		while not board.is_finish():
			MaxQU = sim_edge = None
			nu = np.random.dirichlet([0.03] * len(currentNode.edges))
			eps = 0.25 if currentNode == self.root else 0.05
			N = currentNode.N
			for i, edge in enumerate(currentNode.edges):
				action, P, son = edge
				ni = 0 if son is None else son.N
				U = ((1 - eps) * P + eps * nu[i]) * np.sqrt(N) / (1 + ni)
				Q = 0 if son is None else -son.Q
				if MaxQU is None or Q + U > MaxQU:
					MaxQU = Q + U
					sim_edge = edge
			# print('action: %d %d, Q: %.3f, U: %.3f, Q+U: %.3f' % (action // 8, action % 8, Q, U, Q + U))
			board.move(sim_edge[0], player)
			player = -player
			if sim_edge[2] is None:
				sim_edge[2] = Node(currentNode)
				return sim_edge[2], board, player
			currentNode = sim_edge[2]
		return currentNode, board, player

	def pi(self, temperature=1):
		pi = np.zeros(65)
		if temperature:
			N = self.root.N - 1
			for action, _, son in self.root.edges:
				ni = 0 if son is None else son.N
				pi[action] = ni / N
		else:
			select_edge = None
			select_edge_N = 0
			for edge in self.root.edges:
				if edge[2] is not None and edge[2].N > select_edge_N:
					select_edge = edge
					select_edge_N = edge[2].N
			pi[select_edge[0]] = 1
		return pi


def back_up(node, v):
	while node is not None:
		node.update(v)
		node = node.father
		v = -v


def _reach_leaf(mct):
	return mct.reach_leaf()


def main():
	pass


if __name__ == '__main__':
	main()
# profile.run('main()', sort=1)
