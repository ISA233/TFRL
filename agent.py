from MCTS import MCT, back_up
from chess.chess import ChessBoard
from vnet import Network
import numpy as np
from config import config


class Agent:
	def __init__(self, net):
		self.net = net

	def simulate(self, mct):
		leaf, board, player = mct.reach_leaf()
		if board.is_finish():
			v = board.win(player)
		else:
			v, dist = self.net.pred(board, player)
			leaf.expand(board, player, dist)
		back_up(leaf, v)

	def simulates(self, mcts):
		leaf_board_player, net_inputs = [], []
		for mct in mcts:
			leaf, board, player = mct.reach_leaf()
			if board.is_finish():
				v = board.win(player)
				back_up(leaf, v)
			else:
				net_inputs.append(board.to_network_input(player))
				leaf_board_player.append((leaf, board, player))
		if net_inputs:
			vs, dists = self.net.preds(net_inputs)
			for v, dist, (leaf, board, player) in zip(vs, dists, leaf_board_player):
				leaf.expand(board, player, dist)
				back_up(leaf, v)

	def mcts(self, board, player, simulate_number=config.simulate_cnt, root=None):
		mct = MCT(board, player, root)
		for i in range(simulate_number):
			self.simulate(mct)
		# for action, _, son in mct.root.edges:
		# 	if son is not None:
		# 		print('%d %d: %.3f %d' % (action // 8, action % 8, -son.Q, son.N))
		return mct

	def play(self, board, player, temperature=0, root=None):
		mct = self.mcts(board, player, config.simulate_cnt, root)
		pi = mct.pi(temperature)
		return np.random.choice(range(65), p=pi)

	def analysis(self, board, player):
		print(self.net.vhead(board, player))
		self.net.dist_out(board, player)


def main():
	_, x, o = 0, -1, 1
	board = np.array([[_, _, _, _, _, _, _, _],
	                  [_, _, _, _, _, _, _, _],
	                  [_, _, _, _, x, _, _, _],
	                  [_, _, _, o, x, o, _, _],
	                  [_, _, _, x, x, _, _, _],
	                  [_, _, _, _, x, _, _, _],
	                  [_, _, _, _, _, _, _, _],
	                  [_, _, _, _, _, _, _, _]])
	board = ChessBoard(board)
	net = Network('vnet005', bn_training=False, use_GPU=False)
	net.restore()
	agent = Agent(net)
	mct = agent.mcts(board, 1, 100)
	for action, P, son in mct.root.edges:
		print('%d %d: %d' % (action // 8, action % 8, son.N * son.N))


if __name__ == '__main__':
	main()
