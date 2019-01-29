from MCTS import MCT, back_up
from chess.chess import ChessBoard
from vnet import Network
from tools import version_str
import parameter


class Player:
	def __init__(self, net):
		self.net = net

	def simulate(self, mct):
		leaf, board, player = mct.reach_leaf()
		# board.out()
		if board.is_finish():
			v = board.win(player)
		else:
			v, dist = self.net.pred(board, player)
			leaf.expand(board, player, dist)
		# print('evaluate: ', v, player)
		back_up(leaf, v)

	def simulates(self, mcts):
		leaf_board_player, net_inputs = [], []
		for mct in mcts:
			leaf, board, player = mct.reach_leaf()
			# board.out()
			if board.is_finish():
				v = board.win(player)
				back_up(leaf, v)
			else:
				net_inputs.append(board.to_network_input(player))
				leaf_board_player.append((leaf, board, player))
		if net_inputs:
			vs, dists = self.net.sess.run([self.net.net.vhead, self.net.net.dist],
			                              feed_dict={self.net.net.state: net_inputs})
			for v, dist, (leaf, board, player) in zip(vs, dists, leaf_board_player):
				leaf.expand(board, player, dist)
				back_up(leaf, v)

	def mcts(self, board, player, simulate_number=parameter.match_simulate_cnt):
		mct = MCT(board, player)
		for i in range(simulate_number):
			self.simulate(mct)
		# for action, _, son in mct.root.edges:
		# 	if son is not None:
		# 		print('%d %d: %.3f %d' % (action // 8, action % 8, -son.Q, son.N))
		return mct


def main():
	board = ChessBoard()
	net = Network('cnn_vnet', bn_training=False, use_GPU=False)
	net.restore(version=version_str(193))
	player = Player(net)
	player.mcts(board, 1)


if __name__ == '__main__':
	main()
