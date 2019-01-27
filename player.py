from MCTS import MCT
from chess.chess import ChessBoard
from vnet import Network
from tools import version_str


class Player:
	def __init__(self, net):
		self.net = net

	def simulate(self, mct):
		leaf, board, player = mct.reach_leaf()
		board.out()
		if board.is_finish():
			v = board.win(player)
		else:
			v, dist = self.net.pred(board, player)
			leaf.expand(board.drop_list(player), dist)
		print('evaluate: ', v, player)
		mct.back_up(leaf, v)

	def mcts(self, board, player):
		mct = MCT(board, player)
		for i in range(50):
			self.simulate(mct)
			print('====================================')
		for action, _, son in mct.root.edges:
			if son is not None:
				print(action, son.N, -son.Q)


def main():
	board = ChessBoard()
	net = Network('cnn_vnet', bn_training=False, use_GPU=False)
	net.restore(version=version_str(193))
	player = Player(net)
	player.mcts(board, 1)


if __name__ == '__main__':
	main()
