from chess.chess import ChessBoard
from vnet import Network
from player import Player
from MCTS import MCT
from tools import version_str, vector
import numpy as np
import profile


def move(currentPlayer, chessboards, player, temperature, _dataset):
	MCTs = []
	for board in chessboards:
		if not board.is_finish():
			MCTs.append(MCT(board, player))
	for simulate_cnt in range(40):
		print('simulate:', simulate_cnt)
		currentPlayer.simulates(MCTs)
	# for mct in MCTs:
	# 	print(mct)
	# 	for action, _, son in mct.root.edges:
	# 		if son is not None:
	# 			print('%d %d: %d' % (action // 8, action % 8, son.N))
	cnt = 0
	for i, board in enumerate(chessboards):
		if not board.is_finish():
			mct = MCTs[cnt]
			cnt += 1
			pi = mct.pi(temperature)
			action = np.random.choice(range(65), p=pi)
			board.move(action, player)
			_dataset.append((i, board, player, pi))


def gen(Player0, Player1, number=128):
	_dataset, dataset = [], []
	chessboards = [ChessBoard() for i in range(number)]
	player = -1
	for move_cnt in range(64):
		print('############################# move_cnt:', move_cnt)
		currentPlayer = Player0 if player == -1 else Player1
		temperature = 1 if move_cnt < 12 else 0
		move(currentPlayer, chessboards, player, temperature, _dataset)
		player = -player
		# print('\n\n=================================================\n\n')
	mp = dict()
	for i, board in enumerate(chessboards):
		v = board.evaluate()
		v = 1 if v > 0 else -1 if v < 0 else 0
		mp[i] = v
		dataset.append((board.to_network_input(-1), -v, vector(64)))
		dataset.append((board.to_network_input(1), v, vector(64)))
	for i, board, player, pi in _dataset:
		v = -mp[i] if player == -1 else mp[i]
		dataset.append((board.to_network_input(player), v, pi))
	return dataset


def main():
	net = Network('cnn_vnet', bn_training=False)
	net.restore('../vnet_save', version=version_str(193))
	Player0 = Player1 = Player(net)
	dataset = gen(Player0, Player1)
	print(dataset)


if __name__ == '__main__':
	# main()
	profile.run('main()', sort=1)