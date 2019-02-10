from chess.chess import ChessBoard
from vnet import Network
from agent import Agent
from MCTS import MCT, Node
from tools import vector, log, get_time
from config import config
from random import randint, choice
import numpy as np
import pickle
import os


def move(agent, chessboards, roots, player, temperature, _dataset):
	MCTs = []
	for board, root in zip(chessboards, roots):
		if not board.is_finish():
			MCTs.append(MCT(board, player, root))
	if not MCTs:
		return False
	for simulate_cnt in range(config.gen_simulate_cnt):
		# print('simulate:', simulate_cnt)
		agent.simulates(MCTs)
	cnt = 0
	s = 0
	for i, board in enumerate(chessboards):
		if not board.is_finish():
			mct = MCTs[cnt]
			cnt += 1
			pi = mct.pi(temperature)
			_dataset.append((i, board.clone(), player, pi))
			action = np.random.choice(range(65), p=pi)
			board.move(action, player)
			roots[i].move_root(action)
			s += roots[i].N
	print(s / cnt)
	return True


def make(_dataset, chessboards):
	dataset = []
	mp = dict()
	for i, board in enumerate(chessboards):
		v = board.evaluate()
		v = 1 if v > 0 else -1 if v < 0 else 0
		mp[i] = v
		dataset.append((board.board_array(), -1, -v, vector(64)))
		dataset.append((board.board_array(), 1, v, vector(64)))
	for i, board, player, pi in _dataset:
		v = -mp[i] if player == -1 else mp[i]
		dataset.append((board.board_array(), player, v, pi))
	return dataset


def dump(_data, path, chessboards):
	data = make(_data, chessboards)
	if os.path.exists(path):
		data.extend(pickle.load(open(path, 'rb')))
	with open(path, 'wb') as f:
		pickle.dump(data, f)


def gen_batch_xot(agent, number, path):
	log('GEN_batch_xot: ' + path + ' ' + str(number) + '\t' + get_time())
	for init_player in [-1, 1]:
		data = []
		chessboards = [ChessBoard() for i in range(number // 2)]
		roots = [Node() for i in range(number // 2)]
		for board in chessboards:
			move_cnt = randint(4, 12) * 2 + (0 if init_player == -1 else 1)
			current_player = -1
			for i in range(move_cnt):
				dlist = board.drop_list(current_player)
				p = choice(dlist)
				board.move(p, current_player)
				current_player = -current_player
		player = init_player
		for move_cnt in range(120):
			print('gen_xot move_cnt:', move_cnt, player)
			if not move(agent, chessboards, roots, player, 0, data):
				break
			player = -player
		dump(data, path, chessboards)


def gen_batch(agent, number, path):
	log('GEN_batch: ' + path + ' ' + str(number) + '\t' + get_time())
	data = []
	chessboards = [ChessBoard() for i in range(number)]
	roots = [Node() for i in range(number)]
	player = -1
	for move_cnt in range(120):
		print('gen move_cnt:', move_cnt)
		temperature = 1 if move_cnt < 10 else 0
		if not move(agent, chessboards, roots, player, temperature, data):
			break
		player = -player
	dump(data, path, chessboards)


def gen(agent, number, path):
	log('GEN: ' + path + ' ' + str(number))
	while number:
		gen_batch_number = min(number, config.gen_batch_size)
		gen_batch(agent, gen_batch_number, path)
		number -= gen_batch_number


def main():
	net = Network('vnet008_11_2f', bn_training=False)
	net.restore()
	agent = Agent(net)
	gen_batch_xot(agent, 1024, 'gen/train008.pkl')


if __name__ == '__main__':
	main()
