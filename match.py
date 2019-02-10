from chess.chess import ChessBoard
from vnet import Network
from agent import Agent
from tools import version
from MCTS import Node
import numpy as np


def match(agent0, agent1, stdout=False):
	_, x, o = 0, -1, 1
	board = np.array([[_, _, _, _, _, _, _, _],
	                  [_, _, x, _, _, _, _, _],
	                  [x, x, _, o, x, _, _, _],
	                  [x, x, o, x, x, _, x, _],
	                  [x, o, x, o, x, x, o, _],
	                  [x, o, o, x, x, x, x, x],
	                  [x, o, o, o, x, o, _, _],
	                  [_, _, x, o, x, x, _, _]])
	board = ChessBoard(board)
	root0, root1 = Node(), Node()
	current_player = 1
	while not board.is_finish():
		agent = agent0 if current_player == -1 else agent1
		root = root0 if current_player == -1 else root1
		if stdout:
			agent.analysis(board, current_player)
		print('root:', root.N)
		action = agent.play(board, current_player, root=root)
		board.move(action, current_player)
		root0.move_root(action)
		root1.move_root(action)
		if stdout:
			print('x' if current_player == -1 else 'o')
			board.out()
			print('=============================')
		current_player = -current_player
	return board


def contest(agent0, agent1, match_number=100):
	win_cnt = 0
	_agent0, _agent1 = agent0, agent1
	for cnt in range(match_number):
		_agent0, _agent1 = _agent1, _agent0
		if _agent0 == agent0:
			print('x')
		else:
			print('o')
		board = match(_agent0, _agent1)
		v = board.evaluate()
		winner = None if not v else _agent0 if v < 0 else _agent1
		if winner == agent0:
			win_cnt += 1
		print('contest %d %.3f' % (cnt + 1, win_cnt / (cnt + 1)))
	return win_cnt / match_number


def main():
	net0 = Network('train', bn_training=False, use_GPU=False)
	net0.restore()
	net1 = Network('vnet' + version(6), bn_training=False, use_GPU=False)
	net1.restore()
	agent0 = Agent(net0)
	agent1 = Agent(net1)
	# contest(agent0, agent0)
	match(agent0, agent0, stdout=True)


# contest(agent0, agent1)


if __name__ == '__main__':
	main()
# profile.run('main()', sort=1)
