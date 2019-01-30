from chess.chess import ChessBoard
from vnet import Network
from agent import Agent
from tools import version_str
from random import random
import profile


def match(agent0, agent1, stdout=False):
	board = ChessBoard()
	current_player = 1
	while not board.is_finish():
		current_player = -current_player
		if not board.could_drop_by(current_player):
			continue
		agent = agent0 if current_player == -1 else agent1
		if stdout:
			agent.analysis(board, current_player)
		action = agent.play(board, current_player)
		board.move(action, current_player)
		if stdout:
			print('x' if current_player == -1 else 'o')
			board.out()
			print('=============================')
	return board


def contest(agent0, agent1, match_number=100):
	win_cnt = 0
	for cnt in range(match_number):
		_agent0, _agent1 = agent0, agent1
		if random() < 0.5:
			_agent0, _agent1 = agent1, agent0
		board = match(_agent0, _agent1)
		v = board.evaluate()
		winner = None if not v else _agent0 if v < 0 else _agent1
		if winner == agent0:
			win_cnt += 1
		print('contest %d %.3f' % (cnt + 1, win_cnt / (cnt + 1)))
	return win_cnt / match_number


def main():
	net0 = Network('vnet' + version_str(1, 3), use_GPU=False)
	net0.restore()
	net1 = Network('vnet' + version_str(2, 3), use_GPU=False)
	net1.restore()
	agent0 = Agent(net0)
	agent1 = Agent(net1)
	match(agent0, agent1, stdout=True)


if __name__ == '__main__':
	main()
# profile.run('main()', sort=1)
