from chess.chess import ChessBoard
from vnet import Network
from agent import Agent
from tools import version_str
import profile


def match_with_human(agent, human_player=-1):
	board = ChessBoard()
	board.out()
	print('---------------')
	current_player = -1
	while not board.is_finish():
		print('Player: ' + ('x' if current_player == -1 else 'o'))
		print(agent.net.vhead(board, current_player))
		if not board.could_drop_by(current_player):
			current_player = -current_player
			continue
		if current_player == human_player:
			print('Your turn: ', end='')
			try:
				x, y = map(int, input().split())
			except ValueError:
				print('* Input format ERROR.')
				continue
			if not board.could_drop_xy(x, y, current_player):
				print('* Can\'t move here.')
				continue
			board.move_xy(x, y, current_player)
		else:
			action = agent.play(board, current_player)
			board.move(action, current_player)
		board.out()
		current_player = -current_player
		print('---------------')
	v = board.evaluate()
	print('Result:', v)


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
	net0 = Network('vnet' + version_str(5, 3), bn_training=False, use_GPU=False)
	net0.restore()
	net1 = Network('vnet' + version_str(4, 3), bn_training=False, use_GPU=False)
	net1.restore()
	agent0 = Agent(net0)
	agent1 = Agent(net1)
	# match(agent0, agent1, stdout=True)
	match_with_human(agent0, 1)


# contest(agent0, agent1)


if __name__ == '__main__':
	main()
# profile.run('main()', sort=1)
