from chess.chess import ChessBoard
from vnet import Network
from agent import Agent
from tools import version_str
from MCTS import Node
import profile


def match_with_human(agent, human_player=-1):
	board = ChessBoard()
	root = Node()
	board.out()
	print('---------------')
	current_player = -1
	while not board.is_finish():
		print('Player: ' + ('x' if current_player == -1 else 'o'))
		print(agent.net.vhead(board, current_player))
		if not board.could_drop_by(current_player):
			root = root.son(64)
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
			root = root.son(x * 8 + y)
		else:
			action = agent.play(board, current_player, root=root)
			board.move(action, current_player)
			root = root.son(action)
		board.out()
		current_player = -current_player
		print('---------------')
	v = board.evaluate()
	print('Result:', v)


def match(agent0, agent1, stdout=False):
	board = ChessBoard()
	root0, root1 = Node(), Node()
	current_player = 1
	while not board.is_finish():
		current_player = -current_player
		agent = agent0 if current_player == -1 else agent1
		root = root0 if current_player == -1 else root1
		if stdout:
			agent.analysis(board, current_player)
		print('root:', root.N)
		action = agent.play(board, current_player, root=root)
		board.move(action, current_player)
		root0 = root0.son(action)
		root1 = root1.son(action)
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
	net0 = Network('train', bn_training=False, use_GPU=True)
	net0.restore()
	net1 = Network('vnet' + version_str(6, 3), bn_training=False, use_GPU=False)
	net1.restore()
	agent0 = Agent(net0)
	agent1 = Agent(net1)
	# contest(agent0, agent0)
	# match(agent0, agent0, stdout=True)
	match_with_human(agent0, 1)


# contest(agent0, agent1)


if __name__ == '__main__':
	main()
# profile.run('main()', sort=1)
