from network import Network
from chess.chess import ChessBoard
import pickle


def gen_one(player0, player1):
	board = ChessBoard()
	current_player = 1
	states = []
	while True:
		current_player = -current_player
		if not board.could_drop_by(current_player):
			states.append([board.board_array(), current_player, -1, 0.5])
			continue
		if current_player == -1:
			move = player0.play(board, current_player)
		else:
			move = player1.play(board, current_player)
		states.append([board.board_array(), current_player, move, 0.5])
		board.move(move, current_player)
		if board.is_finish():
			states.append([board.board_array(), current_player, -1, 0.5])
			value = board.evaluate()
			for state in states:
				# state[0].out()
				if value > 0:
					state[3] = 1.0 if state[1] == 1 else 0.0
				if value < 0:
					state[3] = 0.0 if state[1] == 1 else 1.0
			# print(state[1:])
			# print(value)
			break
	return states


def gen(player0, player1, data_size=10000):
	status = []
	for i in range(data_size):
		if i % 100 == 0:
			print('GENing: ', i)
		status += gen_one(player0, player1)
	# print(status)
	with open('train.pkl', 'wb') as f:
		pickle.dump(status, f)


def main():
	# print(pickle.load(open('train.pkl', 'rb')))
	net0 = Network()
	net1 = Network()
	net0.restore('../model_save/cnn_fc_net0/cnn_fc_net0')
	net1.restore('../model_save/cnn_fc_net0/cnn_fc_net0')
	gen(net0, net1, 10000)


if __name__ == '__main__':
	main()
