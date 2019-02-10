import numpy as np
import pickle
from chess.chess import ChessBoard
from chess.mr import rotate_int, rotate_double, mirror_int, mirror_double
import time


def get_time():
	return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))


def version(ver, zfill=3):
	return str(ver).zfill(zfill)


def unzip(data):
	X = [_data[0] for _data in data]
	V = [_data[1] for _data in data]
	P = [_data[2] for _data in data]
	return X, V, P


def load_data(path='gen/train.pkl'):
	_data = pickle.load(open(path, 'rb'))
	data = []  # (net_input, v, pi)
	for board, player, v, dist in _data:
		data.append((ChessBoard(board).to_network_input(player), [v], dist))
	return data


def load_data_with_mr(path='gen/train.pkl'):
	_data = pickle.load(open(path, 'rb'))
	data = []  # (net_input, v, pi)
	for board, player, v, dist in _data:
		mirror_cnt = np.random.randint(2)
		rotate_cnt = np.random.randint(4)
		_board = rotate(mirror(board, mirror_cnt), rotate_cnt)
		_dist = dist.copy()
		_dist_board = _dist[:64].reshape(8, 8)
		_dist_board = rotate(mirror(_dist_board, mirror_cnt), rotate_cnt)
		_dist[:64] = _dist_board.reshape(64)
		data.append((ChessBoard(_board).to_network_input(player), [v], _dist))
	return data


def rotate(board, k=1):
	if board.dtype == 'int32':
		return np.array(rotate_int(board, k))
	return np.array(rotate_double(board, k))


def mirror(board, k=1):
	if board.dtype == 'int32':
		return np.array(mirror_int(board, k))
	return np.array(mirror_double(board, k))


def log(string='', file='log.txt'):
	logfile = open(file, 'a')
	logfile.write(string + '\n')
	logfile.close()


def player_01(player):
	if player == 1:
		return 1
	return 0


def vector(place, size=65):
	v = np.zeros([size])
	v[place] = 1
	return v


def porn(x):
	if x > 0:
		return 1
	if x < 0:
		return -1
	return 0


def main():
	# data = pickle.load(open('gen/test004.pkl', 'rb'))
	# np.random.shuffle(data)
	# board = data[0][0]
	# b0 = ChessBoard(board)
	# board = rotate(board)
	# b1 = ChessBoard(board)
	# b0.out()
	# print('------------')
	# b1.out()
	# dist = data[0][3].copy()
	# print(dist)
	# dist_board = dist[:64].reshape(8, 8)
	# print(dist_board)
	# dist_board = mirror(dist_board)
	# print(dist_board)
	# print(dist)
	# data = load_data('gen/test004.pkl')[20000:20008]
	# for board, v, dist in data:
	# 	board = np.array(board[:, :, 0])
	# 	print(board)
	# 	dist = dist[:64].reshape(8, 8)
	# 	for i in range(8):
	# 		for j in range(8):
	# 			print('%.2f' % dist[i, j], end='\t')
	# 		print()
	# 	print('------------')
	data = load_data_with_mr('gen/test004.pkl')
	np.random.shuffle(data)
	board, v, dist = data[0]
	print(np.array(board[:, :, 0]))
	print(v)
	print(dist[:64].reshape(8, 8))


if __name__ == '__main__':
	main()
