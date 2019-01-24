import numpy as np
import pickle
from chess.chess import ChessBoard
import time


def get_time():
	return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))


def version_str(ver):
	return str(ver).zfill(5)


def unzip(data):
	X = [_data[0] for _data in data]
	V = [_data[1] for _data in data]
	P = [_data[2] for _data in data]
	return X, V, P


def load_data(path='gen/train.pkl'):
	__data = pickle.load(open(path, 'rb'))
	data = []
	for _data in __data:
		data.append((ChessBoard(_data[0]).to_network_input(_data[1]), [_data[3] * 2 - 1], vector(_data[2])))
	return data


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
	print(type(get_time()))


if __name__ == '__main__':
	main()
