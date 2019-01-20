import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from train_v import Network
from chess.chess import ChessBoard
from random import sample
import numpy as np


def test(net):
	print('test.')
	# print(net.data)
	data = sample(net.data, 10)
	state = [_data[0] for _data in data]
	value = net.sess.run(net.net.probability, feed_dict={net.net.state: state})
	for _data, v in zip(data, value):
		print(np.array(_data[0])[:, :, 0])
		print(np.array(_data[0])[0, 0, 2])
		print(_data[1], v)


def test2(net):
	_, x, o = 0, -1, 1
	board = np.array([[x, _, _, _, _, _, _, o],
	                  [_, _, _, _, _, _, _, _],
	                  [_, _, _, _, _, _, _, _],
	                  [_, _, _, o, x, _, _, _],
	                  [_, _, _, x, o, _, _, _],
	                  [_, _, _, _, _, _, _, _],
	                  [_, _, _, _, _, _, _, _],
	                  [x, _, _, _, _, _, _, x]])
	board = ChessBoard(board)
	# board = ChessBoard()
	print(net.getv(board, -1))


def main():
	net = Network('cnn_vnet', bn_training=False)
	net.load_data('gen/test.pkl')
	net.restore('vnet_save/cnn_vnet/cnn_vnet' + str(50).zfill(5))
	test(net)


if __name__ == '__main__':
	main()
