import os
import parameter
from chess.chess import ChessBoard
import tensorflow as tf
from vnet_struct import net_train, net_test, train_graph, test_graph
from tools import log


class Network:
	def __init__(self, name='net', bn_training=True, use_GPU=True):
		if not use_GPU:
			os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
			os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
		self.bn_training = bn_training
		self.net = net_train if bn_training else net_test
		self.name = name
		self.sess = tf.Session(graph=train_graph if bn_training else test_graph)
		self.sess.run(self.net.initializer)
		self.saver = tf.train.Saver(self.net.vars, max_to_keep=0)

	def change_learning_rate(self, learning_rate=parameter.learning_rate):
		self.sess.run(tf.assign(self.net.learning_rate, learning_rate))
		log('### Change learning rate to: ' + str(learning_rate))

	def vhead(self, chessboard, player):
		return self.sess.run(self.net.vhead, feed_dict={self.net.state: [chessboard.to_network_input(player)]})[0]

	def phead(self, chessboard, player):
		return self.sess.run(self.net.phead, feed_dict={self.net.state: [chessboard.to_network_input(player)]})[0]

	def dist(self, chessboard, player):
		return self.sess.run(self.net.dist, feed_dict={self.net.state: [chessboard.to_network_input(player)]})[0]

	def dist_out(self, chessboard, player):
		dist = self.dist(chessboard, player)
		pass_move = dist[-1]
		dist = dist[:-1].reshape([8, 8])
		for i in range(8):
			for j in range(8):
				print('%.2f' % (dist[i, j] * 100), end='\t')
			print()
		print('pass: %.2f' % (pass_move * 100))
		print('--------------------------------')

	def save(self, path='vnet_save', version=''):
		self.saver.save(self.sess, path + '/' + self.name + '/' + self.name + version)

	def restore(self, path='vnet_save', name='', version=''):
		if name == '':
			name = self.name
		self.saver.restore(self.sess, path + '/' + name + '/' + name + version)


def main():
	net = Network('cnn_vnet', use_GPU=False)
	board = ChessBoard()
	net.dist_out(board, -1)


if __name__ == '__main__':
	main()
