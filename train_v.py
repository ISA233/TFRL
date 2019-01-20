from chess.chess import ChessBoard
import numpy as np
import tensorflow as tf
import pickle
from net_struct_v import net_train, net_test, train_graph, test_graph


class Network:
	def __init__(self, name='net', bn_training=True):
		self.bn_training = bn_training
		self.net = net_train if bn_training else net_test
		self.name = name
		self.sess = tf.Session(graph=train_graph if bn_training else test_graph)
		self.sess.run(self.net.initializer)
		self.saver = tf.train.Saver(self.net.vars, max_to_keep=0)
		self.data = []
		self.data_size = 0


	def load_data(self, path='gen/train.pkl'):
		print('load data.')
		__data = pickle.load(open(path, 'rb'))
		# __data = __data[:120]
		# print(__data[:3])
		for _data in __data:
			self.data.append((ChessBoard(_data[0]).to_network_input(_data[1]), [_data[3]]))
		self.data_size = len(self.data)
		# for data in self.data:
		# 	jb = np.array(data[0])
		# 	print(jb[:, :, 0])
		# 	print(jb[:, :, 1])
		# 	print(jb[:, :, 2])
		# 	print(data[1])
		# print(self.data[:3])
		print('load data down.')

	def train(self, epochs=100000, mini_batch_size=256):
		if not self.bn_training:
			raise Exception('bn_training is Flase. should not train.')
		print('training.')
		save_interval = 400 * mini_batch_size
		save_cnt = 0
		for epoch in range(epochs):
			print('epoch:', epoch)
			np.random.shuffle(self.data)
			for i in range(0, self.data_size, mini_batch_size):
				# print(i)
				mini_batch_data = self.data[i: i + mini_batch_size]
				_, loss = self.sess.run([self.net.train, self.net.cost],
				                        feed_dict={self.net.state: [_data[0] for _data in mini_batch_data],
				                                   self.net.value: [_data[1] for _data in mini_batch_data]})
				print('loss:', loss)
				# print('cost:',
				#       self.sess.run(self.net.cost, feed_dict={self.net.state: [_data[0] for _data in mini_batch_data],
				#                                               self.net.value: [_data[1] for _data in mini_batch_data]}))
				# if i % save_interval == 0 and epoch % 100 == 0:
				if i % save_interval == 0:
					print('save:', save_cnt)
					self.save(self.name + str(save_cnt).zfill(5))
					save_cnt += 1

	def getv(self, chessboard, player):
		return self.sess.run(self.net.probability, feed_dict={self.net.state: [chessboard.to_network_input(player)]})

	def save(self, name='None'):
		if name == 'None':
			name = self.name
		self.saver.save(self.sess, 'vnet_save/' + self.name + '/' + name)

	def restore(self, path='None'):
		if path == 'None':
			path = 'vnet_save/' + self.name + '/' + self.name
		self.saver.restore(self.sess, path)


def main():
	net = Network('cnn_vnet')
	net.load_data()
	net.train()


if __name__ == '__main__':
	main()
