from vnet import Network
from tools import log, get_time, load_data, unzip, version_str
import numpy as np


def test(net, X, V, P, save_cnt):
	v_loss, p_loss, loss = net.sess.run([net.net.v_loss, net.net.p_loss, net.net.loss],
	                                    feed_dict={net.net.state: X, net.net.v: V, net.net.p: P})
	log('%d:\t%.5f\t%.5f\t%.5f' % (save_cnt, loss, v_loss, p_loss))


def train(net, epochs=100000, mini_batch_size=256):
	if not net.bn_training:
		raise Exception('bn_training is Flase. should not train.')
	log('\n###  start  ###')
	log('# ' + get_time())
	net.restore(version=version_str(187))
	net.change_learning_rate(0.003)

	print('load data.')
	train_data = load_data('gen/train.pkl')
	train_data_size = len(train_data)
	test_data = load_data('gen/test.pkl')
	test_X, test_V, test_P = unzip(test_data)

	print('training.')
	save_interval = 200 * mini_batch_size
	save_cnt = 0
	for epoch in range(epochs):
		print('epoch:', epoch)
		np.random.shuffle(train_data)
		for i in range(0, train_data_size, mini_batch_size):
			mini_batch_data = train_data[i: i + mini_batch_size]
			X, V, P = unzip(mini_batch_data)
			_, v_loss, p_loss = net.sess.run([net.net.train, net.net.v_loss, net.net.p_loss],
			                                 feed_dict={net.net.state: X, net.net.v: V, net.net.p: P})
			print('loss:', v_loss, p_loss)
			if i % save_interval == 0:
				print('save:', save_cnt)
				net.save(version=version_str(save_cnt))
				test(net, test_X, test_V, test_P, save_cnt)
				save_cnt += 1


def train_vnet():
	net = Network('cnn_vnet')
	train(net)


def main():
	train_vnet()


if __name__ == '__main__':
	main()
