from vnet import Network
from tools import log, get_time, load_data, unzip
import numpy as np
import parameter


def test(net, X, V, P, save_cnt):
	v_loss, p_loss, loss = net.sess.run([net.net.v_loss, net.net.p_loss, net.net.loss],
	                                    feed_dict={net.net.state: X, net.net.v: V, net.net.p: P})
	log('%d:\t%.5f\t%.5f\t%.5f' % (save_cnt, loss, v_loss, p_loss))
	return loss, v_loss, p_loss


def train(net, trainPath='gen/train.pkl', testPath='gen/test.pkl', mini_batch_size=256):
	if not net.bn_training:
		raise Exception('bn_training is Flase. should not train.')
	log('\n###  start  ###')
	log('# ' + get_time())
	# net.restore(version=version_str(187))
	learning_rate = parameter.learning_rate
	net.change_learning_rate(learning_rate)
	INF = 9999
	min_loss = min_vploss = INF
	bad_net = 0

	print('load data.')
	train_data = load_data(trainPath)
	train_data_size = len(train_data)
	test_data = load_data(testPath)
	test_X, test_V, test_P = unzip(test_data)

	print('training.')
	save_interval = 200 * mini_batch_size
	save_cnt = epoch = 0
	while True:
		epoch += 1
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
				loss, v_loss, p_loss = test(net, test_X, test_V, test_P, save_cnt)
				vploss = v_loss + p_loss
				if loss > min_loss and vploss > min_vploss:
					bad_net += 1
					if bad_net >= 5:
						if learning_rate == parameter.learning_rate2:
							return
						else:
							learning_rate = parameter.learning_rate2
							net.change_learning_rate(learning_rate)
							min_loss = INF
							bad_net = 0
				if vploss < min_vploss:
					net.save()
					min_vploss = vploss
					bad_net = 0
				if loss < min_loss:
					min_loss = loss
					bad_net = 0
				save_cnt += 1


def train_vnet():
	net = Network('train')
	trainPath = 'gen/train001.pkl'
	testPath = 'gen/test001.pkl'
	train(net, trainPath, testPath)


def main():
	train_vnet()


if __name__ == '__main__':
	main()
