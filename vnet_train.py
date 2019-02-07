from vnet import Network
from tools import log, get_time, load_data, load_data_with_mr, unzip
import numpy as np
from config import config


def test(net, test_data, mini_batch_size=1024):
	test_data_size = len(test_data)
	np.random.shuffle(test_data)
	v_loss_list, p_loss_list, loss_list = [], [], []
	for i in range(0, test_data_size, mini_batch_size):
		mini_batch_data = test_data[i: i + mini_batch_size]
		X, V, P = unzip(mini_batch_data)
		_v_loss, _p_loss, _loss = net.sess.run([net.net.v_loss, net.net.p_loss, net.net.loss],
		                                       feed_dict={net.net.state: X, net.net.v: V, net.net.p: P})
		v_loss_list.append(_v_loss)
		p_loss_list.append(_p_loss)
		loss_list.append(_loss)
	loss = float(np.mean(loss_list))
	v_loss = float(np.mean(v_loss_list))
	p_loss = float(np.mean(p_loss_list))
	return loss, v_loss, p_loss


def save(net, loss, vploss, args):
	if loss > args[1] and vploss > args[2]:
		args[3] += 1
		if args[3] >= config.bad_net_limit:
			if args[4] == config.learning_rate3:
				return False
			else:
				args[4] = config.learning_rate2 if args[4] == config.learning_rate else config.learning_rate3
				net.change_learning_rate(args[4])
				args[1] = 9999
				args[3] = 0
	if vploss < args[2]:
		net.save()
		# log('saved.' + '   ' + get_time())
		args[2] = vploss
		args[3] = 0
	if loss < args[1]:
		args[1] = loss
		args[3] = 0
	return True


def train_epoch(net, train_data, test_data, args, mini_batch_size):
	train_data_size = len(train_data)
	save_interval = 200 * mini_batch_size
	np.random.shuffle(train_data)
	for i in range(0, train_data_size, mini_batch_size):
		mini_batch_data = train_data[i: i + mini_batch_size]
		X, V, P = unzip(mini_batch_data)
		_, v_loss, p_loss = net.sess.run([net.net.train, net.net.v_loss, net.net.p_loss],
		                                 feed_dict={net.net.state: X, net.net.v: V, net.net.p: P})
		print('loss: %.6f %.6f' % (v_loss, p_loss))
		if i % save_interval == 0:
			print('save:', args[0])
			loss, v_loss, p_loss = test(net, test_data)
			vploss = v_loss + p_loss
			if not save(net, loss, vploss, args):
				return False
			log('%d:\t%.5f\t%.5f\t%.5f\t\t%.5f' % (args[0], loss, v_loss, p_loss, args[2]))
			args[0] += 1
	return True


def train(net, trainPath, testPath, mini_batch_size=256):
	log('\n###  start  ###')
	log('# ' + get_time())
	learning_rate = config.learning_rate
	net.change_learning_rate(learning_rate)
	test_data = load_data(testPath)

	print('training.')
	min_loss = min_vploss = 9999
	save_cnt = bad_net = epoch = 0
	args = [save_cnt, min_loss, min_vploss, bad_net, learning_rate]
	while True:
		train_data = load_data_with_mr(trainPath)
		epoch += 1
		print('epoch:', epoch)
		if not train_epoch(net, train_data, test_data, args, mini_batch_size):
			return


def train_vnet():
	net = Network('train100')
	# net.restore()
	trainPath = 'gen/train007.pkl'
	testPath = 'gen/test007.pkl'
	train(net, trainPath, testPath)


def main():
	train_vnet()


if __name__ == '__main__':
	main()
