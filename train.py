from network import Network
import match
import random
import parameter

network_number = parameter.network_number
net_pool = [Network('cnn_net' + str(i)) for i in range(network_number)]


def learning(max_epoch=parameter.max_epoch):
	print('Learning.')
	for epoch in range(max_epoch):
		print('--------------------------')
		print('train: ', epoch)
		_player0 = random.randint(1, network_number) - 1
		_player1 = random.randint(1, network_number - 1) - 1
		if _player1 >= _player0:
			_player1 += 1
		print('player:', _player0, _player1)
		player0 = net_pool[_player0]
		player1 = net_pool[_player1]
		result = match.match(player0, player1)
		print('result:', result[0])
		if result[0] > 0:
			player0.learn_to(result[1:], iam=-1, value=result[0])
		elif result[0] < 0:
			player1.learn_to(result[1:], iam=1, value=-result[0])
		if epoch % 100 == 0:
			for i in range(network_number):
				net_pool[i].save()
			print('model SAVED.')
	print('learning end.')


def main():
	# import profile
	# profile.run('learning()')
	learning()


if __name__ == '__main__':
	main()
