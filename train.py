from network import Network
import match
import random
import parameter

network_number = parameter.network_number
net_pool = [Network('single_learn' + str(i)) for i in range(network_number)]


def random_choice_player():
	_player0 = random.randint(1, network_number) - 1
	_player1 = random.randint(1, network_number - 1) - 1
	if _player1 >= _player0:
		_player1 += 1
	print('player:', _player0, _player1)
	return net_pool[_player0], net_pool[_player1]


def save():
	for i in range(network_number):
		net_pool[i].save()
	print('model SAVED.')


def learning(max_epoch=parameter.max_epoch):
	print('Learning.')
	for epoch in range(max_epoch):
		print('--------------------------')
		print('train: ', epoch)
		player0, player1 = random_choice_player()
		result = match.match(player0, player1)
		print('result:', result[0])
		if result[0] > 0:
			player0.learn_to(result[1:], iam=-1, value=result[0])
		elif result[0] < 0:
			player1.learn_to(result[1:], iam=1, value=-result[0])
		if epoch % parameter.save_interval == 0:
			save()

	print('learning end.')


def main():
	# import profile
	# profile.run('learning()')
	learning()


if __name__ == '__main__':
	main()
