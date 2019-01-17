from network import Network
from tools import porn
import match
import random
import parameter
import profile

network_number = parameter.network_number
network_name = 'cnn_fc_net'


def learning(max_epoch=parameter.max_epoch):
	print('Learning.')
	net_pool = [Network(network_name + str(i)) for i in range(network_number)]
	new_player_id = network_number - 1
	for epoch in range(max_epoch):
		print('----------------------------')
		new_player_id = (new_player_id + 1) % (network_number + 1)
		new_player = Network(network_name + str(new_player_id))
		print('train', epoch, new_player_id, parameter.learning_rate)
		history = [0] * 250
		cnt = 0
		while True:
			cnt += 1
			if cnt >= 5000:
				break
			opponent = random.choice(net_pool)
			xo = (random.randint(0, 1) * 2) - 1
			if xo == -1:
				result, moves = match.match(new_player, opponent)
				result = -porn(result)
			else:
				result, moves = match.match(opponent, new_player)
				result = porn(result)
			new_player.learn_to(moves, iam=xo, value=result)
			history = history[1:] + [result]
			p = history.count(1) / 250
			print(cnt, ':', p)
			# if p > 0.65:
			# 	break
		net_pool = net_pool[1:] + [new_player]
		new_player.save()
		break
	print('learning end.')


def main():
	# import profile
	# profile.run('learning()', sort=1)
	learning()


if __name__ == '__main__':
	main()
