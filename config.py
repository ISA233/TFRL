import time


class Config:
	def __init__(self):
		self.momentum = 0
		self.learning_rate = 0
		self.learning_rate2 = 0
		self.save_interval = 0
		self.l2_regularizer_alpha = 0
		self.simulate_cnt = 0
		self.gen_simulate_cnt = 0
		self.bad_net_limit = 0
		self.test_games_size = 0
		self.train_games_size = 0
		self.gen_batch_size = 0
		self.reload()

	def reload(self):
		with open('config.txt', 'r') as f:
			configs = f.read().replace(' ', '').split('\n')
		config_mp = dict()
		for _config in configs:
			if not _config.count('='):
				continue
			key, value = _config.split('=')
			if value.count('.'):
				config_mp[key] = float(value)
			else:
				config_mp[key] = int(value)
		self.momentum = config_mp['momentum']
		self.learning_rate = config_mp['learning_rate']
		self.learning_rate2 = config_mp['learning_rate2']
		self.save_interval = config_mp['save_interval']
		self.l2_regularizer_alpha = config_mp['l2_regularizer_alpha']
		self.simulate_cnt = config_mp['simulate_cnt']
		self.gen_simulate_cnt = config_mp['gen_simulate_cnt']
		self.bad_net_limit = config_mp['bad_net_limit']
		self.test_games_size = config_mp['test_games_size']
		self.train_games_size = config_mp['train_games_size']
		self.gen_batch_size = config_mp['gen_batch_size']
		self.print()

	def print(self):
		print('===============')
		print('### CONFIG:')
		print('momentum:', self.momentum)
		print('learning_rate:', self.learning_rate)
		print('learning_rate2:', self.learning_rate2)
		print('save_interval:', self.save_interval)
		print('l2_regularizer_alpha:', self.l2_regularizer_alpha)
		print('simulate_cnt:', self.simulate_cnt)
		print('gen_simulate_cnt:', self.gen_simulate_cnt)
		print('bad_net_limit:', self.bad_net_limit)
		print('test_games_size:', self.test_games_size)
		print('train_games_size:', self.train_games_size)
		print('gen_batch_size:', self.gen_batch_size)
		print('===============')


config = Config()


def main():
	while True:
		config.reload()
		config.print()
		time.sleep(3.0)


if __name__ == '__main__':
	main()
