from agent import Agent
from vnet import Network
from gen import gen
from tools import version_str
from vnet_train import train
from match import contest
from tools import log, get_time
from config import config

bestVersion = 2
startVersion = 3


def init():
	if not bestVersion:
		_Net = Network('vnet' + version_str(0, 3))
		_Net.save()


def reinforce(currentVersion):
	global bestVersion
	log('\n**************************')
	log('reinforce: %d %d' % (bestVersion, currentVersion))
	log(get_time())

	bestNet = Network('vnet' + version_str(bestVersion, 3), bn_training=False)
	bestNet.restore()
	bestAgent = Agent(bestNet)

	config.reload()

	trainPath = 'gen/train' + version_str(currentVersion, 3) + '.pkl'
	testPath = 'gen/test' + version_str(currentVersion, 3) + '.pkl'
	gen(bestAgent, bestAgent, config.train_games_size, trainPath)
	gen(bestAgent, bestAgent, config.test_games_size, testPath)
	log('gen over.' + '   ' + get_time())

	currentNet = Network('vnet' + version_str(currentVersion, 3), bn_training=True)
	train(currentNet, trainPath, testPath)
	currentAgent = Agent(currentNet)
	log('train over.' + '   ' + get_time())

	pro = contest(currentAgent, bestAgent, 100)
	log('contest: %f' % pro)
	log(get_time())
	if pro > 0.5:
		bestVersion = currentVersion


def main():
	init()
	for currentVersion in range(startVersion, 1000000):
		reinforce(currentVersion)


if __name__ == '__main__':
	main()
