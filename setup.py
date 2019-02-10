from agent import Agent
from vnet import Network
from gen import gen
from tools import version
from vnet_train import train
from match import contest
from tools import log, get_time
from config import config

bestVersion = 6
startVersion = 7


def init():
	if not bestVersion:
		_Net = Network('vnet' + version(0))
		_Net.save()


def rgen(trainPath, testPath):
	bestNet = Network('vnet' + version(bestVersion), bn_training=False)
	bestNet.restore()
	bestAgent = Agent(bestNet)
	gen(bestAgent, config.train_games_size, trainPath)
	gen(bestAgent, config.test_games_size, testPath)


def rtrain(currentVersion, trainPath, testPath):
	currentNet = Network('vnet' + version(currentVersion), bn_training=True)
	train(currentNet, trainPath, testPath)


def rcontest(currentVersion):
	currentNet = Network('vnet' + version(currentVersion), bn_training=False)
	bestNet = Network('vnet' + version(bestVersion), bn_training=False)
	currentNet.restore()
	bestNet.restore()
	currentAgent = Agent(currentNet)
	bestAgent = Agent(bestNet)
	return contest(currentAgent, bestAgent, 100)


def reinforce(currentVersion):
	global bestVersion
	log('\n**************************')
	log('reinforce: %d %d' % (bestVersion, currentVersion))
	log(get_time())
	config.reload()

	trainPath = 'gen/train' + version(currentVersion) + '.pkl'
	testPath = 'gen/test' + version(currentVersion) + '.pkl'
	rgen(trainPath, testPath)
	log('gen over.' + '   ' + get_time())

	rtrain(currentVersion, trainPath, testPath)
	log('train over.' + '   ' + get_time())

	prob = rcontest(currentVersion)
	log('contest: %f' % prob)
	log(get_time())
	if prob >= 0.5:
		bestVersion = currentVersion


def main():
	init()
	for currentVersion in range(startVersion, 1000000):
		reinforce(currentVersion)


if __name__ == '__main__':
	main()
