from player import Player
from vnet import Network
from gen.gen import gen
from tools import version_str
from vnet_train import train
from match import contest
from tools import log, get_time

bestVersion = 0
startVersion = 1


def init():
	if not bestVersion:
		_Net = Network('vnet' + version_str(0, 3))
		_Net.save()


def better(currentVersion):
	global bestVersion
	log('\n**************************')
	log('better: %d %d' % (bestVersion, currentVersion))
	log(get_time())

	bestNet = Network('vnet' + version_str(bestVersion, 3), bn_training=False)
	bestNet.restore()
	bestPlayer = Player(bestNet)

	trainPath = 'gen/train' + version_str(currentVersion, 3) + '.pkl'
	testPath = 'gen/test' + version_str(currentVersion, 3) + '.pkl'
	gen(bestPlayer, bestPlayer, 2048, trainPath)
	gen(bestPlayer, bestPlayer, 128, testPath)
	log('gen over.' + '   ' + get_time())

	currentNet = Network('vnet' + version_str(currentVersion, 3), bn_training=True)
	train(currentNet, trainPath, testPath)
	currentPlayer = Player(currentNet)
	log('train over.' + '   ' + get_time())

	pro = contest(currentPlayer, bestPlayer, 200)
	log('contest: %f' % pro)
	log(get_time())
	if pro > 0.5:
		bestVersion = currentVersion


def main():
	init()
	for currentPlayer in range(startVersion, 1000000):
		better(currentPlayer)


if __name__ == '__main__':
	main()
