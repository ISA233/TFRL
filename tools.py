import numpy as np


def player_01(player):
	if player == 1:
		return 1
	return 0


def to_vector(place, size=64):
	v = np.zeros([size])
	v[place] = 1
	return v


def porn(x):
	if x > 0:
		return 1
	if x < 0:
		return -1
	return 0


def main():
	print(to_vector(4))


if __name__ == '__main__':
	main()
