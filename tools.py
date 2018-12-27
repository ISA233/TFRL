import numpy as np


def player_01(player):
	return (player + 1) // 2


def to_vector(place):
	v = np.zeros([9])
	v[place] = 1
	return v


def main():
	print(to_vector(4))


if __name__ == '__main__':
	main()
