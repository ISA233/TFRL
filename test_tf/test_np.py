import numpy as np
import random

def choice():
	dlist = []
	probability = []
	sum_p = np.log(0)
	print(sum_p)
	# print(sum_p)
	if sum_p < 1e-8:
		return np.random.choice(dlist)
	for i in range(64):
		probability[i] /= sum_p
	return np.random.choice(range(64), p=probability)


# choice()
n = 100
l = [(i, i ** 3) for i in range(n)]
np.random.shuffle(l)
mini_batch_size = 32
for i in range(0, n, mini_batch_size):
	x = i
	y = min(i + mini_batch_size, n)
	print(x, y)
	print(l[i: (i + mini_batch_size)])
