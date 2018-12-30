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

choice()