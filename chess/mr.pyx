# cython: boundscheck=False, wraparound=False, initializedcheck=False

cimport cython
import numpy as np

cdef int n = 8

cpdef int[:, :] rotate_int(int[:, :] board, int k):
	if not k:
		return board.copy()
	cdef int[:, :] ret = np.zeros([n, n], dtype=int)
	cdef int i, j, l, x, y
	for i in range(n):
		for j in range(n):
			x, y = i, j
			for l in range(k):
				x, y = y, n - 1 - x
			ret[x, y] = board[i, j]
	return ret

cpdef double[:, :] rotate_double(double[:, :] board, int k):
	if not k:
		return board.copy()
	cdef double[:, :] ret = np.zeros([n, n])
	cdef int i, j, l, x, y
	for i in range(n):
		for j in range(n):
			x, y = i, j
			for l in range(k):
				x, y = y, n - 1 - x
			ret[x, y] = board[i, j]
	return ret

cpdef int[:, :] mirror_int(int[:, :] board, int k):
	if not k:
		return board.copy()
	cdef int[:, :] ret = np.zeros([n, n], dtype=int)
	cdef int i, j, l, x, y
	for i in range(n):
		for j in range(n):
			x, y = i, n - 1 - j
			ret[x, y] = board[i, j]
	return ret

cpdef double[:, :] mirror_double(double[:, :] board, int k):
	if not k:
		return board.copy()
	cdef double[:, :] ret = np.zeros([n, n])
	cdef int i, j, l, x, y
	for i in range(n):
		for j in range(n):
			x, y = i, n - 1 - j
			ret[x, y] = board[i, j]
	return ret
