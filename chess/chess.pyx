# cython: boundscheck=False, wraparound=False, initializedcheck=False

cimport cython
import numpy as np

cdef extern from 'stdio.h':
	extern int printf(const char * format, ...)

cdef int[:, :] init_board = np.zeros([8, 8], dtype=int)
init_board[3, 3] = 1
init_board[4, 4] = 1
init_board[3, 4] = -1
init_board[4, 3] = -1

cdef int n = 8
cdef int[:] xx = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=int)
cdef int[:] yy = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=int)


cpdef inline int out_board(int x, int y):
	if x >= 8 or x < 0 or y >= 8 or y < 0:
		return 1
	return 0


cdef class ChessBoard:
	cdef int[:, :] board
	def __init__(self, int[:, :] _board=init_board):
		self.board = _board.copy()

	cpdef int[:, :, :] to_network_input(self, int player):
		cdef int[:, :, :] network_input = np.zeros([n, n, 2], dtype=int)
		cdef int x, y
		for x in range(n):
			for y in range(n):
				network_input[x, y, 0] = self.board[x, y]
				network_input[x, y, 1] = player
		return network_input

	cpdef int evaluate(self):
		cdef int v0 = 0
		cdef int v1 = 0
		cdef int i, j
		for i in range(n):
			for j in range(n):
				if self.board[i, j] == -1:
					v0 += 1
				if self.board[i, j] == 1:
					v1 += 1
		return v1 - v0

	cpdef int win(self, int player):
		cdef int v = self.evaluate()
		if v * player > 0:
			return 1
		if v * player < 0:
			return -1
		return 0

	cpdef int is_finish(self):
		if not self.have_space():
			return 1
		if self.could_drop_by(1) or self.could_drop_by(-1):
			return 0
		return 1

	cdef int check_reverse(self, int x, int y, int directx, int directy, int player):
		x += directx
		y += directy
		if out_board(x, y):
			return 0
		if self.board[x, y] == 0 or self.board[x, y] == player:
			return 0
		while True:
			x += directx
			y += directy
			if out_board(x, y) or self.board[x, y] == 0:
				break
			if self.board[x, y] == player:
				return 1
		return 0

	cpdef int could_drop_xy(self, int x, int y, int player):
		if out_board(x, y) or self.board[x, y]:
			return 0
		cdef int i
		cdef int directx, directy
		for i in range(8):
			directx, directy = xx[i], yy[i]
			if self.check_reverse(x, y, directx, directy, player):
				return 1
		return 0

	cpdef int could_drop(self, int p, int player):
		cdef int x, y
		x, y = p // n, p % n
		return self.could_drop_xy(x, y, player)

	cpdef int could_drop_by(self, int player):
		cdef int p
		for p in range(n * n):
			if self.could_drop(p, player):
				return 1
		return 0

	def drop_list(self, player):
		dlist = []
		for p in range(64):
			if self.could_drop(p, player):
				dlist.append(p)
		return dlist if dlist else [64]

	cpdef int move_xy(self, int x, int y, int player):
		if not self.could_drop_xy(x, y, player):
			return 0
		self.board[x, y] = player
		cdef int original_x, original_y
		cdef int directx, directy
		original_x, original_y = x, y
		cdef int i
		for i in range(8):
			directx, directy = xx[i], yy[i]
			x, y = original_x, original_y
			if self.check_reverse(x, y, directx, directy, player):
				while True:
					x += directx
					y += directy
					if self.board[x, y] == player:
						break
					self.board[x, y] = player
		return 1

	cpdef int move(self, int p, int player):
		cdef int x, y
		x, y = p // 8, p % 8
		return self.move_xy(x, y, player)

	cpdef void out(self):
		cpdef int i, j
		for i in range(n):
			for j in range(n):
				if self.board[i][j] == -1:
					printf('x ')
				elif self.board[i][j] == 1:
					printf('o ')
				else:
					printf('_ ')
			printf('\n')

	cpdef int have_space(self):
		cdef int i, j
		for i in range(n):
			for j in range(n):
				if self.board[i, j] == 0:
					return 1
		return 0

	cpdef ChessBoard clone(self):
		return ChessBoard(self.board)

	def board_array(self):
		return np.array(self.board.copy())

