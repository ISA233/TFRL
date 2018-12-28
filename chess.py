import numpy as np


def init_board():
	board = np.zeros([8, 8])
	board[3, 3] = 1
	board[4, 4] = 1
	board[3, 4] = -1
	board[4, 3] = -1
	return board


def out_board(x, y):
	if x >= 8 or x < 0 or y >= 8 or y < 0:
		return 1
	return 0


class ChessBoard:
	def __init__(self, board=init_board()):
		self.board = np.array(board, dtype='int')
		self.xx = [0, 1, 1, 1, 0, -1, -1, -1]
		self.yy = [-1, -1, 0, 1, 1, 1, 0, -1]

	def to_network_input(self, player):
		return np.append(self.board.reshape(64), player)

	def evaluate(self):
		v0, v1 = 0, 0
		for p in self.board.flat:
			if p == -1:
				v0 += 1
			if p == 1:
				v1 += 1
		# print('value:', v0, v1)
		return v1 - v0

	def is_finish(self):
		if not self.have_space():
			return 1
		for p in range(64):
			if self.could_drop(p, -1) or self.could_drop(p, 1):
				return 0
		return 1

	def check_reverse(self, x, y, direct, player):
		x += direct[0]
		y += direct[1]
		if out_board(x, y):
			return 0
		if self.board[x, y] == 0 or self.board[x, y] == player:
			return 0
		while True:
			x += direct[0]
			y += direct[1]
			if out_board(x, y) or self.board[x, y] == 0:
				break
			if self.board[x, y] == player:
				return 1
		return 0

	def could_drop_xy(self, x, y, player):
		if out_board(x, y) or self.board[x, y]:
			return 0
		for direct in zip(self.xx, self.yy):
			# print(direct)
			if self.check_reverse(x, y, direct, player):
				return 1
		return 0

	def could_drop(self, p, player):
		x, y = p // 8, p % 8
		return self.could_drop_xy(x, y, player)

	def drop_list(self, player):
		dlist = []
		for p in range(64):
			if self.could_drop(p, player):
				dlist.append(p)
		return dlist

	def drop_list_xy(self, player):
		dlist = self.drop_list(player)
		dlist_xy = []
		for p in dlist:
			dlist_xy.append((p // 8, p % 8))
		return dlist_xy

	def move_xy(self, x, y, player):
		if not self.could_drop_xy(x, y, player):
			return 0
		self.board[x, y] = player
		original_x, original_y = x, y
		for direct in zip(self.xx, self.yy):
			x, y = original_x, original_y
			if self.check_reverse(x, y, direct, player):
				while True:
					x += direct[0]
					y += direct[1]
					if self.board[x, y] == player:
						break
					self.board[x, y] = player
		return 1

	def move(self, p, player):
		x, y = p // 8, p % 8
		return self.move_xy(x, y, player)

	def out(self):
		for i in range(8):
			for j in range(8):
				if self.board[i][j] == -1:
					print('x', end=' ')
				elif self.board[i][j] == 1:
					print('o', end=' ')
				else:
					print('_', end=' ')
			print()

	def have_space(self):
		return 0 in self.board


def main():
	board = np.array([[0, 1, 1, -1, 0, 0, 0, 0],
	                  [0, 0, 1, 1, -1, 0, 0, 0],
	                  [0, 1, 1, 0, 0, 0, 0, 0],
	                  [0, 1, 0, 1, 1, 0, 0, 0],
	                  [0, 1, 0, 1, -1, 0, 0, 0],
	                  [0, -1, 0, 0, 0, 0, 0, 0],
	                  [0, 0, 0, 0, 0, 0, 0, 0],
	                  [0, 0, 0, 0, 0, 0, 0, 0]])
	board = ChessBoard(board)
	board.out()
	# print(board.evaluate())
	# print(board.to_network_input())
	# print(board.could_drop_xy(1, 1, -1))
	print(board.drop_list_xy(-1))
	board.move_xy(1, 1, -1)
	board.out()


if __name__ == '__main__':
	main()

# https://www.jianshu.com/p/713f0bd8de7b?from=timeline
# https://www.jianshu.com/p/e2f62043d02b
# https://icml.cc/2016/tutorials/deep_rl_tutorial.pdf
# https://blog.csdn.net/juanjuan1314/article/details/78048065
# https://blog.csdn.net/songrotek/article/details/51065143
# http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/mnist_beginners.html
# https://www.jianshu.com/p/2ccbab48414b
