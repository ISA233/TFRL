import numpy as np


def check_same(lis):
	for li in lis:
		if li != lis[0]:
			return 0
	return 1


class ChessBoard:
	def __init__(self, board=np.zeros([3, 3])):
		self.board = np.array(board, dtype='int')

	def to_network_input(self, player=-1):
		return np.append(self.board.reshape(9), player)

	def is_finish(self):
		for i in range(3):
			if self.board[i, 0] and check_same(self.board[i, :]):
				return self.board[i, 0]
			if self.board[0, i] and check_same(self.board[:, i]):
				return self.board[0, i]
		if self.board[1, 1] != 0:
			if self.board[0, 0] == self.board[1, 1] and self.board[1, 1] == self.board[2, 2]:
				return self.board[1, 1]
			if self.board[0, 2] == self.board[1, 1] and self.board[1, 1] == self.board[2, 0]:
				return self.board[1, 1]
		return 0

	def move(self, p, player):
		x, y = p // 3, p % 3
		if self.board[x, y]:
			return -player
		self.board[x, y] = player
		return self.is_finish()

	def out(self):
		for i in range(3):
			for j in range(3):
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
	board = np.array([[0, 0, 1],
	                  [0, -1, 1],
	                  [0, 0, 1]])
	board = ChessBoard(board)
	board.out()
	print(board.to_network_input())


if __name__ == '__main__':
	main()

# https://www.jianshu.com/p/713f0bd8de7b?from=timeline
# https://www.jianshu.com/p/e2f62043d02b
# https://icml.cc/2016/tutorials/deep_rl_tutorial.pdf
# https://blog.csdn.net/juanjuan1314/article/details/78048065
# https://blog.csdn.net/songrotek/article/details/51065143
# http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/mnist_beginners.html
# https://www.jianshu.com/p/2ccbab48414b
