import numpy as np

init_board = np.zeros([8, 8])

n = 8
xx = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=int)
yy = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=int)


def out_board(x, y):
	pass


class ChessBoard:
	def __init__(self, _board=init_board):
		self.board = _board.copy()

	def to_network_input(self, player):
		pass

	def evaluate(self):
		pass

	def win(self, player):
		pass

	def is_finish(self):
		pass

	def check_reverse(self, x, y, direct, player):
		pass

	def could_drop_xy(self, x, y, player):
		pass

	def could_drop(self, p, player):
		pass

	def could_drop_by(self, player):
		pass

	def drop_list(self, player):
		pass

	def drop_list_xy(self, player):
		pass

	def move_xy(self, x, y, player):
		pass

	def move(self, p, player):
		pass

	def out(self):
		pass

	def have_space(self):
		pass

	def clone(self):
		pass

	def board_array(self):
		pass


def main():
	print(init_board)
	A = ChessBoard()
	print(A.board)
	print(init_board)


if __name__ == '__main__':
	main()

# https://www.jianshu.com/p/713f0bd8de7b?from=timeline
# https://www.jianshu.com/p/e2f62043d02b
# https://icml.cc/2016/tutorials/deep_rl_tutorial.pdf
# https://blog.csdn.net/juanjuan1314/article/details/78048065
# http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/mnist_beginners.html
# http://wiki.jikexueyuan.com/project/tensorflow-zh/tutorials/mnist_beginners.html
# https://www.jianshu.com/p/2ccbab48414b
