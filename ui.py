import pygame
from vnet import Network
from agent import Agent
from MCTS import Node, MCT
from chess.chess import ChessBoard, out_board
from config import config

pygame.init()

white = (255, 255, 255)
red = (255, 0, 0)
black = (0, 0, 0)

background = (201, 202, 187)
checkerboard = (80, 80, 80)
button = (52, 53, 44)

v = 40
b = 20
c = 16

FPS = 120
clock = pygame.time.Clock()


def draw_chessman(screen, x, y, player):
	pos = (int(y * v + v // 2 + b), int(x * v + v // 2 + b))
	if player == -1:
		pygame.draw.circle(screen, black, pos, c)
	if player == 1:
		pygame.draw.circle(screen, white, pos, c)


def draw_chessboard(screen, board):
	board = board.board_array()
	for x in range(8):
		for y in range(8):
			if board[x, y]:
				draw_chessman(screen, x, y, board[x, y])


def draw_ai_button(screen, use_ai):
	font = pygame.font.Font(None, 20)
	text = font.render('AI:', True, red)
	screen.blit(text, (40 * 8 + int(b * 1.5), 40 * 8 + b))
	if use_ai:
		text = font.render('ON', True, red)
	else:
		text = font.render('OFF', True, black)
	screen.blit(text, (40 * 8 + int(b * 2.5), 40 * 8 + b))


def draw(screen, board, human_player, use_ai):
	screen.fill(background)
	for i in range(9):
		pygame.draw.line(screen, checkerboard, (v * i + b, b), (v * i + b, v * 8 + b))
		pygame.draw.line(screen, checkerboard, (b, v * i + b), (v * 8 + b, v * i + b))
	draw_chessboard(screen, board)
	draw_chessman(screen, 3, 8.25, human_player)
	draw_ai_button(screen, use_ai)
	pygame.display.update()


def simulate(board, current_player, root, agent):
	mct = MCT(board, current_player, root)
	for i in range(15):
		agent.simulate(mct)


def run(screen, agent):
	board = ChessBoard()
	root = Node()
	current_player = -1
	human_player = -1
	use_ai = True
	draw(screen, board, human_player, use_ai)

	while True:
		clock.tick(FPS)
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				return
		draw(screen, board, human_player, use_ai)
		if not board.could_drop_by(current_player):
			root.move_root(64)
			current_player = -current_player
			continue
		if current_player == human_player or not use_ai:
			_y, _x = pygame.mouse.get_pos()
			if pygame.mouse.get_pressed()[0]:
				x = (_x - b) / v
				y = (_y - b) / v
				if 3 < x < 4 and 8.25 < y < 9.25:
					human_player = -human_player
					continue
				if 7.9 < x and 8.15 < y:
					use_ai = not use_ai
					continue
				x = (_x - b) // v
				y = (_y - b) // v
				if out_board(x, y) or not board.could_drop_xy(x, y, current_player):
					continue
				board.move_xy(x, y, current_player)
				root.move_root(x * 8 + y)
				current_player = -current_player
			simulate(board, current_player, root, agent)
		else:
			print('vhead:', agent.net.vhead(board, current_player))
			action = agent.play(board, current_player, root=root)
			print('MCTS: %d %.5f' % (root.N, root.Q))
			board.move(action, current_player)
			root.move_root(action)
			current_player = -current_player
			board.out()


def main():
	config.simulate_cnt = 700
	net = Network('vnet008_11_2f', bn_training=False, use_GPU=True)
	net.restore()
	agent = Agent(net)
	screen = pygame.display.set_mode([40 * 9 + b * 2, 40 * 8 + b * 2])
	run(screen, agent)


if __name__ == '__main__':
	main()
