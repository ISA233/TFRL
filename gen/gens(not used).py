def gen_ones(player0, player1):
	boards_isfinish = [[ChessBoard(), 0] for i in range(100)]
	current_player = 1
	states = []
	for p in range(60):
		current_player = -current_player
		could_move_boards = []
		for board, isfinish in boards_isfinish:
			if isfinish:
				continue
			if not board.could_drop_by(current_player):
				states.append([board.board_array(), current_player, -1, 0.5])
				continue
			could_move_boards.append(board)
		if current_player == -1:
			moves = player0.plays(could_move_boards, current_player)
		else:
			moves = player1.plays(could_move_boards, current_player)
		for board, move in zip(could_move_boards, moves):
			states.append([board.board_array(), current_player, move, 0.5])
			board.move(move, current_player)
		if board.is_finish():
			states.append([board.board_array(), current_player, -1, 0.5])
			value = board.evaluate()
			for state in states:
				# state[0].out()
				if value > 0:
					state[3] = 1.0 if state[1] == 1 else 0.0
				if value < 0:
					state[3] = 0.0 if state[1] == 1 else 1.0
			# print(state[1:])
			# print(value)
			break
	return states