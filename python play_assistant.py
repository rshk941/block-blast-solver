import numpy as np
import torch

from complete_blockblast2_6 import (
    blockblast,
    pieces_shapes,
    value_net,
    device,
    iterate_best_sequence,
    print_board,
    place_piece,
    clear_lines,
    get_state,
)

state_dict = torch.load("value_net.pt", map_location=device)
saved_input_dim = state_dict["layer1.weight"].shape[1]
_probe_game = blockblast()
current_input_dim = len(get_state(_probe_game))
value_net.load_state_dict(state_dict)
value_net.eval()


def print_available_pieces():
    print("Available piece names:")
    for name, shape in pieces_shapes.items():
        print(f"\n{name}:")
        for row in shape:
            print("  " + "".join("#" if c else "." for c in row))


def print_board_with_highlight(board, piece_array, row, col):
    piece_rows, piece_cols = piece_array.shape
    highlight_cells = set()
    for i in range(piece_rows):
        for j in range(piece_cols):
            if piece_array[i][j] == 1:
                highlight_cells.add((row + i, col + j))

    print("   " + " ".join(str(c) for c in range(8)))
    for r in range(8):
        row_chars = []
        for c in range(8):
            if (r, c) in highlight_cells:
                row_chars.append("X")
            elif board[r][c] == 1:
                row_chars.append("#")
            else:
                row_chars.append(".")
        print(f"{r}: {' '.join(row_chars)}")


def input_board():
    print("Enter your board, row by row (8 rows), using 0 for empty and 1 for filled.")
    print("Example row: 00110000")
    board = []
    for r in range(8):
        while True:
            row_str = input(f"Row {r}: ").strip()
            if len(row_str) == 8 and all(c in "01" for c in row_str):
                board.append([int(c) for c in row_str])
                break
            print("Invalid row, must be exactly 8 characters of 0/1.")
    return np.array(board, dtype=int)


def input_pieces():
    print_available_pieces()
    piece_names = []
    for i in range(3):
        while True:
            name = input(f"\nPiece {i+1} name: ").strip()
            if name in pieces_shapes:
                piece_names.append(name)
                break
            print(f"'{name}' not recognized, check spelling against the list above.")
    return piece_names


def input_streak():
    while True:
        s = input("Current streak (0 if none): ").strip()
        if s.isdigit():
            return int(s)
        print("Enter a non-negative integer.")


def apply_sequence(game, order):
    exactly like the real game's turn()/calculate_score() would."""
    piece_lines = []
    board_clear_count = 0
    current_streak = game.streak

    for piece_name, (row, col) in order:
        piece_array = pieces_shapes[piece_name]
        game.board = place_piece(piece_array, game.board, row, col)
        game.board, lines_cleared = clear_lines(game.board)
        piece_lines.append((piece_name, lines_cleared))
        if not np.any(game.board):
            board_clear_count += 1

    game.calculate_score(current_streak, piece_lines, board_clear_count)


def snapshot(game):
    return {
        "board": game.board.copy(),
        "score": game.score,
        "streak": game.streak,
    }


def restore(game, snap):
    game.board = snap["board"].copy()
    game.score = snap["score"]
    game.streak = snap["streak"]


def recommend_move(game):
    piece_names = input_pieces()
    game.current_pieces = piece_names

    print("\nCurrent board:")
    print_board(game.board)

    best_so_far, best_perm = iterate_best_sequence(game, value_net)

    if best_so_far["current_best"] == float("-inf"):
        print("\nNo valid sequence found for these 3 pieces on this board — likely game over.")
        return False, None

    placements = [best_so_far["placement1"], best_so_far["placement2"], best_so_far["placement3"]]
    order = [(piece_names[idx], pos) for idx, pos in zip(best_perm, placements)]

    print("\nRecommended move order (X = where to place this piece):")
    working_board = game.board.copy()
    for step_num, (piece_name, (row, col)) in enumerate(order, start=1):
        piece_array = pieces_shapes[piece_name]
        print(f"\n--- Step {step_num}: place '{piece_name}' at row={row}, col={col} ---")
        print_board_with_highlight(working_board, piece_array, row, col)
        working_board = place_piece(piece_array, working_board, row, col)
        working_board, _ = clear_lines(working_board)

    before_state = snapshot(game)
    apply_sequence(game, order)
    print(f"\nBoard updated. Current score: {game.score}, streak: {game.streak}")

    if game.is_game_over():
        print("\n*** Game over — no piece in the next tray will have a valid placement. ***")

    return True, before_state


if __name__ == "__main__":
    game = blockblast()
    game.board = input_board()
    game.streak = input_streak()

    history = []

    while True:
        ok, before_state = recommend_move(game)
        if not ok:
            break
        if before_state is not None:
            history.append(before_state)

        while True:
            choice = input("\n(n)ext tray, (u)ndo last move, (c)orrect board, (q)uit: ").strip().lower()
            if choice == "n":
                break
            elif choice == "u":
                if history:
                    restore(game, history.pop())
                    print("\nUndone. Board restored to:")
                    print_board(game.board)
                else:
                    print("Nothing to undo.")
            elif choice == "c":
                print("\nRe-enter the board as it actually looks right now.")
                game.board = input_board()
                new_streak = input("Current streak (leave blank to keep current value): ").strip()
                if new_streak.isdigit():
                    game.streak = int(new_streak)
                print("\nBoard corrected to:")
                print_board(game.board)
            elif choice == "q":
                exit()
            else:
                print("Type 'n', 'u', 'c', or 'q'.")
