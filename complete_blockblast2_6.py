import numpy as np
import random
from collections import namedtuple, deque
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import itertools
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

num_episodes = 800
TAU = 0.005
gamma = 0.99
batch_size = 64
MEMORY_CAPACITY = 100000
LR = 3e-4 #learning rate
game_over_penalty = 500
reward_scale = 1/100
EPS_START, EPS_END, EPS_DECAY = 0.3, 0.02, 2000
streak_cap = 20

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
print("Using device:", device)

pieces_shapes = {"2x3":np.array([[1,1,1],[1,1,1]]),
                     "3x2":np.array([[1,1],[1,1],[1,1]]),
                     "L_TL":np.array([[1,1,1],[1,0,0],[1,0,0]]),
                     "L_TR":np.array([[1,1,1],[0,0,1],[0,0,1]]),
                     "L_BL":np.array([[1,0,0],[1,0,0],[1,1,1]]),
                     "L_BR":np.array([[0,0,1],[0,0,1],[1,1,1]]),
                     "4x1":np.array([[1],[1],[1],[1]]),
                     "1x4":np.array([[1,1,1,1]]),
                     "SL_TL1":np.array([[1,1],[1,0],[1,0]]),
                     "SL_TL2":np.array([[1,1,1],[1,0,0]]),
                     "SL_TR1":np.array([[1,1],[0,1],[0,1]]),
                     "SL_TR2":np.array([[1,1,1],[0,0,1]]),
                     "SL_BL1":np.array([[1,0],[1,0],[1,1]]),
                     "SL_BL2":np.array([[1,0,0],[1,1,1]]),
                     "SL_BR1":np.array([[0,1],[0,1],[1,1]]),
                     "SL_BR2":np.array([[0,0,1],[1,1,1]]),
                     "5x1":np.array([[1],[1],[1],[1],[1]]),
                     "1x5":np.array([[1,1,1,1,1]]),
                     "z1":np.array([[0,1,1],[1,1,0]]),
                     "z2":np.array([[1,0],[1,1],[0,1]]),
                     "z3":np.array([[0,1],[1,1],[1,0]]),
                     "z4":np.array([[1,1,0],[0,1,1]]),
                     "t1":np.array([[0,1,0],[1,1,1]]),
                     "t2":np.array([[1,0],[1,1],[1,0]]),
                     "t3":np.array([[0,1],[1,1],[0,1]]),
                     "t4":np.array([[1,1,1],[0,1,0]]),
                     "tinyL1":np.array([[0,1],[1,1]]),
                     "tinyL2":np.array([[1,0],[1,1]]),
                     "tinyL3":np.array([[1,1],[0,1]]),
                     "tinyL4":np.array([[1,1],[1,0]]),
                     "3x3":np.array([[1,1,1],[1,1,1],[1,1,1]]),
                     "3x1":np.array([[1],[1],[1]]),
                     "1x3":np.array([[1,1,1]]),
                     "2x1":np.array([[1],[1]]),
                     "1x2":np.array([[1,1]]),
                     "2x2":np.array([[1,1],[1,1]]),
                     "staircase1":np.array([[0,0,1],[0,1,0],[1,0,0]]),
                     "staircase2":np.array([[1,0,0],[0,1,0],[0,0,1]])}
weights = [10,10,2,2,2,2,6,6,2,2,2,2,2,2,2,2,10,10,1,1,1,1,1.5,1.5,1.5,1.5,0.1,0.1,0.1,0.1,7,0.1,0.1,0.5,0.5,9,0.1,0.1]
pieces = list(pieces_shapes.keys())

def variants (board2d):
    variants = []
    for i in range (4):
        variants.append (np.rot90(board2d,i))
        variants.append(np.fliplr(np.rot90(board2d,i)))
    return variants

def normalized_streak (streak):
    return min(streak, streak_cap)/streak_cap

def has_valid_placement(board, piece):
    board_rows, board_cols = board.shape
    piece_rows, piece_cols = piece.shape
    for row in range(board_rows - piece_rows + 1):
        for col in range(board_cols - piece_cols + 1):
            fits = True
            for i in range(piece_rows):
                for j in range(piece_cols):
                    if piece[i][j] == 1 and board[row + i][col + j] == 1:
                        fits = False
                        break
                if not fits:
                    break
            if fits:
                return True
    return False

def placeability(board):
    total_w, fits_w = sum(weights), 0.0
    for name, w in zip(pieces, weights):
        if has_valid_placement(board, pieces_shapes[name]):
            fits_w += w
    return fits_w / total_w

def get_epsilon(episode_i):
    return EPS_END + (EPS_START - EPS_END) * math.exp(-episode_i / EPS_DECAY)

def piece_squares (piece_name):
    return np.count_nonzero(pieces_shapes[piece_name])

def get_valid_placements(board, piece):
    board_rows, board_cols = board.shape
    piece_rows, piece_cols = piece.shape
    valid_positions = []
    for row in range(board_rows - piece_rows + 1):
        for col in range(board_cols - piece_cols + 1):
            fits = True
            for i in range(piece_rows):
                for j in range(piece_cols):
                    if piece[i][j] == 1:
                        if board[row + i][col + j] == 1:
                            fits = False
                            break
                if not fits:
                    break
            if fits:
                valid_positions.append((row, col))
    return valid_positions


def place_piece(piece, board, row, col):
    piece_rows, piece_cols = piece.shape
    board_copy = board.copy()
    for i in range(piece_rows):
        for j in range(piece_cols):
            if piece[i][j] == 1:
                board_copy[row + i][col + j] = 1
    return board_copy


def clear_lines(board):
    board_copy = board.copy()
    board_rows, board_cols = board_copy.shape
    rows_to_clear = []
    cols_to_clear = []
    lines_cleared = [0, 0]

    for row in range(board_rows):
        if all(board_copy[row][i] == 1 for i in range(board_cols)):
            rows_to_clear.append(row)
            lines_cleared[0] += 1

    for col in range(board_cols):
        if all(board_copy[j][col] == 1 for j in range(board_rows)):
            cols_to_clear.append(col)
            lines_cleared[1] += 1

    for row in rows_to_clear:
        for i in range(board_cols):
            board_copy[row][i] = 0
    for j in range(board_rows):
        for col in cols_to_clear:
            board_copy[j][col] = 0

    return board_copy, lines_cleared


class blockblast():
    def __init__(self):
        self.board = np.zeros((8, 8), dtype=int)
        self.current_pieces = []
        self.score = 0
        self.streak = 0
        self.tray_lines_cleared = []
        self.board_clear_count = 0
        self.reset()

    def reset(self):
        self.current_pieces = []
        self.score = 0
        self.streak = 0
        self.board = np.zeros((8, 8), dtype=int)
        self.board_clear_count = 0
        self.tray_lines_cleared = []
        self.refill_tray()

    def is_game_over(self):
        game_over = True
        for piece_name in self.current_pieces:
            if piece_name is None:
                continue
            piece = pieces_shapes[piece_name]
            if get_valid_placements(self.board, piece):
                game_over = False
                break
        return game_over

    def refill_tray(self):
        self.current_pieces = random.choices(pieces, weights=weights, k=3)

    def calculate_score(self, current_streak, piece_lines, board_clear_count):
        grand_total = sum(sum(lc) for _, lc in piece_lines)

        if grand_total == 0:
            streak = 0
            total_delta = sum(piece_squares(name) for name, _ in piece_lines)
            self.streak = streak
            self.score += total_delta
            return total_delta

        streak = current_streak
        total_delta = 0
        for piece_name, lines_cleared in piece_lines:
            total_lines = sum(lines_cleared)
            if total_lines == 0:
                total_delta += piece_squares(piece_name)
            else:
                streak += total_lines
                total_delta += streak * piece_squares(piece_name) * total_lines * 5
                
        total_delta += board_clear_count * 300
        
        self.streak = streak
        self.score += total_delta
        return total_delta

    def turn(self, piece_index, row, col):
        reward = 0
        piece_name = self.current_pieces[piece_index]
        if piece_name is None:
            return {"valid": False}
        piece = pieces_shapes[piece_name]
        valid_positions = get_valid_placements(self.board, piece)
        if (row, col) not in valid_positions:
            return {"valid": False}
        self.board = place_piece(piece, self.board, row, col)
        self.board, lines_cleared = clear_lines(self.board)
        self.tray_lines_cleared.append((piece_name, lines_cleared))
        self.current_pieces[piece_index] = None
        if not np.any(self.board):
            self.board_clear_count += 1
        if all(p is None for p in self.current_pieces):
            reward = self.calculate_score(self.streak, self.tray_lines_cleared, self.board_clear_count)
            self.refill_tray()
            self.tray_lines_cleared = []
            self.board_clear_count = 0
        game_over = self.is_game_over()
        return {"valid": True, "lines_cleared": lines_cleared, "game_over": game_over, "reward": reward}

def count_holes(board):
    rows, cols = board.shape
    holes = 0

    def is_blocked(r, c):
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return True  
        return board[r][c] == 1

    for r in range(rows):
        for c in range(cols):
            if board[r][c] == 0:  
                neighbors_blocked = (
                    is_blocked(r - 1, c) and  
                    is_blocked(r + 1, c) and  
                    is_blocked(r, c - 1) and  
                    is_blocked(r, c + 1)      
                )
                if neighbors_blocked:
                    holes += 1

    return holes

def index_to_action(action_index):
    piece_index = action_index // 64
    row = (action_index % 64) // 8
    col = (action_index % 64) % 8
    return piece_index, row, col

def action_to_index(piece_index, row, col):
    return piece_index * 64 + row * 8 + col

def get_action_mask(game):
    action_mask = [False] * 192
    valid_slot_indices = [i for i, x in enumerate(game.current_pieces) if x is not None]
    for piece_index in valid_slot_indices:
        piece_name = game.current_pieces[piece_index]
        piece = pieces_shapes[piece_name]
        valid_positions = get_valid_placements(game.board, piece)
        for row, col in valid_positions:
            action_mask[action_to_index(piece_index, row, col)] = True
    return action_mask


def encode_piece(piece_name):
    encoded = np.zeros(len(pieces_shapes))
    if piece_name is None:
        return encoded
    encoded[pieces.index(piece_name)] = 1
    return encoded

def board_features(board, streak):
    flattened_board = board.flatten()
    streak_arr = np.array([normalized_streak(streak)])
    holes = np.array([count_holes(board)])
    place = np.array([placeability(board)])
    state = np.concatenate([flattened_board, streak_arr, board.sum(axis=1), board.sum(axis=0), holes, place])
    return state
    
def get_state(game):
    return board_features (game.board, game.streak)

def reset(game):
    game.reset()
    return get_state(game), {"action_mask": get_action_mask(game)}

def step(game, action_index):
    piece_index, row, col = index_to_action(action_index)
    turn_output = game.turn(piece_index, row, col)
    terminated = turn_output["game_over"]
    truncated = False
    reward = move_bonus
    if terminated:
        reward -= game_over_penalty
    return (get_state(game), reward + sum(turn_output["lines_cleared"]), terminated, truncated, {"action_mask": get_action_mask(game)})

class ValueNet(nn.Module):
    def __init__(self, n_observations):
        super(ValueNet, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

def get_score(current_streak, piece_lines, board_clear_count):
    new_streak = current_streak
    grand_total = sum(sum(lc) for _, lc in piece_lines)
    if grand_total == 0:
        total_delta = sum(piece_squares(name) for name, _ in piece_lines)
        new_streak = 0
        return total_delta, new_streak

    total_delta = 0
    for piece_name, lines_cleared in piece_lines:
        total_lines = sum(lines_cleared)
        if total_lines == 0:
            total_delta += piece_squares(piece_name)
        else:
            new_streak += total_lines
            total_delta += new_streak * piece_squares(piece_name) * total_lines * 5
            
    total_delta += board_clear_count * 300
    return total_delta, new_streak

def search_best_sequence(board, current_streak, valuenet, piece1_name, piece2_name, piece3_name):
    candidates = []  

    valid_positions1 = get_valid_placements(board, pieces_shapes[piece1_name])
    for position1 in valid_positions1:
        board_copy1 = place_piece(pieces_shapes[piece1_name], board, position1[0], position1[1])
        board_copy1, lines_cleared1 = clear_lines(board_copy1)
        clear_board_count1 = 1 if not np.any(board_copy1) else 0
        pieces_lines1 = (piece1_name, lines_cleared1)

        valid_positions2 = get_valid_placements(board_copy1, pieces_shapes[piece2_name])
        for position2 in valid_positions2:
            board_copy2 = place_piece(pieces_shapes[piece2_name], board_copy1, position2[0], position2[1])
            board_copy2, lines_cleared2 = clear_lines(board_copy2)
            clear_board_count2 = 1 if not np.any(board_copy2) else 0
            pieces_lines2 = (piece2_name, lines_cleared2)

            valid_positions3 = get_valid_placements(board_copy2, pieces_shapes[piece3_name])
            for position3 in valid_positions3:
                board_copy3 = place_piece(pieces_shapes[piece3_name], board_copy2, position3[0], position3[1])
                board_copy3, lines_cleared3 = clear_lines(board_copy3)
                clear_board_count3 = 1 if not np.any(board_copy3) else 0
                pieces_lines3 = (piece3_name, lines_cleared3)

                clear_board_count = clear_board_count1 + clear_board_count2 + clear_board_count3
                pieces_lines = [pieces_lines1, pieces_lines2, pieces_lines3]

                reward, new_streak = get_score(current_streak, pieces_lines, clear_board_count)
                candidates.append((reward, board_copy3, new_streak, position1, position2, position3))

    if not candidates:
        return {"placement1":[0,0], "placement2":[0,0], "placement3":[0,0], "current_best": float('-inf')}

    states = np.stack([board_features(c[1],c[2]) for c in candidates])
    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    with torch.no_grad():
        values = valuenet(states_t).squeeze(1)

    best_idx = None
    best_total = float('-inf')
    for i, c in enumerate(candidates):
        reward = c[0]
        total_reward = reward + gamma * values[i].item() 
        if total_reward > best_total:
            best_total = total_reward
            best_idx = i

    best = candidates[best_idx]
    return {"placement1": best[3], "placement2": best[4], "placement3": best[5], "current_best": best_total}

def iterate_best_sequence(game, valuenet, epsilon=0.0):
    best_so_far = {"placement1":[0,0], "placement2":[0,0], "placement3":[0,0],"current_best":float('-inf')}
    best_perm = []
    results = []
    perms = itertools.permutations([0,1,2])
    for permutation in perms:
        first, second, third = permutation
        best_sequence = search_best_sequence(game.board, game.streak, valuenet, game.current_pieces[first], game.current_pieces[second], game.current_pieces[third])
        if best_sequence["current_best"] > best_so_far["current_best"]:
            best_so_far = best_sequence
            best_perm = permutation
        results.append ((best_sequence, permutation))
    valid_results = [(seq,perm)for seq,perm in results if seq["current_best"]>float('-inf')]
    if valid_results and random.random () < epsilon:
        best_so_far, best_perm = random.choice (valid_results)
        
    return best_so_far, best_perm
Transition = namedtuple('Transition', ('state_before', 'reward', 'state_after'))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def push_with_rotation (memory, board_before_2d, streak_before, reward, board_after_2d, streak_after):
    before_variants = variants (board_before_2d)
    after_variants = (variants (board_after_2d) if board_after_2d is not None else [None]*8)
    for b_before, b_after in zip(before_variants, after_variants):
        state_before = board_features(b_before, streak_before)
        state_before_t = torch.tensor(state_before, dtype=torch.float32, device=device).unsqueeze(0)
        reward_t = torch.tensor([reward], dtype=torch.float32, device=device)

        state_after_t = None
        if b_after is not None:
            state_after = board_features(b_after, streak_after)
            state_after_t = torch.tensor(state_after, dtype=torch.float32, device=device).unsqueeze(0)

        memory.push(state_before_t, reward_t, state_after_t)

def soft_update (value_net, target_value_net, TAU):
    target_value_net_state_dict = target_value_net.state_dict()
    value_net_state_dict = value_net.state_dict() 
    for key in value_net_state_dict:
        target_value_net_state_dict[key] = (value_net_state_dict[key] * TAU
                                       + target_value_net_state_dict[key] * (1 - TAU))
    target_value_net.load_state_dict(target_value_net_state_dict)

def plot_scores(game_scores, show_result=False):
    plt.figure(1)
    scores_t = torch.tensor(game_scores, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(scores_t.numpy())
    if len(scores_t) >= 100:
        means = scores_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def train():
    plt.ion()
    for episode_i in range(num_episodes):
        final_score = play_episode(game, value_net, TAU, episode_i)
        game_scores.append(final_score)

        if episode_i % 10 == 0:
            plot_scores(game_scores)

        if episode_i % 50 == 0:
            print(f"Episode {episode_i}/{num_episodes}  "
                  f"score={game_scores[-1]}  "
                  f"avg_last_50={np.mean(game_scores[-50:]):.1f}")

    print('Training complete')
    plot_scores(game_scores, show_result=True)
    plt.ioff()
    plt.show()

    torch.save(value_net.state_dict(), "value_net.pt")
    print("Saved trained weights to value_net.pt")

def play_episode(game, value_net, tau, episode_i):
    game.reset()
    while True:
        board_before_2d = game.board.copy()
        streak_before = game.streak
        state_before = get_state (game)
        state_before_t = torch.tensor(state_before, dtype=torch.float32, device=device).unsqueeze(0)
        best_so_far, best_perm = iterate_best_sequence (game, value_net, epsilon = get_epsilon(episode_i))
        placements = [best_so_far["placement1"], best_so_far["placement2"], best_so_far["placement3"]]
        if best_so_far["current_best"] == float('-inf'):
            state_after = None
            reward = 0
            reward_t = torch.tensor([reward],dtype=torch.float32, device=device)
            push_with_rotation(memory, board_before_2d, streak_before, 0, None, None)
            optimize_model(memory, value_net, target_value_net, optimizer, batch_size, gamma) 
            soft_update (value_net, target_value_net, tau)
            break
        else:
            inputs = [(idx, pos[0], pos[1]) for idx, pos in zip(best_perm, placements)]
            for turn in inputs:
                turn_output = game.turn(turn[0], turn[1], turn[2])
            reward = turn_output["reward"]*reward_scale
            reward_t = torch.tensor([reward], dtype=torch.float32, device=device)
            if turn_output["game_over"]:
                state_after = None
                push_with_rotation(memory, board_before_2d, streak_before, reward, None, None)
                optimize_model(memory, value_net, target_value_net, optimizer, batch_size, gamma) 
                soft_update (value_net, target_value_net, tau)
                break
            else:
                board_after_2d = game.board.copy()
                streak_after_2d = game.streak
                push_with_rotation(memory, board_before_2d, streak_before, reward, board_after_2d, streak_after_2d)
                optimize_model(memory, value_net, target_value_net, optimizer, batch_size, gamma) 
                soft_update (value_net, target_value_net, tau)
    return game.score

def optimize_model(memory, value_net, target_value_net, optimizer, batch_size, gamma):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    
    state_before_batch = torch.cat(batch.state_before)
    predicted_values = value_net (state_before_batch)

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.state_after)), device=device, dtype=torch.bool)
    next_state_values = torch.zeros(batch_size, device=device)

    if non_final_mask.any():
        non_final_next_states = torch.cat([s for s in batch.state_after if s is not None])
        with torch.no_grad():
            next_state_values[non_final_mask] = target_value_net(non_final_next_states).squeeze(-1)

    reward_batch = torch.cat(batch.reward)
    expected_values = reward_batch + gamma * next_state_values

    criterion = nn.SmoothL1Loss()
    loss = criterion(predicted_values, expected_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(value_net.parameters(), 100)
    optimizer.step()

def evaluate(value_net, num_games=50):
    scores = []
    eval_game = blockblast()
    for _ in range(num_games):
        eval_game.reset()
        while True:
            best_so_far, best_perm = iterate_best_sequence (eval_game, value_net)
            placements = [best_so_far["placement1"], best_so_far["placement2"], best_so_far["placement3"]]
            if best_so_far["current_best"] == float('-inf'):
                scores.append(eval_game.score)
                break
            else:
                inputs = [(idx, pos[0], pos[1]) for idx, pos in zip(best_perm, placements)]
                for turn in inputs:
                    turn_output = eval_game.turn(turn[0], turn[1], turn[2])
                reward = turn_output["reward"]
                if turn_output["game_over"]:
                    scores.append(eval_game.score)
                    break
                    
    scores = np.array(scores)
    print(f"Evaluated over {num_games} games:")
    print(f"  mean score:   {scores.mean():.1f}")
    print(f"  median score: {np.median(scores):.1f}")
    print(f"  max score:    {scores.max()}")
    print(f"  min score:    {scores.min()}")
    return scores

def watch_game(value_net, delay=0.4, max_moves=300):
    import time as _time

    watch = blockblast()
    watch.reset()
    move_num = 0

    print("=" * 30)
    print("Starting fresh game")
    print_board(watch.board)

    while move_num < max_moves:
        best_so_far, best_perm = iterate_best_sequence(watch, value_net)
        placements = [best_so_far["placement1"], best_so_far["placement2"], best_so_far["placement3"]]

        if best_so_far["current_best"] == float('-inf'):
            print("\nGAME OVER (no valid 3-piece sequence found)")
            print(f"Final score: {watch.score}  (in {move_num} moves)")
            break

        inputs = [(idx, pos[0], pos[1]) for idx, pos in zip(best_perm, placements)]

        for piece_index, row, col in inputs:
            piece_name = watch.current_pieces[piece_index] 
            turn_output = watch.turn(piece_index, row, col)
            move_num += 1

            print("\n" + "=" * 30)
            print(f"Move {move_num}: placed '{piece_name}' at (row={row}, col={col})  "
                  f"lines_cleared={sum(turn_output['lines_cleared'])}  "
                  f"score={watch.score}  streak={watch.streak}")
            print_board(watch.board)

            _time.sleep(delay)

            if turn_output["game_over"]:
                print("\nGAME OVER")
                print(f"Final score: {watch.score}  (in {move_num} moves)")
                return
    else:
        print(f"\nStopped after reaching max_moves={max_moves} without a game-over "
              f"(current score: {watch.score})")

def print_board(board):
    print("   " + " ".join(str(c) for c in range(8)))
    for r in range(8):
        row_str = " ".join("#" if cell == 1 else "." for cell in board[r])
        print(f"{r}: {row_str}")

game = blockblast()
n_observations = len(get_state(game))

value_net = ValueNet(n_observations).to(device)
target_value_net = ValueNet(n_observations).to(device)
target_value_net.load_state_dict(value_net.state_dict())

optimizer = optim.AdamW(value_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(MEMORY_CAPACITY)
game_scores = []

if __name__ == "__main__":
    train()
    evaluate(value_net, num_games=50)
    watch_game(value_net, delay=0.4)



