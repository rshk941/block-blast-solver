# block-blast-solver
An AI that plays Block Blast (an 8x8 block-puzzle mobile game) using exhaustive lookahead search and a learned value network, trained with reinforcement learning.

Video showing part of a game using block blast solver (solver is on another device - this video was edited to show the moves only)

https://github.com/user-attachments/assets/57d86931-0c00-41a0-9cf3-8b247ef0abd8

# What it does
Given the current 8x8 board and the 3 pieces in your tray, the solver:
1. Tries every possible order to place the 3 pieces (all 6 permutations).
2. For each order, exhaustively enumerates every legal placement of each piece, simulating line clears along the way.
3. Scores each resulting board using a trained ValueNet (a small neural network that estimates how "good" a board state is for future scoring potential), combined with the immediate reward from that sequence.
4. Picks the sequence + placements with the highest immediate reward + discounted future value.

# How it works
**Environment** (`blockblast` class)
- Simulates the real game: 8x8 board, 38 possible piece shapes (weighted to roughly match real drop rates), line/column clearing, and Block Blast's streak-based scoring system (consecutive clears multiply your score).

**State representation** - The board is encoded as a feature vector combining:
- The flattened 8x8 board
- Row and column fill counts
- A normalized clear-streak value
- A hole count (empty cells fully boxed in and unfillable)
- A placeability score (how much of the piece set could still legally be placed)

**Search** 
- Rather than a fixed heuristic, the search tree considers all 3! = 6 orderings of the tray and every valid (row, col) placement for each piece 

**Value network** 
- A 2-hidden-layer MLP (128 → 128 → 1) trained via TD bootstrapping with a target network and soft (Polyak) updates — similar in spirit to DQN, but learning a state-value function rather than Q-values, since the search step already handles action selection.

**Training**
- Replay memory of past board transitions, trained over ~800+ episodes
- 8-way symmetry augmentation (rotations + flips) applied to every board added to replay memory, effectively multiplying the training data
- Reward is the actual in-game score delta (scaled down), with random exploration via epsilon-greedy sequence selection

**Limitations**

This is very much a work in progress!! 
- It does not yet reliably outperform a skilled human player.
- The current interface for humans (play_assistant.py) is a manual, text-based CLI: you type in the board and pieces by hand. A proper UI is planned.

**Contributions, and ideas are welcome!!**

# Usage
**Train the model from scratch**

Run `python complete_blockblast2_6.py`

This trains for `num_episodes` games, plots score progress, evaluates over 50 games, and saves weights to `value_net.pt`.

**Get move recommendations for a live game**

Run `python play_assistant.py`

You'll be prompted to enter your current board (as rows of 0s/1s) and the 3 tray pieces by name. The script returns the recommended placement order, prints a board with the placement highlighted, and lets you undo or correct the board as you go
