import time
import os
import pickle
import random
from pathlib import Path

import chess
import iceberg as ice
import matplotlib.pyplot as plt
import numpy as np
import torch
from leela_interp import Lc0Model, Lc0sight, LeelaBoard
from leela_interp.core.iceberg_board import palette
from leela_interp.tools import figure_helpers as fh
from leela_interp.tools.piece_movement_heads import (
    bishop_heads,
    knight_heads,
    rook_heads,
)

with open("../puzzles_with_move_trees.pkl", "rb") as f:
    puzzles = pickle.load(f)
print(f"len of puzzles = {len(puzzles)}")
device = "mps"

all_effects = -torch.load(
    "../results/move_trees_patching/residual_stream_results.pt", map_location=device
)
print(f"len of all_effects = {len(all_effects)}")

# Initialize columns with default values if they don't exist
for col in ['effects_00000_tgt', 'effects_000_tgt', 'effects_001_tgt', 'effects_010_tgt', 'effects_00_tgt', 'effects_100_tgt']:
    if col not in puzzles.columns:
        puzzles[col] = 0.0  # Initialize with 0.0 instead of False

effects_00000_tgt = []
effects_000_tgt = []
effects_001_tgt = []
effects_010_tgt = []
effects_00_tgt = []
effects_100_tgt = []
residual_effects_idx = []

for i, (idx, puzzle) in enumerate(puzzles.iterrows()):
    residual_effects_idx.append(i)
    board = LeelaBoard.from_puzzle(puzzle)
    corrupted_board = LeelaBoard.from_fen(puzzle.corrupted_fen)

    patching_squares = []
    for square in chess.SQUARES:
        if board.pc_board.piece_at(square) != corrupted_board.pc_board.piece_at(
            square
        ):
            patching_squares.append(chess.SQUARE_NAMES[square])

    candidate_squares = [puzzle.principal_variation[0][2:4]]
    squares_00000_tgt = [puzzle.move_tree.get("00000", puzzle.principal_variation[0])[2:4]]
    squares_000_tgt = [puzzle.move_tree.get("000", puzzle.principal_variation[0])[2:4]]
    squares_001_tgt = [puzzle.move_tree.get("001", puzzle.principal_variation[0])[2:4]]
    squares_010_tgt = [puzzle.move_tree.get("010", puzzle.principal_variation[0])[2:4]]
    squares_00_tgt = [puzzle.move_tree.get("00", puzzle.principal_variation[0])[2:4]]
    squares_100_tgt = [puzzle.move_tree.get("100", puzzle.principal_variation[0])[2:4]]

    def process_effects(squares, patching_squares, candidate_squares, squares_000_tgt, is_000=False):
        if not (
            set(patching_squares).intersection(set(candidate_squares))
            or set(patching_squares).intersection(set(squares))
            or set(candidate_squares).intersection(set(squares))
            or (not is_000 and set(squares_000_tgt).intersection(set(squares)))
        ):
            return all_effects[i, :, [board.sq2idx(square) for square in squares]].mean().item()
        return 0.0

    effects_00000_tgt.append((idx, process_effects(squares_00000_tgt, patching_squares, candidate_squares, squares_000_tgt)))
    effects_000_tgt.append((idx, process_effects(squares_000_tgt, patching_squares, candidate_squares, squares_000_tgt, is_000=True)))
    effects_001_tgt.append((idx, process_effects(squares_001_tgt, patching_squares, candidate_squares, squares_000_tgt)))
    effects_010_tgt.append((idx, process_effects(squares_010_tgt, patching_squares, candidate_squares, squares_000_tgt)))
    effects_00_tgt.append((idx, process_effects(squares_00_tgt, patching_squares, candidate_squares, squares_000_tgt)))
    effects_100_tgt.append((idx, process_effects(squares_100_tgt, patching_squares, candidate_squares, squares_000_tgt)))

# Update the puzzles DataFrame with the mean values
for effects_list, col_name in [
    (effects_00000_tgt, "effects_00000_tgt"),
    (effects_000_tgt, "effects_000_tgt"),
    (effects_001_tgt, "effects_001_tgt"),
    (effects_010_tgt, "effects_010_tgt"),
    (effects_00_tgt, "effects_00_tgt"),
    (effects_100_tgt, "effects_100_tgt")
]:
    effects_dict = dict(effects_list)
    puzzles[col_name] = puzzles.index.map(effects_dict)

puzzles["residual_effects_idx"] = residual_effects_idx

with open("../puzzles_with_move_trees_and_tags.pkl", "wb") as f:
    pickle.dump(puzzles, f)