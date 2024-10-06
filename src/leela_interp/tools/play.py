"""Utility wrappers for running an Lc0 model.

TODO: should probably merge this with Lc0 trees at some point, but for now they
don't have batch support yet.
"""

import pandas as pd
from collections import deque
import torch
import tqdm
from leela_interp import Lc0Model, LeelaBoard

DEFAULT_WANTED_PATHS = ["0", "00", "000", "0000", "00000", "001", "010", "002", "020", "01", "02", "1", "10", "100", "1000", "10000", "101", "2", "20", "200"]


def get_lc0_pv_probabilities(
    model: Lc0Model,
    puzzles: pd.DataFrame,
    batch_size: int = 100,
    pbar: bool | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Computes Lc0's probability for each move in the principal variation.

    Args:
        model: an LC0Model
        puzzles: a dataframe of puzzles. Will be batched automatically.
        batch_size: how many puzzles to feed into Lc0 at once.
        pbar: whether to show a progress bar. If None, determine automatically based
            on number of batches.

    Returns:
        A pandas series of lists of probabilities, where each list has one entry
        per move in the principal variation.
    """
    probs = []
    moves = []
    wdls = []
    if pbar is None:
        pbar = len(puzzles) > batch_size

    _range = tqdm.trange if pbar else range
    for i in _range(0, len(puzzles), batch_size):
        new_probs, new_moves, new_wdls = _get_lc0_pv_probabilities_single_batch(
            model, puzzles.iloc[i : i + batch_size]
        )
        probs.extend(new_probs)
        moves.extend(new_moves)
        wdls.extend(new_wdls)

    return (
        pd.Series(probs, index=puzzles.index),
        pd.Series(moves, index=puzzles.index),
        pd.Series(wdls, index=puzzles.index),
    )


def _get_lc0_pv_probabilities_single_batch(
    model: Lc0Model,
    puzzles: pd.DataFrame,
) -> tuple[list[list[float]], list[list[str]], list[list[float]]]:
    """Single batch of get_lc0_pv_probabilities, just a helper function."""
    max_len = puzzles.principal_variation.apply(len).max()
    boards = [LeelaBoard.from_puzzle(p) for _, p in puzzles.iterrows()]

    probs = [[] for _ in range(len(puzzles))]
    moves = [[] for _ in range(len(puzzles))]
    wdls = []

    for i in range(max_len):
        policies, wdl, _ = model.batch_play(boards, return_probs=True)
        if i == 0:
            wdls.extend(wdl.tolist())
        # Policies can be NaN if the board is in checkmate. We need to filter these
        # out for the allclose check.
        not_nan = ~torch.isnan(policies).any(-1)
        num_not_nan = not_nan.sum().item()
        assert isinstance(num_not_nan, int)  # make the type checker happy
        assert torch.allclose(
            policies[not_nan].sum(-1),
            torch.ones(num_not_nan, device=policies.device),
        ), policies.sum(-1)

        # Update all boards that have moves left:
        for j, board in enumerate(boards):
            pv = puzzles.iloc[j].principal_variation
            if i < len(pv):
                correct_move = pv[i]
                top_moves = model.top_moves(board, policies[j], top_k=None)
                model_move = next(iter(top_moves))
                probs[j].append(top_moves[correct_move])
                moves[j].append(model_move)
                board.push_uci(correct_move)

    return probs, moves, wdls


# TODO: we don't need this, but should have a test that checks get_lc0_pv_probabilities_batch
# against this implementation
def get_lc0_pv_probabilities_non_batched(puzzle):
    probs = []
    board = LeelaBoard.from_puzzle(puzzle)
    for move in puzzle.principal_variation:
        policy, _, _ = lc0_model.play(board, return_probs=True)
        assert torch.allclose(policy.sum(), torch.tensor(1.0))
        policy = lc0_model.policy_as_dict(board, policy)
        probs.append(policy[move])
        board.push_uci(move)

    return probs

def get_lc0_pv_probabilities_tree(
    model: Lc0Model,
    puzzles: pd.DataFrame,
    batch_size: int = 100,
    pbar: bool | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Computes Lc0's probability tree for the top 2 moves at each position.

    Args:
        model: an LC0Model
        puzzles: a dataframe of puzzles. Will be batched automatically.
        batch_size: how many puzzles to feed into Lc0 at once.
        pbar: whether to show a progress bar. If None, determine automatically based
            on number of batches.

    Returns:
        A tuple of two pandas series:
        1. A series of move trees, where each tree is a list of lists representing
           the top 2 moves and their probabilities for 3 levels deep.
        2. A series of WDL (win/draw/loss) scores for the initial position.
    """
    move_trees = []
    if pbar is None:
        pbar = len(puzzles) > batch_size

    _range = tqdm.trange if pbar else range
    for i in _range(0, len(puzzles), batch_size):
        new_move_trees = _get_lc0_pv_probabilities_single_batch_tree(
            model, puzzles.iloc[i : i + batch_size]
        )
        move_trees.extend(new_move_trees)

    return pd.Series(move_trees, index=puzzles.index)
    

def _get_lc0_pv_probabilities_single_batch_tree(
    model: Lc0Model,
    puzzles: pd.DataFrame,
    num_moves: int = 3,
    wanted_paths = None,
) -> tuple[list[list[float]], list[list[str]], list[list[float]]]:
    """Single batch of get_lc0_pv_probabilities, just a helper function."""
    if wanted_paths is None:
        wanted_paths = DEFAULT_WANTED_PATHS
    boards = [(LeelaBoard.from_puzzle(p)) for _, p in puzzles.iterrows()]
    boards_and_paths = deque([(i,"", board) for i, board in enumerate(boards)])

    batch_size = len(boards)
    move_trees = [dict() for _ in range(batch_size)]

    while boards_and_paths:
        batch_boards_and_paths = [boards_and_paths.popleft() for _ in range(min(len(boards_and_paths), batch_size))]    
        board_idxs, paths, boards = zip(*batch_boards_and_paths)
        policies, wdl, _ = model.batch_play(boards, return_probs=True)
        for k, (board_idx, path, board) in enumerate(batch_boards_and_paths):
            if policies[k][0] is None:
                continue
            top_moves = model.top_moves(board, policies[k], top_k=num_moves)
            for move_idx, (move, prob) in enumerate(top_moves.items()):
                new_path = path + str(move_idx)
                if new_path not in wanted_paths:
                    continue
                move_trees[board_idx][new_path] = move
                next_board = board.copy()
                next_board.push_uci(move)
                boards_and_paths.append((board_idx, new_path, next_board))
    return move_trees