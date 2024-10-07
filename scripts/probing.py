import argparse
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import tqdm
from leela_interp import ActivationCache, Lc0Model, LeelaBoard
from leela_interp.tools import probing



def main(args):
    if not (args.main or args.random_model):
        raise ValueError("Please specify at least one of --main or --random_model")

    torch.set_num_threads(args.num_threads)

    base_dir = Path(args.base_dir)

    try:
        with open(base_dir / "interesting_puzzles_without_corruptions.pkl", "rb") as f:
            puzzles = pickle.load(f)
    except FileNotFoundError:
        raise ValueError("Puzzles not found, run make_puzzles.py first")

    if args.split == "all":
        pass
    elif args.split == "different_targets":
        puzzles = puzzles[puzzles["different_targets"]]
    elif args.split == "same_targets":
        puzzles = puzzles[~puzzles["different_targets"]]
    else:
        raise ValueError(
            f"Unknown split: {args.split}, "
            "expected 'all', 'different_targets', or 'same_targets'"
        )

    if args.n_puzzles > 0:
        puzzles = puzzles.iloc[: args.n_puzzles]

    # Use 70% of puzzles for training, rest for testing
    n_train = int(len(puzzles) * 0.7)
    print(f"Using {len(puzzles)} puzzles total, {n_train} for training.")

    hparams = {
        "n_epochs": 5,
        "lr": 1e-2,
        "weight_decay": 0,
        "k": 32,
        "batch_size": 64,
        "device": args.device,
    }

    if args.main:
        model = Lc0Model(base_dir / "lc0.onnx", device=args.device)

        activations = ActivationCache.capture(
            model=model,
            boards=[LeelaBoard.from_puzzle(p) for _, p in puzzles.iterrows()],
            # There's a typo in Lc0, so we mirror it; "rehape" is deliberate
            names=["attn_body/ma_gating/rehape2"]
            + [f"encoder{layer}/ln2" for layer in range(15)],
            n_samples=len(puzzles),
            # Uncomment to store activations on disk (they're about 70GB).
            # Without a path, they'll be kept in memory, which is faster but uses 70GB of RAM.
            # path="residual_activations.zarr",
            overwrite=True,
        )

        for seed in range(args.n_seeds):
            torch.manual_seed(seed)

            target_probes, source_probes = probing.train_probes(
                activations, puzzles, n_train, hparams, probing.collect_data
            )

            save_dir = base_dir / f"results/probing/{args.split}/{seed}"
            save_dir.mkdir(parents=True, exist_ok=True)

            with open(save_dir / "target_probes.pkl", "wb") as f:
                pickle.dump(target_probes, f)
            with open(save_dir / "source_probes.pkl", "wb") as f:
                pickle.dump(source_probes, f)

            probing.eval_probes(
                target_probes,
                source_probes,
                puzzles,
                activations,
                n_train,
                save_dir / "main.pkl",
                collect_data_fn=probing.collect_data
            )

        # Free up memory in case we're running the random model next
        del activations

    if args.random_model:
        random_model = Lc0Model(onnx_model_path=base_dir / "lc0-random.onnx", device=args.device)
        activations = ActivationCache.capture(
            boards=[LeelaBoard.from_puzzle(p) for _, p in puzzles.iterrows()],
            names=["attn_body/ma_gating/rehape2"]
            + [f"encoder{layer}/ln2" for layer in range(15)],
            n_samples=len(puzzles),
            # path="random_activations.zarr",
            store_boards=True,
            overwrite=True,
            model=random_model,
        )

        for seed in range(args.n_seeds):
            torch.manual_seed(seed)

            save_dir = base_dir / f"results/probing/{args.split}/{seed}"
            save_dir.mkdir(parents=True, exist_ok=True)

            target_probes, source_probes = probing.train_probes(
                activations, puzzles, n_train, hparams, probing.collect_data
            )
            probing.eval_probes(
                target_probes,
                source_probes,
                puzzles,
                activations,
                n_train,
                save_dir / "random_model.pkl",
                collect_data_fn=probing.collect_data
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--n_seeds", default=1, type=int)
    parser.add_argument("--base_dir", default=".", type=str)
    parser.add_argument("--n_puzzles", default=0, type=int)
    parser.add_argument("--num_threads", default=1, type=int)
    parser.add_argument("--main", action="store_true")
    parser.add_argument("--random_model", action="store_true")
    parser.add_argument("--split", default="all", type=str)
    args = parser.parse_args()
    main(args)
