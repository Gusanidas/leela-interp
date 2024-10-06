import time
import argparse
import pickle
from pathlib import Path

import pandas as pd
from leela_interp import Lc0Model, get_lc0_pv_probabilities_tree

def main(args):
    base_dir = Path(args.base_dir)
    input_file = base_dir / args.input_file
    output_file = base_dir / args.output_file

    # Load the puzzles from the pickle file
    with open(input_file, 'rb') as f:
        puzzles = pickle.load(f)

    # Limit the number of puzzles if specified
    if args.n_puzzles:
        puzzles = puzzles.iloc[:args.n_puzzles]

    # Create a Lc0Model instance
    big_model = Lc0Model(base_dir / "lc0.onnx", device=args.device)
    small_model = Lc0Model(base_dir / "LD2.onnx", device=args.device)

    # Compute the move trees and WDL scores
    move_trees = get_lc0_pv_probabilities_tree(
        big_model,
        puzzles,
        batch_size=args.batch_size,
        pbar=True
    )

    # Add the move_tree and wdl_score columns to the DataFrame
    puzzles['move_tree'] = move_trees
    #puzzles['wdl_score'] = wdl_scores

    sparring_move_trees = get_lc0_pv_probabilities_tree(
        small_model,
        puzzles,
        batch_size=args.batch_size,
        pbar=True
    )

    puzzles['sparring_move_tree'] = sparring_move_trees

    # Save the updated DataFrame to a new pickle file
    with open(output_file, 'wb') as f:
        pickle.dump(puzzles, f)

    print(f"Processed {len(puzzles)} puzzles. Results saved to {output_file}")

if __name__ == "__main__":
    t0 = time.time()
    parser = argparse.ArgumentParser(description="Process chess puzzles and add move trees.")
    parser.add_argument('--base_dir', type=str, default='..',
                        help='Base directory for input/output files and models')
    parser.add_argument('--input_file', type=str, default='interesting_puzzles.pkl',
                        help='Input pickle file containing chess puzzles')
    parser.add_argument('--output_file', type=str, default='puzzles_with_move_trees.pkl',
                        help='Output pickle file to save processed puzzles')
    parser.add_argument('--device', type=str, default='mps',
                        help='Device to run the model on (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for processing puzzles')
    parser.add_argument('--n_puzzles', type=int, default=12_000,
                        help='Number of puzzles to process (default: all)')
    args = parser.parse_args()
    main(args)
    print(f"Elapsed time: {time.time() - t0:.2f} seconds")