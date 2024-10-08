import functools
import warnings
from dataclasses import dataclass
from typing import Callable, Literal
import pickle

import numpy as np
import pandas as pd
import torch
import tqdm
from einops import einsum
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from leela_interp import ActivationCache, LeelaBoard



def train_probe(
    X: np.ndarray,
    y: list[int] | np.ndarray,
    z_squares: list[int] | np.ndarray,
    *,
    n_epochs,
    lr,
    batch_size,
    weight_decay,
    k,
    device,
):
    """Train a bilinear probe to predict y from X and Z.

    Args:
        X: should have shape (n_boards, 64, d).
        y: should have shape (n_boards,) and values in {0, ..., 63}.
        z_squares: should have shape (n_boards,) and values in {0, ..., 63}.

    Returns:
        A trained bilinear probe.
    """
    Z = X[np.arange(len(X)), z_squares, :]
    assert Z.shape == (len(X), 768)

    dataset = ProbeData.create(X=X, y=y, Z=Z)
    split_data = dataset.split(val_split=0)

    probe = BilinearSquarePredictor()
    probe.train(
        split_data,
        n_epochs=n_epochs,
        lr=lr,
        batch_size=batch_size,
        weight_decay=weight_decay,
        k=k,
        device=device,
        pbar=False,
    )

    return probe


def collect_data(
    puzzles: pd.DataFrame,
    activations: ActivationCache,
    activation_name: str,
    prediction_target: Literal["target", "source"],
    n_train: int,
    train=True,
):
    ys = []
    z_squares = []
    if train:
        boards = activations.boards[:n_train]
        puzzles = puzzles.iloc[:n_train]
    else:
        boards = activations.boards[n_train:]
        puzzles = puzzles.iloc[n_train:]

    for board, (_, puzzle) in zip(boards, puzzles.iterrows()):
        # Important check to make sure we're not accidentally using activations
        # from different puzzles:
        assert board.fen() == LeelaBoard.from_puzzle(puzzle).fen()
        if prediction_target == "target":
            # Predict the third move target from the first move target:
            y = puzzle.principal_variation[2][2:4]
            z = puzzle.principal_variation[0][2:4]
        elif prediction_target == "source":
            # Predict the third move source from the third move target:
            y = puzzle.principal_variation[2][:2]
            z = puzzle.principal_variation[2][2:4]
        ys.append(board.sq2idx(y))
        z_squares.append(board.sq2idx(z))

    if train:
        X = activations[activation_name][:n_train]
    else:
        X = activations[activation_name][n_train:]
    assert X.shape[1:] == (64, 768), X.shape

    ys = np.array(ys)
    z_squares = np.array(z_squares)

    return X, ys, z_squares


def eval_probe(target_probe, source_probe, puzzles, activations, name, n_train, collect_data_fn):
    X, target_y, z_squares = collect_data_fn(
        puzzles, activations, name, "target", n_train=n_train, train=False
    )
    _, source_y, _ = collect_data_fn(
        puzzles, activations, name, "source", n_train=n_train, train=False
    )

    # Predict target squares:
    Z = X[np.arange(len(X)), z_squares, :]
    assert Z.shape == (len(X), 768)
    target_squares = target_probe.predict(X, Z)

    # Predict source squares:
    Z = X[np.arange(len(X)), target_squares, :]
    assert Z.shape == (len(X), 768)
    source_squares = source_probe.predict(X, Z)

    source_accuracy = (source_squares == source_y).mean()
    target_accuracy = (target_squares == target_y).mean()
    accuracy = ((source_squares == source_y) * (target_squares == target_y)).mean()

    return source_accuracy, target_accuracy, accuracy

def eval_probe_top_k(target_probe, source_probe, puzzles, activations, name, n_train, collect_data_fn, k=3):
    X, target_y, z_squares = collect_data_fn(
        puzzles, activations, name, "target", n_train=n_train, train=False
    )
    _, source_y, _ = collect_data_fn(
        puzzles, activations, name, "source", n_train=n_train, train=False
    )

    # Predict target squares:
    Z = X[np.arange(len(X)), z_squares, :]
    assert Z.shape == (len(X), 768)
    target_squares_top_k = target_probe.predict_top_k(X, Z, k)

    # Predict source squares:
    Z = X[np.arange(len(X)), target_squares_top_k[:, 0], :]  # Use the top prediction for Z
    assert Z.shape == (len(X), 768)
    source_squares_top_k = source_probe.predict_top_k(X, Z, k)

    source_accuracy_top_k = np.mean([source_y[i] in source_squares_top_k[i] for i in range(len(source_y))])
    target_accuracy_top_k = np.mean([target_y[i] in target_squares_top_k[i] for i in range(len(target_y))])
    accuracy_top_k = np.mean([(source_y[i] in source_squares_top_k[i]) and (target_y[i] in target_squares_top_k[i]) for i in range(len(source_y))])

    # For compatibility, also calculate top-1 accuracy
    source_accuracy = (source_squares_top_k[:, 0] == source_y).mean()
    target_accuracy = (target_squares_top_k[:, 0] == target_y).mean()
    accuracy = ((source_squares_top_k[:, 0] == source_y) & (target_squares_top_k[:, 0] == target_y)).mean()

    return {
        'source_accuracy': source_accuracy,
        'target_accuracy': target_accuracy,
        'accuracy': accuracy,
        'source_accuracy_top_k': source_accuracy_top_k,
        'target_accuracy_top_k': target_accuracy_top_k,
        'accuracy_top_k': accuracy_top_k,
        'k': k
    }


def train_probes(activations, puzzles, n_train, hparams, collect_data_fn):
    target_probes = []
    for layer in range(15):
        #print(f"Layer {layer}")
        name = f"encoder{layer}/ln2"
        X, y, z_squares = collect_data_fn(
            puzzles, activations, name, "target", n_train=n_train
        )
        probe = train_probe(X, y, z_squares, **hparams)
        target_probes.append(probe)

    source_probes = []
    for layer in range(15):
        #print(f"Layer {layer}")
        name = f"encoder{layer}/ln2"
        X, y, z_squares = collect_data_fn(
            puzzles, activations, name, "source", n_train=n_train
        )
        probe = train_probe(X, y, z_squares, **hparams)
        source_probes.append(probe)

    return target_probes, source_probes


def eval_probes(target_probes, source_probes, puzzles, activations, n_train, path, collect_data_fn):
    accuracies = []
    source_accuracies = []
    target_accuracies = []
    
    for layer, target_probe, source_probe in zip(
        tqdm.trange(15), target_probes, source_probes
    ):
        name = f"encoder{layer}/ln2"
        source_accuracy, target_accuracy, accuracy = eval_probe(
            target_probe, source_probe, puzzles, activations, name, n_train, collect_data_fn)
        
        accuracies.append(accuracy)
        source_accuracies.append(source_accuracy)
        target_accuracies.append(target_accuracy)
    
    results_dict = {
        "accuracies": np.array(accuracies),
        "source_accuracies": np.array(source_accuracies),
        "target_accuracies": np.array(target_accuracies),
    }
    
    if path is not None:
        with open(path, "wb") as f:
            pickle.dump(results_dict, f)
    
    return results_dict


@dataclass
class ProbeData:
    X: np.ndarray
    y: np.ndarray
    indices: np.ndarray
    # Additional information, one for each sample. Not used directly by any probing
    # code, just for use-case-specific debugging.
    extra: list | None = None
    # Variable to condition on in addition to X.
    Z: np.ndarray | None = None

    @classmethod
    def create(
        cls,
        X: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor | list,
        extra: list | None = None,
        indices: np.ndarray | None = None,
        Z: np.ndarray | torch.Tensor | None = None,
    ):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.cpu().numpy()
        if isinstance(y, list):
            y = np.array(y)
        if indices is None:
            indices = np.arange(len(X))
        if isinstance(Z, torch.Tensor):
            Z = Z.cpu().numpy()

        return cls(X=X, y=y, indices=indices, extra=extra, Z=Z)

    def __len__(self):
        return len(self.X)

    def split(self, val_split: float = 0.3):
        return SplitProbeData.create(self, val_split=val_split)


@dataclass
class SplitProbeData:
    train: ProbeData
    val: ProbeData

    @classmethod
    def create(cls, data: ProbeData, val_split: float = 0.3):
        n_samples = len(data.X)

        assert data.y.shape == (n_samples,)
        assert data.indices.shape == (n_samples,)
        assert data.extra is None or len(data.extra) == n_samples

        n_train = int(n_samples * (1 - val_split))

        return cls(
            train=ProbeData(
                X=data.X[:n_train],
                y=data.y[:n_train],
                indices=data.indices[:n_train],
                extra=data.extra[:n_train] if data.extra is not None else None,
                Z=data.Z[:n_train] if data.Z is not None else None,
            ),
            val=ProbeData(
                X=data.X[n_train:],
                y=data.y[n_train:],
                indices=data.indices[n_train:],
                extra=data.extra[n_train:] if data.extra is not None else None,
                Z=data.Z[n_train:] if data.Z is not None else None,
            ),
        )


def make_data(
    activations: ActivationCache,
    *,
    boards: list[LeelaBoard] | None = None,
    puzzles: pd.DataFrame | None = None,
    data_func: Callable[[list[LeelaBoard] | pd.DataFrame, ActivationCache], ProbeData],
    val_split: float = 0.3,
    max_boards: int | None = None,
    use_puzzles: bool = False,
) -> SplitProbeData:
    n_positions = len(activations)
    if max_boards is not None:
        n_positions = min(n_positions, max_boards)

    if puzzles is not None:
        assert boards is None
        puzzles = puzzles.iloc[:n_positions]
        if len(puzzles) != n_positions:
            warnings.warn(
                f"{n_positions} positions requested, but only {len(puzzles)} available."
            )
    elif boards is not None:
        boards = boards[:n_positions]
        if len(boards) != n_positions:
            warnings.warn(
                f"{n_positions} positions requested, but only {len(boards)} available."
            )
    else:
        raise ValueError("Either boards or puzzles must be provided")

    if use_puzzles:
        assert puzzles is not None
        data = data_func(puzzles, activations)
    else:
        if boards is None:
            assert puzzles is not None
            boards = [LeelaBoard.from_puzzle(p) for _, p in puzzles.iterrows()]
        data = data_func(boards, activations)

    return SplitProbeData.create(data, val_split=val_split)


def batch_data_func(activation_name: str, pbar: bool = False):
    """Wraps a data_func that operates on a single board and batches it.

    Important: the data_func must still return a (potentially singleton) batch dimension!
    """

    def decorator(data_func):
        @functools.wraps(data_func)
        def wrapper(
            boards_or_puzzles: list[LeelaBoard] | pd.DataFrame,
            activations: ActivationCache,
        ):
            np_activations = activations.numpy(activation_name)
            X = []
            y = []
            indices = []
            extra = []
            Z = []

            if isinstance(boards_or_puzzles, list):
                iterable = boards_or_puzzles
            else:
                assert isinstance(boards_or_puzzles, pd.DataFrame)
                iterable = (p for _, p in boards_or_puzzles.iterrows())

            iterable = tqdm.tqdm(iterable) if pbar else iterable
            for i, board_or_puzzle in enumerate(iterable):
                result = data_func(board_or_puzzle, np_activations[i])
                X.extend(result.X)
                y.extend(result.y)
                indices.extend(np.full(len(result.X), i))
                if result.extra is not None:
                    extra.extend(result.extra)
                if result.Z is not None:
                    Z.extend(result.Z)

            assert len(X) == len(y) == len(indices)
            assert len(extra) == 0 or len(extra) == len(X)
            assert len(Z) == 0 or len(Z) == len(X)

            return ProbeData(
                X=np.array(X),
                y=np.array(y),
                extra=extra if len(extra) > 0 else None,
                indices=np.array(indices),
                Z=np.array(Z) if len(Z) > 0 else None,
            )

        return wrapper

    return decorator


class LinearProbe:
    def __init__(self, data: SplitProbeData):
        self.data = data

    def train(
        self,
        inverse_regularizer: float = 1.0,
        class_weight: str | dict | None = None,
    ):
        self.scaler = preprocessing.StandardScaler()
        X_scaled = self.scaler.fit_transform(self.data.train.X)

        self.clf = LogisticRegression(C=inverse_regularizer, class_weight=class_weight)
        self.clf.fit(X_scaled, self.data.train.y)
        predictions = self.clf.predict(X_scaled)
        print(classification_report(self.data.train.y, predictions))

    def predict(self, X: np.ndarray | torch.Tensor):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        X_scaled = self.scaler.transform(X)
        return self.clf.predict(X_scaled)

    def predict_proba(self, X: np.ndarray | torch.Tensor):
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        X_scaled = self.scaler.transform(X)
        return self.clf.predict_proba(X_scaled)

    def validate(self):
        predictions = self.predict(self.data.val.X)
        print(classification_report(self.data.val.y, predictions))


class BilinearSquarePredictor:
    """Predicts a square y in {0, ..., 63} from inputs X and Z using a bilinear model.

    X should have shape (n_samples, 64, d_x) and Z should be (n_samples, d_z).
    The logit for square y is then computed from X[:, y, :] and Z, using a low-rank
    bilinear model.
    """

    def predict_proba(self, X: np.ndarray, Z: np.ndarray):
        device = self.W_x.device
        with torch.inference_mode():
            X_pt = torch.tensor(X, dtype=torch.float32, device=device)
            Z_pt = torch.tensor(Z, dtype=torch.float32, device=device)
            y_pred = einsum(
                self.W_x,
                X_pt,
                self.W_z,
                Z_pt,
                "k d_x, batch square d_x, k d_z, batch d_z -> batch square",
            )
            y_pred = y_pred + self.c

            return torch.softmax(y_pred, dim=-1).cpu().numpy()

    def predict(self, X: np.ndarray, Z: np.ndarray):
        return self.predict_proba(X, Z).argmax(axis=-1)

    def predict_top_k(self, X: np.ndarray, Z: np.ndarray, k: int):
        probs = self.predict_proba(X, Z)
        return np.argsort(probs, axis=-1)[:, -k:][:, ::-1]

    def train(
        self,
        data: SplitProbeData,
        n_epochs: int = 100,
        batch_size: int = 1024,
        weight_decay: float = 0.0,
        lr: float = 3e-4,
        k: int = 32,
        sqrt_k_factor: bool = True,
        device: str = "cpu",
        pbar: bool = True,
    ):
        X = data.train.X
        y = data.train.y
        Z = data.train.Z
        assert Z is not None
        assert Z.ndim == 2
        assert X.ndim == 3
        assert X.shape[1] == 64
        assert len(X) == len(y) == len(Z)

        d_x = X.shape[2]
        d_z = Z.shape[1]

        self.W_x = torch.randn(k, d_x, device=device) / d_x**0.5
        self.W_z = torch.randn(k, d_z, device=device) / d_z**0.5
        # Set after dividing by sqrt(d) to make sure these are leaf tensors:
        self.W_x.requires_grad = True
        self.W_z.requires_grad = True
        self.c = torch.randn(1, device=device, requires_grad=True)

        optimizer = torch.optim.Adam(
            [self.W_x, self.W_z, self.c], lr=lr, weight_decay=weight_decay
        )

        with tqdm.trange(n_epochs, disable=not pbar) as epochs:
            for epoch in epochs:
                running_loss = 0.0
                for i in range(0, len(X), batch_size):
                    X_batch = torch.tensor(
                        X[i : i + batch_size], dtype=torch.float32, device=device
                    )
                    Z_batch = torch.tensor(
                        Z[i : i + batch_size], dtype=torch.float32, device=device
                    )
                    y_batch = torch.tensor(
                        y[i : i + batch_size], dtype=torch.long, device=device
                    )

                    y_pred = einsum(
                        self.W_x,
                        X_batch,
                        self.W_z,
                        Z_batch,
                        "k d_x, batch square d_x, k d_z, batch d_z -> batch square",
                    )
                    if sqrt_k_factor:
                        y_pred = y_pred / k**0.5
                    y_pred = y_pred + self.c
                    assert y_batch.ndim == 1
                    assert y_pred.shape == (len(y_batch), 64)
                    loss = torch.nn.functional.cross_entropy(y_pred, y_batch)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.detach()

                running_loss /= len(X) / batch_size

                epochs.set_postfix({"loss": running_loss.item()})

    def validate(self, data: SplitProbeData):
        predictions = self.predict(data.val.X, data.val.Z)
        print(classification_report(data.val.y, predictions))
