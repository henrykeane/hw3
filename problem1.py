"""
HW3 - Problem 1: Logistic Regression (unit-testable)

You will implement binary logistic regression (from scratch) and utilities.
Allowed: numpy, scipy, pandas, scikit-learn.

IMPORTANT:
- Do not rename functions.
- Return types/shapes must match docstrings.
- Your implementation must be deterministic given the seed.
"""

from __future__ import annotations

from typing import Tuple, Dict, Any, Iterable, Optional
import numpy as np


def load_iris_binary(path: str, pair: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load iris data from `path` and create a binary dataset using label pair.

    Parameters
    ----------
    path : str
        Path to data/iris.txt
    pair : (int, int)
        Either (0,1) or (1,2).
        The smaller class is mapped to 0 and the larger class to 1.

    Returns
    -------
    X : np.ndarray, shape (N, 2)
        Standardized (zero mean, unit variance) using ONLY the returned subset.
        Uses only the first two features.
    y : np.ndarray, shape (N,)
        Binary labels in {0,1}.
    """
    # TODO
    raise NotImplementedError


def is_linearly_separable(X: np.ndarray, y: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Return True if there exists a linear separator with zero training error.

    Hint: you may use sklearn.svm.LinearSVC with large C.
    """
    # TODO
    raise NotImplementedError


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Numerically-stable sigmoid applied elementwise.

    Parameters
    ----------
    z : np.ndarray

    Returns
    -------
    s : np.ndarray, same shape as z
        s = 1 / (1 + exp(-z))
    """
    # TODO
    raise NotImplementedError


def logistic_loss(X: np.ndarray, y: np.ndarray, theta: np.ndarray, reg: float = 0.0) -> float:
    """
    Average negative log-likelihood with optional L2 regularization on theta[1:].

    Parameters
    ----------
    X : np.ndarray, shape (N, d)
    y : np.ndarray, shape (N,), values in {0,1}
    theta : np.ndarray, shape (d+1,)
        theta[0] is bias, theta[1:] correspond to features.
    reg : float
        L2 regularization strength (nonnegative).

    Returns
    -------
    loss : float
    """
    # TODO
    raise NotImplementedError


def logistic_grad(X: np.ndarray, y: np.ndarray, theta: np.ndarray, reg: float = 0.0) -> np.ndarray:
    """
    Gradient of logistic_loss with respect to theta.

    Returns
    -------
    grad : np.ndarray, shape (d+1,)
    """
    # TODO
    raise NotImplementedError


def predict_proba(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Predict p(y=1|x) for each row in X.

    Returns
    -------
    p : np.ndarray, shape (N,)
    """
    # TODO
    raise NotImplementedError


def predict(X: np.ndarray, theta: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Predict labels in {0,1} using predicted probability and a threshold.

    Returns
    -------
    yhat : np.ndarray, shape (N,)
        dtype can be int or bool, but must be values in {0,1}.
    """
    # TODO
    raise NotImplementedError


def train_logreg(
    X: np.ndarray,
    y: np.ndarray,
    step_size: float = 0.1,
    max_epochs: int = 2000,
    tol: float = 1e-6,
    batch_size: int = 0,
    reg: float = 0.0,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Train logistic regression from theta=0 using full-batch GD or mini-batch GD.

    If batch_size == 0: full-batch gradient descent per epoch.
    Else: use mini-batches of size batch_size.

    Stopping criteria:
      - reach max_epochs, OR
      - abs(loss_t - loss_{t-1}) < tol  (computed once per epoch on full dataset)

    Returns
    -------
    out : dict with at least:
      - "theta": np.ndarray shape (d+1,)
      - "loss_history": list[float] (or np.ndarray)
      - "err_history": list[float] training error rate per epoch
      - "epochs": int number of epochs actually run
    """
    # TODO
    raise NotImplementedError


def decision_boundary(theta: np.ndarray, x1_grid: np.ndarray) -> np.ndarray:
    """
    For 2D features, return x2 values where theta0 + theta1*x1 + theta2*x2 = 0.

    If theta2 == 0, return np.full_like(x1_grid, np.nan).

    Parameters
    ----------
    theta : np.ndarray, shape (3,)
    x1_grid : np.ndarray, shape (M,)

    Returns
    -------
    x2_grid : np.ndarray, shape (M,)
    """
    # TODO
    raise NotImplementedError


def evaluate_trained(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> Dict[str, Any]:
    """
    Compute evaluation stats for a trained model.

    Returns
    -------
    out : dict with keys:
      - "loss": float
      - "error_rate": float
      - "confusion": np.ndarray shape (2,2)
        confusion[true, pred] counts, where true/pred are 0 or 1.
    """
    # TODO
    raise NotImplementedError


def best_threshold(
    y_true: np.ndarray, p_hat: np.ndarray, metric: str = "f1"
) -> float:
    """
    Search thresholds in {0.05, 0.10, ..., 0.95} and return the threshold
    that maximizes the chosen metric.

    metric in {"f1", "balanced_accuracy"}.
    Break ties by choosing the smallest threshold.

    Returns
    -------
    thr : float
    """
    # TODO
    raise NotImplementedError


def regularization_path(
    X: np.ndarray,
    y: np.ndarray,
    regs: Iterable[float],
    **train_kwargs: Any,
) -> Dict[str, Any]:
    """
    Extra credit: train models for each reg in regs.

    Parameters
    ----------
    regs : iterable of nonnegative floats
    train_kwargs : forwarded to train_logreg (e.g., step_size, max_epochs, tol, batch_size, seed)

    Returns
    -------
    out : dict with keys:
      - "regs": list[float] same order as input regs
      - "thetas": list[np.ndarray] each shape (d+1,)
      - "losses": list[float] final loss for each trained model
    """
    # TODO
    raise NotImplementedError


if __name__ == "__main__":
    # Quick sanity checks (not graded)
    import numpy as np

    # 1) Load binary Iris tasks (requires data/iris.txt to exist)
    try:
        X01, y01 = load_iris_binary("data/iris.txt", (0, 1))
        X12, y12 = load_iris_binary("data/iris.txt", (1, 2))
        print("Loaded iris (0,1):", X01.shape, y01.shape, "separable?", is_linearly_separable(X01, y01))
        print("Loaded iris (1,2):", X12.shape, y12.shape, "separable?", is_linearly_separable(X12, y12))
    except FileNotFoundError:
        print("data/iris.txt not found. Skipping iris load demo.")

    # 2) Tiny synthetic demo for loss/grad/training
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 2))
    true_theta = np.array([0.2, 1.0, -1.2])
    p = predict_proba(X, true_theta) if "predict_proba" in globals() else None
    y = (rng.random(20) < (1 / (1 + np.exp(-(true_theta[0] + X @ true_theta[1:]))))).astype(int)

    theta0 = np.zeros(3)
    try:
        L0 = logistic_loss(X, y, theta0)
        g0 = logistic_grad(X, y, theta0)
        print("Initial loss:", float(L0))
        print("Initial grad norm:", float(np.linalg.norm(g0)))
        out = train_logreg(X, y, step_size=0.5, max_epochs=200, tol=1e-9, batch_size=0, reg=0.0, seed=0)
        print("Trained epochs:", out["epochs"])
        print("Final loss:", out["loss_history"][-1])
        print("Final error:", out["err_history"][-1])
        thr = best_threshold(y, predict_proba(X, out["theta"]), metric="f1")
        print("Best threshold (F1):", thr)
    except NotImplementedError:
        print("Some functions are not implemented yet.")
