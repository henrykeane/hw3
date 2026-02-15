"""
HW3 - Problem 3: Linear SVM via hinge loss minimization (from scratch)

Objective:
  J(w) = 0.5 * ||w[1:]||^2 + C * (1/N) * sum max(0, 1 - y_i*(w0 + x_i @ w[1:]))

y in {-1, +1}. Bias w0 is not regularized.

Allowed: numpy (required), optionally scipy/pandas for loading.
Do NOT use sklearn models for training.

Autograder imports these functions directly. Keep names/signatures unchanged.
"""

from __future__ import annotations
from typing import Dict, Any, Tuple
import numpy as np


def hinge_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray, C: float = 1.0) -> float:
    """
    Compute the hinge-loss SVM objective.

    Parameters
    ----------
    X : (N, d)
    y : (N,) in {-1, +1}
    w : (d+1,)
    C : float > 0

    Returns
    -------
    loss : float
    """
    N = X.shape[0]
    scores = w[0] + X.dot(w[1:])
    margins = y * scores
    hinge = np.maximum(0, 1 - margins)

    loss = 1 / 2 * np.sum(w[1:] ** 2) + C * np.mean(hinge)
    return loss


def hinge_grad(X: np.ndarray, y: np.ndarray, w: np.ndarray, C: float = 1.0) -> np.ndarray:
    """
    Subgradient of hinge_loss w.r.t. w.

    At margin exactly 1, any subgradient is acceptable; choose a deterministic convention.

    Returns
    -------
    grad : np.ndarray, shape (d+1,)
    """
    N = X.shape[0]
    scores = w[0] + X.dot(w[1:])
    margins = y * scores

    violated = margins < 1

    grad = np.zeros_like(w)

    grad[1:] = w[1:]

    if np.any(violated):
        grad[0] -= C * np.sum(y[violated]) / N
        grad[1:] -= C * C * (X[violated].T @ y[violated]) / N
    return grad


def predict_svm(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Predict labels in {-1, +1} using sign(w0 + X @ w[1:]).

    Break ties (score == 0) as +1.

    Returns
    -------
    yhat : np.ndarray, shape (N,)
        Entries in {-1, +1}.
    """
    scores = w[0] + X.dot(w[1:])
    return np.sign(scores)

def train_svm_hinge(
    X: np.ndarray,
    y: np.ndarray,
    C: float = 1.0,
    step_size: float = 0.1,
    max_epochs: int = 5000,
    tol: float = 1e-6,
    batch_size: int = 0,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Train hinge-loss SVM starting from w = zeros(d+1).

    If batch_size == 0: full-batch GD per epoch.
    Else: mini-batch GD with deterministic shuffling controlled by seed.

    Early stop if abs(obj_t - obj_{t-1}) < tol (objective computed on full data).

    Returns dict with at least:
      - "w": np.ndarray (d+1,)
      - "loss_history": list[float] objective value per epoch
      - "err_history": list[float] training error rate
      - "epochs": int
    """
    loss_history = []
    err_history = []
    w = np.zeros(X.shape[1]+1)

    if batch_size == 0:
        for i in range(max_epochs):
            g = hinge_grad(X, y, w, C)
            w -= step_size * g
            loss_t = hinge_loss(X, y, w, C)
            loss_history.append(loss_t)
            err_history.append(np.mean(predict_svm(X, w) != y))
            if(i > 0 and abs(loss_t - loss_history[i-1]) < tol):
                break
    else:
        np.random.seed(seed)
        for i in range(max_epochs):
            idx = np.random.permutation(X.shape[0])
            for j in range(0, X.shape[0], batch_size):
                X_batch = X[idx[j:j+batch_size]]
                y_batch = y[idx[j:j+batch_size]]
                g = hinge_grad(X_batch, y_batch, w, C)
                w -= step_size * g
            loss_t = hinge_loss(X, y, w, C)
            loss_history.append(loss_t)
            err_history.append(np.mean(predict_svm(X, w) != y))
            if(i > 0 and abs(loss_t - loss_history[i-1]) < tol):
                break
    return {"w": w, "loss_history": loss_history, "err_history": err_history, "epochs": i+1}

if __name__ == "__main__":
    # Quick self-check (not graded): binary iris (0 vs 1) if present
    try:
        data = np.loadtxt("data/iris.txt")
        Xall = data[:, :4]
        yall = data[:, 4].astype(int)

        mask = (yall == 0) | (yall == 1)
        X = Xall[mask]
        y01 = yall[mask]
        y = np.where(y01 == 0, -1, +1).astype(int)

        # Standardize using subset
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

        out = train_svm_hinge(X, y, C=1.0, step_size=0.05, max_epochs=5000, tol=1e-1, batch_size=0, seed=0)
        w = out["w"]
        yhat = predict_svm(X, w)
        print("Train acc:", float(np.mean(yhat == y)))
        print("Final objective:", out["loss_history"][-1], "epochs:", out["epochs"])
    except FileNotFoundError:
        print("data/iris.txt not found. Put dataset in data/ and rerun.")
    except NotImplementedError:
        print("Implement the TODOs first.")
