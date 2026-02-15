"""
HW3 - Problem 1: Binary Logistic Regression (from scratch)

Allowed: numpy (required), optionally scipy/pandas for loading.
Do NOT use sklearn models for training in this homework.

Autograder imports these functions directly. Keep names/signatures unchanged.
"""

from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np


def load_iris_binary(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load iris data from `path` and create a binary dataset using ONLY classes {0,1}.

    Keep only samples with labels 0 or 1.
    Map label 0 -> 0 and label 1 -> 1.

    Returns
    -------
    X : np.ndarray, shape (N, 4)
        Standardized using only the retained subset.
    y : np.ndarray, shape (N,)
        Labels in {0,1}.
    """
    data = np.loadtxt(path)
    X = data[:, :4]
    y = data[:, 4]
    mask = (y == 0) | (y == 1)
    X = X[mask]
    y = y[mask]

    verysmallnumber = 1e-12 # prevent division by zero
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + verysmallnumber)
    X = np.c_[np.ones(X.shape[0]), X]

    return X, y



def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid, elementwise.

    Returns
    -------
    s : np.ndarray, same shape as z
    """
    return 1 / (1 + np.exp(-z))



def logistic_loss(X: np.ndarray, y: np.ndarray, w: np.ndarray, reg: float = 0.0) -> float:
    """
    Average negative log-likelihood with optional L2 regularization on w[1:].

    Model: p(y=1|x) = sigmoid(w0 + x @ w[1:])

    Parameters
    ----------
    X : (N, d)
    y : (N,) in {0,1}
    w : (d+1,) bias is w[0]
    reg : nonnegative float

    Returns
    -------
    loss : float
    """
    N = X.shape[0]
    p = sigmoid(X.dot(w))
    loss = -np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)) / N
    if reg > 0:
        loss += reg * np.sum(w[1:] ** 2)
    return loss


def logistic_grad(X: np.ndarray, y: np.ndarray, w: np.ndarray, reg: float = 0.0) -> np.ndarray:
    """
    Gradient of logistic_loss w.r.t. w.

    Returns
    -------
    grad : np.ndarray, shape (d+1,)
    """

    N = X.shape[0]
    p = sigmoid(X.dot(w))
    grad = X.T @ (p - y) / N
    if reg > 0:
        grad[1:] += 2 * reg * w[1:]
    return grad


def predict_proba(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """
    Return p(y=1|x) for each row of X.

    Returns
    -------
    p : np.ndarray, shape (N,)
    """
    return sigmoid(X.dot(w))

def predict(X: np.ndarray, w: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Return predicted labels in {0,1} using threshold on predicted probability.

    Returns
    -------
    yhat : np.ndarray, shape (N,)
    """
    return np.where(predict_proba(X, w) >= threshold, 1, 0)

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
    Train logistic regression starting from w = zeros(d+1).

    If batch_size == 0: full-batch GD (one gradient step per epoch).
    Else: mini-batch GD with deterministic shuffling controlled by seed.

    Early stop if abs(loss_t - loss_{t-1}) < tol (loss computed on full data).

    Returns dict with at least:
      - "w": np.ndarray (d+1,)
      - "loss_history": list[float] (or np.ndarray)
      - "err_history": list[float] training error rate (0..1)
      - "epochs": int
    """
    loss_history = []
    err_history = []
    w = np.zeros(X.shape[1])

    if batch_size == 0:
        for i in range(max_epochs):
            g = logistic_grad(X, y, w, reg)
            w -= step_size * g
            loss_t = logistic_loss(X, y, w, reg)
            loss_history.append(loss_t)
            err_history.append(np.mean(predict(X, w) != y))

            if(i > 0 and abs(loss_t - loss_history[i-1]) < tol):
                break
    else:
        np.random.seed(seed)
        for i in range(max_epochs):
            idx = np.random.permutation(X.shape[0])
            for j in range(0, X.shape[0], batch_size):
                X_batch = X[idx[j:j+batch_size]]
                y_batch = y[idx[j:j+batch_size]]
                g = logistic_grad(X_batch, y_batch, w, reg)
                w -= step_size * g

            loss_t = logistic_loss(X, y, w, reg)
            loss_history.append(loss_t)
            err_history.append(np.mean(predict(X, w) != y))

            if(i > 0 and abs(loss_t - loss_history[i-1]) < tol):
                break
    return {"w": w, "loss_history": loss_history, "err_history": err_history, "epochs": i+1}

if __name__ == "__main__":
    # Quick self-check (not graded)
    try:
        X, y = load_iris_binary("data/iris.txt")
        out = train_logreg(X, y, step_size=0.2, max_epochs=2000, tol=1e-9, batch_size=0, reg=1e-3, seed=0)
        w = out["w"]
        p = predict_proba(X, w)
        yhat = predict(X, w)
        print("X shape:", X.shape, "y mean:", float(np.mean(y)))
        print("Final loss:", out["loss_history"][-1], "final error:", out["err_history"][-1])
        print("Prob range:", float(np.min(p)), float(np.max(p)))
        print("Accuracy:", float(np.mean(yhat == y)))
    except FileNotFoundError:
        print("data/iris.txt not found. Put dataset in data/ and rerun.")
    except NotImplementedError:
        print("Implement the TODOs first.")
