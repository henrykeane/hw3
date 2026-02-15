"""
HW3 - Problem 2: Softmax Regression (multiclass, from scratch)

Allowed: numpy (required), optionally scipy/pandas for loading.
Do NOT use sklearn models for training.

Autograder imports these functions directly. Keep names/signatures unchanged.
"""

from __future__ import annotations
from typing import Dict, Any
import numpy as np


def softmax(Z: np.ndarray) -> np.ndarray:
    """
    Stable softmax applied row-wise.

    Parameters
    ----------
    Z : np.ndarray, shape (N, K)

    Returns
    -------
    P : np.ndarray, shape (N, K)
        Each row sums to 1.
    """
    Z_max = np.max(Z, axis=1, keepdims=True)
    Z_exp = np.exp(Z - Z_max)
    P = Z_exp / np.sum(Z_exp, axis=1, keepdims=True)
    return P


def one_hot(y: np.ndarray, K: int) -> np.ndarray:
    """
    Convert y in {0,...,K-1} to one-hot.

    Returns
    -------
    Y : np.ndarray, shape (N, K)
    """
    Y = np.zeros((y.shape[0], K))
    Y[np.arange(y.shape[0]), y] = 1
    return Y

def softmax_loss(X: np.ndarray, y: np.ndarray, W: np.ndarray, reg: float = 0.0) -> float:
    """
    Average cross-entropy loss with L2 regularization on W[1:,:] (exclude bias row).

    Model logits:
      Z = b + X @ V
    where bias row is W[0,:] and weights are W[1:,:].

    Parameters
    ----------
    X : (N, d)
    y : (N,) in {0,...,K-1}
    W : (d+1, K)
    reg : nonnegative float

    Returns
    -------
    loss : float
    """
    N = X.shape[0]
    K = W.shape[1]
    Z = W[0,:] + X.dot(W[1:,:])

    Z_max = np.max(Z, axis=1, keepdims=True)
    log_sum_exmp = Z_max + np.log(np.sum(np.exp(Z - Z_max), axis=1, keepdims=True))

    Z_true_class = Z[np.arange(N), y]

    loss = np.mean(-Z_true_class + log_sum_exmp.flatten())

    if reg > 0:
        loss += 0.5 * reg * np.sum(W[1:,:] ** 2)

    return loss


def softmax_grad(X: np.ndarray, y: np.ndarray, W: np.ndarray, reg: float = 0.0) -> np.ndarray:
    """
    Gradient of softmax_loss w.r.t. W.

    Returns
    -------
    grad : np.ndarray, shape (d+1, K)
    """
    N = X.shape[0]
    K = W.shape[1]
    Z = W[0,:] + X.dot(W[1:,:])
    P = softmax(Z)
    Y = one_hot(y, K)
    
    grad = np.zeros_like(W)

    grad[0,:] = np.mean(P - Y, axis=0)

    grad[1:,:] = X.T @ (P - Y) / N

    if reg > 0:
        grad[1:,:] += reg * W[1:,:]
    return grad


def predict_proba_softmax(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Return class probabilities for each row.

    Returns
    -------
    P : np.ndarray, shape (N, K)
    """
    Z = W[0,:] + X.dot(W[1:,:])
    return softmax(Z)


def predict_softmax(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """
    Return predicted class labels (argmax over probabilities/logits).

    Returns
    -------
    yhat : np.ndarray, shape (N,)
    """
    return np.argmax(predict_proba_softmax(X, W), axis=1)


def train_softmax(
    X: np.ndarray,
    y: np.ndarray,
    K: int,
    step_size: float = 0.1,
    max_epochs: int = 3000,
    tol: float = 1e-6,
    batch_size: int = 0,
    reg: float = 0.0,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Train softmax regression from W = zeros(d+1, K).

    If batch_size == 0: full-batch GD per epoch.
    Else: mini-batch GD with deterministic shuffling controlled by seed.

    Early stop if abs(loss_t - loss_{t-1}) < tol (loss computed on full data).

    Returns dict with at least:
      - "W": np.ndarray (d+1, K)
      - "loss_history": list[float]
      - "acc_history": list[float] training accuracy
      - "epochs": int
    """
    loss_history = []
    acc_history = []
    W = np.zeros((X.shape[1]+1, K))

    if batch_size == 0:
        for i in range(max_epochs):
            g = softmax_grad(X, y, W, reg)
            W -= step_size * g
            loss_t = softmax_loss(X, y, W, reg)
            loss_history.append(loss_t)
            acc_history.append(np.mean(predict_softmax(X, W) == y))
            if(i > 0 and abs(loss_t - loss_history[i-1]) < tol):
                break
    else:
        np.random.seed(seed)
        for i in range(max_epochs):
            idx = np.random.permutation(X.shape[0])
            for j in range(0, X.shape[0], batch_size):
                X_batch = X[idx[j:j+batch_size]]
                y_batch = y[idx[j:j+batch_size]]
                g = softmax_grad(X_batch, y_batch, W, reg)
                W -= step_size * g
            loss_t = softmax_loss(X, y, W, reg)
            loss_history.append(loss_t)
            acc_history.append(np.mean(predict_softmax(X, W) == y))
            if(i > 0 and abs(loss_t - loss_history[i-1]) < tol):
                break
    return {"W": W, "loss_history": loss_history, "acc_history": acc_history, "epochs": i+1}


if __name__ == "__main__":
    # Quick self-check (not graded): load iris if present
    try:
        data = np.loadtxt("data/iris.txt")
        X = data[:, :4]
        y = data[:, 4].astype(int)

        # Standardize using all data for this demo
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

        K = int(np.max(y)) + 1
        out = train_softmax(X, y, K=K, step_size=0.2, max_epochs=3000, tol=1e-9, batch_size=0, reg=1e-3, seed=0)
        W = out["W"]
        yhat = predict_softmax(X, W)
        print("Train acc:", float(np.mean(yhat == y)))
        print("Final loss:", out["loss_history"][-1], "epochs:", out["epochs"])
    except FileNotFoundError:
        print("data/iris.txt not found. Put dataset in data/ and rerun.")
    except NotImplementedError:
        print("Implement the TODOs first.")
