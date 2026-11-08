"""
Numerical Linear Algebra from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - forward_substitution
import numpy as np

def forward_substitution(L, b):
    """
    Solve L @ y = b using forward substitution.

    Parameters:
        L : Square lower triangular matrix of shape (n, n).
        b : Right-hand side vector of shape (n,) or matrix of shape (n, k).

    Returns:
        y : Solution with the same shape as b, with float dtype.
    """
    L = np.asarray(L, dtype=float)
    b = np.asarray(b, dtype=float)

    # Validate input dimensions.
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("L must be a square matrix.")

    if b.ndim not in (1, 2) or b.shape[0] != L.shape[0]:
        raise ValueError("b must have shape (n,) or (n, k).")

    n = L.shape[0]

    # Initialize the solution.
    y = np.zeros_like(b, dtype=float)

    # Solve one row at a time, from top to bottom.
    for i in range(n):
        rhs = b[i] - L[i, :i] @ y[:i]
        y[i] = rhs / L[i, i]

    return y


def back_substitution(U, y):
    """
    Solve U @ x = y using backward substitution.

    Parameters:
        U : Square upper triangular matrix of shape (n, n).
        y : Right-hand side vector of shape (n,) or matrix of shape (n, k).

    Returns:
        x : Solution with the same shape as y, with float dtype.
    """
    U = np.asarray(U, dtype=float)
    y = np.asarray(y, dtype=float)

    # Validate input dimensions.
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("U must be a square matrix.")

    if y.ndim not in (1, 2) or y.shape[0] != U.shape[0]:
        raise ValueError("y must have shape (n,) or (n, k).")

    n = U.shape[0]

    # Initialize the solution.
    x = np.zeros_like(y, dtype=float)

    # Solve one row at a time, from bottom to top.
    for i in range(n - 1, -1, -1):
        rhs = y[i] - U[i, i + 1:] @ x[i + 1:]
        x[i] = rhs / U[i, i]

    return x

