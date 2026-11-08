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

# Step 2 - lu_no_pivot
def lu_no_pivot(A):
    """
    Perform LU factorization using Gaussian elimination without pivoting.

    Parameters:
        A : Square matrix of shape (n, n).

    Returns:
        L : Lower triangular matrix with ones on the diagonal.
        U : Upper triangular matrix.

    Raises:
        ValueError: If a zero pivot is encountered before the last
                    diagonal position.
    """
    # Create a floating-point copy so that the original matrix is unchanged.
    U = np.array(A, dtype=float, copy=True)

    # Validate that A is a square matrix.
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("A must be a square matrix.")

    n = U.shape[0]

    # Initialize L as an identity matrix.
    L = np.eye(n, dtype=float)

    # Perform Gaussian elimination without row exchanges.
    for k in range(n - 1):

        # A zero pivot would cause division by zero.
        if U[k, k] == 0:
            raise ValueError("Zero pivot encountered. LU factorization without pivoting is not possible.")

        # Eliminate the entries below the current pivot.
        for i in range(k + 1, n):

            # Calculate the elimination multiplier.
            factor = U[i, k] / U[k, k]

            # Store the multiplier in the lower triangular matrix.
            L[i, k] = factor

            # Update the corresponding row of U.
            U[i, k:] -= factor * U[k, k:]

            # Explicitly set the eliminated entry to zero.
            U[i, k] = 0.0

    return L, U

# Step 3 - lu_partial_pivot
def lu_partial_pivot(A):
    """
    Perform LU factorization with partial pivoting.

    The factorization satisfies P @ A = L @ U.

    Parameters:
        A : Square matrix of shape (n, n).

    Returns:
        P : Permutation matrix.
        L : Lower triangular matrix with unit diagonal.
        U : Upper triangular matrix.

    All returned arrays have float dtype.
    """
    # Create a floating-point copy to preserve the original matrix.
    U = np.array(A, dtype=float, copy=True)

    # Validate that A is square.
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("A must be a square matrix.")

    n = U.shape[0]

    # Initialize the permutation and lower triangular matrices.
    P = np.eye(n, dtype=float)
    L = np.eye(n, dtype=float)

    # Perform Gaussian elimination with partial pivoting.
    for k in range(n - 1):

        # Find the row containing the largest absolute pivot candidate.
        # np.argmax returns the first occurrence when values are tied.
        pivot_row = k + np.argmax(np.abs(U[k:, k]))

        # If the entire remaining column is zero, skip this column.
        if U[pivot_row, k] == 0.0:
            continue

        # Swap rows in U and P when necessary.
        if pivot_row != k:
            U[[k, pivot_row], :] = U[[pivot_row, k], :]
            P[[k, pivot_row], :] = P[[pivot_row, k], :]

            # Swap only the previously computed entries of L.
            L[[k, pivot_row], :k] = L[[pivot_row, k], :k]

        # Eliminate the entries below the pivot.
        for i in range(k + 1, n):

            # Calculate the elimination multiplier.
            factor = U[i, k] / U[k, k]

            # Store the multiplier in L.
            L[i, k] = factor

            # Update the corresponding row of U.
            U[i, k:] -= factor * U[k, k:]

            # Explicitly eliminate the entry below the pivot.
            U[i, k] = 0.0

    return P, L, U

# Step 4 - solve_lu
def solve_lu(A, b):
    """
    Solve the square linear system A @ x = b using
    LU factorization with partial pivoting.

    Parameters:
        A : Square coefficient matrix of shape (n, n).
        b : Right-hand side vector of shape (n,)
            or matrix of shape (n, k).

    Returns:
        x : Solution with the same shape as b and float dtype.

    Raises:
        ValueError: If A is singular (any diagonal entry of U is zero)
                    or if b has an invalid shape.
    """
    # Convert b to a floating-point array without modifying the original.
    b = np.asarray(b, dtype=float)

    # Factor A using LU decomposition with partial pivoting.
    P, L, U = lu_partial_pivot(A)

    n = U.shape[0]

    # Validate the dimensions of the right-hand side.
    if b.ndim not in (1, 2) or b.shape[0] != n:
        raise ValueError("b must have shape (n,) or (n, k).")

    # Check whether U has any exactly zero diagonal entries.
    if np.any(np.diag(U) == 0.0):
        raise ValueError("Matrix A is singular. Cannot solve the system.")

    # Apply the permutation matrix to the right-hand side.
    Pb = P @ b

    # First solve L @ y = P @ b using forward substitution.
    y = forward_substitution(L, Pb)

    # Then solve U @ x = y using backward substitution.
    x = back_substitution(U, y)

    return x

# Step 5 - cholesky_factor
def cholesky_factor(A):
    """
    Perform Cholesky factorization of a symmetric positive
    definite matrix A.

    The factorization satisfies A = L @ L.T.

    Parameters:
        A : Symmetric positive definite matrix of shape (n, n).

    Returns:
        L : Lower triangular matrix with strictly positive
            diagonal entries and zeros above the diagonal.

    Raises:
        ValueError: If A is not square or a pivot is not
                    strictly positive.
    """
    # Create a floating-point copy to preserve the original matrix.
    A = np.array(A, dtype=float, copy=True)

    # Validate that A is square.
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")

    n = A.shape[0]

    # Initialize L as a zero matrix.
    L = np.zeros((n, n), dtype=float)

    # Compute the Cholesky factor one column at a time.
    for j in range(n):

        # Calculate the current diagonal pivot.
        pivot = A[j, j] - np.dot(L[j, :j], L[j, :j])

        # A valid Cholesky pivot must be strictly positive.
        if not pivot > 0.0:
            raise ValueError(
                "Matrix is not positive definite: "
                "Cholesky pivot must be strictly positive."
            )

        # Calculate the diagonal element.
        L[j, j] = np.sqrt(pivot)

        # Calculate the elements below the diagonal.
        for i in range(j + 1, n):

            L[i, j] = (
                A[i, j] - np.dot(L[i, :j], L[j, :j])
            ) / L[j, j]

    return L

# Step 6 - logdet_chol
def logdet_chol(A):
    """
    Compute the log determinant of a symmetric positive
    definite matrix using Cholesky factorization.

    Parameters:
        A : Symmetric positive definite matrix of shape (n, n).

    Returns:
        logdet : Log determinant as a Python float.

    Raises:
        ValueError: If A is not positive definite.
    """
    # Compute the Cholesky factorization: A = L @ L.T.
    L = cholesky_factor(A)

    # Compute log(det(A)) = 2 * sum(log(diag(L))).
    logdet = 2.0 * np.sum(np.log(np.diag(L)))

    # Return a Python float.
    return float(logdet)


def solve_chol(A, b):
    """
    Solve A @ x = b using Cholesky factorization.

    Parameters:
        A : Symmetric positive definite matrix of shape (n, n).
        b : Right-hand side vector of shape (n,)
            or matrix of shape (n, k).

    Returns:
        x : Solution with the same shape as b and float dtype.

    Raises:
        ValueError: If A is not positive definite or b has an
                    invalid shape.
    """
    # Convert the right-hand side to a floating-point array.
    b = np.asarray(b, dtype=float)

    # Perform Cholesky factorization only once.
    L = cholesky_factor(A)

    n = L.shape[0]

    # Validate the dimensions of the right-hand side.
    if b.ndim not in (1, 2) or b.shape[0] != n:
        raise ValueError("b must have shape (n,) or (n, k).")

    # First solve L @ y = b using forward substitution.
    y = forward_substitution(L, b)

    # Then solve L.T @ x = y using backward substitution.
    x = back_substitution(L.T, y)

    return x

# Step 7 - mvn_sample
def mvn_sample(mu, Sigma, n, rng):
    """
    Draw samples from a multivariate normal distribution
    using Cholesky factorization.

    Parameters:
        mu    : Mean vector of shape (d,).
        Sigma : Covariance matrix of shape (d, d).
        n     : Number of samples to generate.
        rng   : NumPy random Generator.

    Returns:
        samples : Array of shape (n, d), with one sample per row.

    Raises:
        ValueError: If the dimensions of mu and Sigma do not match
                    or Sigma is not positive definite.
    """
    # Convert the mean vector to a floating-point array.
    mu = np.asarray(mu, dtype=float)

    # Validate the mean vector.
    if mu.ndim != 1:
        raise ValueError("mu must be a one-dimensional vector.")

    d = mu.shape[0]

    # Validate the covariance matrix dimensions.
    Sigma = np.asarray(Sigma, dtype=float)

    if Sigma.shape != (d, d):
        raise ValueError("Sigma must have shape (d, d), matching mu.")

    # Compute the Cholesky factorization: Sigma = L @ L.T.
    # This also raises ValueError if Sigma is not positive definite.
    L = cholesky_factor(Sigma)

    # Generate n independent standard normal samples.
    # Each row represents one sample in d dimensions.
    Z = rng.standard_normal(size=(n, d))

    # Transform the standard normal samples to the
    # required multivariate normal distribution.
    samples = Z @ L.T + mu

    return samples

