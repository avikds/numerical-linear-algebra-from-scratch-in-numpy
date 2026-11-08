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

# Step 8 - gram_schmidt_qr
def gram_schmidt_qr(A, modified=False):
    """
    Perform reduced QR factorization using Gram-Schmidt
    orthogonalization.

    Parameters:
        A : Matrix of shape (m, n), where m >= n and
            columns are linearly independent.
        modified : If False, use Classical Gram-Schmidt.
                   If True, use Modified Gram-Schmidt.

    Returns:
        Q : Matrix with orthonormal columns, shape (m, n).
        R : Upper triangular matrix with positive diagonal,
            shape (n, n).
    """
    # Convert A to a floating-point array without modifying the original.
    A = np.asarray(A, dtype=float)

    # Validate the input dimensions.
    if A.ndim != 2:
        raise ValueError("A must be a two-dimensional matrix.")

    m, n = A.shape

    if m < n:
        raise ValueError("A must have at least as many rows as columns.")

    # Initialize Q and R.
    Q = np.zeros((m, n), dtype=float)
    R = np.zeros((n, n), dtype=float)

    # Orthogonalize the columns of A one at a time.
    for j in range(n):

        # Start with the current column.
        v = A[:, j].copy()

        # Subtract projections onto previously computed
        # orthonormal columns.
        for i in range(j):

            if modified:
                # Modified Gram-Schmidt:
                # Compute the coefficient using the running residual.
                R[i, j] = np.dot(Q[:, i], v)

            else:
                # Classical Gram-Schmidt:
                # Compute the coefficient using the original column.
                R[i, j] = np.dot(Q[:, i], A[:, j])

            # Remove the projection from the residual.
            v -= R[i, j] * Q[:, i]

        # Calculate the norm of the remaining vector.
        norm_v = np.sqrt(np.dot(v, v))

        # Check for linear dependence or numerical breakdown.
        if norm_v == 0.0:
            raise ValueError("Columns of A are linearly dependent.")

        # Store the positive diagonal element of R.
        R[j, j] = norm_v

        # Normalize the remaining vector to obtain the next Q column.
        Q[:, j] = v / norm_v

    return Q, R

# Step 9 - householder_qr
def householder_qr(A):
    """
    Perform reduced QR factorization using Householder reflections.

    The factorization satisfies A = Q @ R.

    Parameters:
        A : Matrix of shape (m, n), where m >= n.

    Returns:
        Q : Matrix of shape (m, n) with orthonormal columns.
        R : Upper triangular matrix of shape (n, n)
            with non-negative diagonal entries.

    Raises:
        ValueError: If A is not a two-dimensional matrix or
                    if the number of rows is less than columns.
    """
    # Convert A to a floating-point copy to preserve the original.
    A = np.asarray(A, dtype=float)

    # Validate the input dimensions.
    if A.ndim != 2:
        raise ValueError("A must be a two-dimensional matrix.")

    m, n = A.shape

    if m < n:
        raise ValueError("A must have at least as many rows as columns.")

    # Working copy on which Householder reflections are applied.
    R_work = A.copy()

    # Store the Householder vectors for constructing Q later.
    reflectors = []

    # Perform one Householder reflection per column.
    for k in range(n):

        # Extract the part of the current column below and including
        # the diagonal position.
        x = R_work[k:, k].copy()

        # Scale the vector to improve numerical stability.
        scale = np.max(np.abs(x))

        # If the entire remaining column is zero, skip the reflection.
        # This also handles rank-deficient matrices safely.
        if scale == 0.0:
            reflectors.append(None)
            continue

        x_scaled = x / scale

        # Calculate the norm without using np.linalg.
        norm_scaled = np.sqrt(np.dot(x_scaled, x_scaled))

        # Choose the sign to avoid cancellation.
        if x_scaled[0] >= 0.0:
            alpha_scaled = -norm_scaled
        else:
            alpha_scaled = norm_scaled

        # Construct the Householder vector.
        v = x_scaled.copy()
        v[0] -= alpha_scaled

        # Normalize the Householder vector.
        v_norm = np.sqrt(np.dot(v, v))

        if v_norm == 0.0:
            reflectors.append(None)
            continue

        v /= v_norm

        # Apply the Householder reflection to the remaining columns:
        # H = I - 2 * v @ v.T
        R_work[k:, k:] -= 2.0 * np.outer(
            v,
            v @ R_work[k:, k:]
        )

        # Set the eliminated entries explicitly to zero.
        R_work[k, k] = alpha_scaled * scale
        R_work[k + 1:, k] = 0.0

        # Save the reflector for constructing Q.
        reflectors.append(v)

    # Construct the reduced Q matrix.
    # Apply the Householder reflections in reverse order.
    Q = np.eye(m, n, dtype=float)

    for k in range(n - 1, -1, -1):

        v = reflectors[k]

        # Skip columns for which no reflection was necessary.
        if v is not None:
            Q[k:, :] -= 2.0 * np.outer(
                v,
                v @ Q[k:, :]
            )

    # Extract the reduced upper triangular R matrix.
    R = R_work[:n, :].copy()

    # Normalize signs so every diagonal entry of R is non-negative.
    for j in range(n):
        if R[j, j] < 0.0:
            Q[:, j] *= -1.0
            R[j, :] *= -1.0

    return Q, R

# Step 10 - lstsq_normal
def lstsq_normal(A, b):
    """
    Solve an overdetermined least squares problem using
    the normal equations.

    The normal equations are:
        (A.T @ A) @ x = A.T @ b

    Parameters:
        A : Matrix of shape (m, n), where m > n.
        b : Right-hand side vector of shape (m,)
            or matrix of shape (m, k).

    Returns:
        x : Least squares solution with the same shape
            as b's column dimension, i.e. (n,) or (n, k).

    Raises:
        ValueError: If the input dimensions are invalid
                    or the Cholesky factorization fails.
    """
    # Convert inputs to floating-point arrays.
    # This does not modify the original inputs.
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    # Validate the dimensions of A.
    if A.ndim != 2:
        raise ValueError("A must be a two-dimensional matrix.")

    m, n = A.shape

    if m <= n:
        raise ValueError("A must be overdetermined (m > n).")

    # Validate the dimensions of b.
    if b.ndim not in (1, 2) or b.shape[0] != m:
        raise ValueError("b must have shape (m,) or (m, k).")

    # Form the normal equations.
    AtA = A.T @ A
    Atb = A.T @ b

    # Solve the normal equations using the Cholesky solver.
    # Any ValueError from the Cholesky factorization propagates.
    x = solve_chol(AtA, Atb)

    return x

# Step 11 - lstsq_qr
def lstsq_qr(A, b):
    """
    Solve a least squares problem using Householder QR factorization.

    The method computes:
        A = Q @ R
        R @ x = Q.T @ b

    Parameters:
        A : Matrix of shape (m, n), where m >= n,
            with full column rank.
        b : Right-hand side vector of shape (m,)
            or matrix of shape (m, k).

    Returns:
        x : Least squares solution of shape (n,)
            or (n, k), matching the number of right-hand sides.

    Raises:
        ValueError: If the input dimensions are invalid.
    """
    # Convert inputs to floating-point arrays without modifying
    # the original inputs.
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    # Validate the dimensions of A.
    if A.ndim != 2:
        raise ValueError("A must be a two-dimensional matrix.")

    m, n = A.shape

    if m < n:
        raise ValueError("A must have at least as many rows as columns.")

    # Validate the dimensions of b.
    if b.ndim not in (1, 2) or b.shape[0] != m:
        raise ValueError("b must have shape (m,) or (m, k).")

    # Perform reduced Householder QR factorization.
    # A = Q @ R
    Q, R = householder_qr(A)

    # Project the right-hand side onto the orthonormal
    # columns of Q.
    Qtb = Q.T @ b

    # Solve the upper triangular system:
    # R @ x = Q.T @ b
    x = back_substitution(R, Qtb)

    return x

# Step 12 - power_method
def power_method(A, v0, n_iter):
    """
    Estimate the dominant eigenvalue and eigenvector of a
    real symmetric matrix using the Power Method.

    Parameters:
        A      : Real square matrix of shape (n, n).
        v0     : Initial vector of shape (n,).
        n_iter : Number of power iterations to perform.

    Returns:
        lam : Rayleigh quotient, returned as a Python float.
        u   : Final normalized eigenvector estimate of shape (n,).

    Raises:
        ValueError: If A is not square, v0 has an invalid shape,
                    or a zero vector prevents normalization.
    """
    # Convert inputs to floating-point arrays without
    # modifying the original inputs.
    A = np.asarray(A, dtype=float)
    u = np.asarray(v0, dtype=float).copy()

    # Validate the matrix dimensions.
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")

    n = A.shape[0]

    # Validate the initial vector.
    if u.ndim != 1 or u.shape[0] != n:
        raise ValueError("v0 must be a vector of shape (n,).")

    # Validate the number of iterations.
    if not isinstance(n_iter, (int, np.integer)) or n_iter < 0:
        raise ValueError("n_iter must be a non-negative integer.")

    # Normalize the initial vector.
    norm_u = np.sqrt(np.dot(u, u))

    if norm_u == 0.0:
        raise ValueError("Initial vector must be nonzero.")

    u = u / norm_u

    # Perform exactly n_iter power iterations.
    for _ in range(n_iter):

        # Multiply the matrix by the current eigenvector estimate.
        v = A @ u

        # Calculate the Euclidean norm.
        norm_v = np.sqrt(np.dot(v, v))

        # A zero result cannot be normalized.
        if norm_v == 0.0:
            raise ValueError(
                "Power iteration produced a zero vector; "
                "normalization is not possible."
            )

        # Normalize the updated vector.
        u = v / norm_v

    # Calculate the Rayleigh quotient.
    lam = float(np.dot(u, A @ u))

    return lam, u

# Step 13 - top_k_deflation
def top_k_deflation(A, k, n_iter, seed=0):
    """
    Find the top k eigenpairs of a real symmetric matrix
    using the Power Method and matrix deflation.

    Parameters:
        A      : Real symmetric matrix of shape (n, n).
        k      : Number of eigenpairs to compute.
        n_iter : Number of power iterations for each eigenpair.
        seed   : Random seed for reproducibility (default=0).

    Returns:
        vals : Array of shape (k,) containing the eigenvalues
               in the order they are found.
        vecs : Array of shape (n, k) containing the corresponding
               normalized eigenvectors as columns.
    """
    # Create a floating-point copy to preserve the original matrix.
    A_work = np.array(A, dtype=float, copy=True)

    # Validate that A is a square matrix.
    if A_work.ndim != 2 or A_work.shape[0] != A_work.shape[1]:
        raise ValueError("A must be a square matrix.")

    n = A_work.shape[0]

    # Validate k.
    if not isinstance(k, (int, np.integer)) or not 0 <= k <= n:
        raise ValueError("k must be an integer between 0 and n.")

    # Create a random number generator.
    rng = np.random.default_rng(seed)

    # Initialize storage for eigenvalues and eigenvectors.
    vals = np.zeros(k, dtype=float)
    vecs = np.zeros((n, k), dtype=float)

    # Find eigenpairs one at a time using deflation.
    for j in range(k):

        # Generate a fresh random initial vector.
        v0 = rng.standard_normal(n)

        # Find the dominant eigenpair of the current working matrix.
        lam, u = power_method(A_work, v0, n_iter)

        # Store the eigenvalue and eigenvector.
        vals[j] = lam
        vecs[:, j] = u

        # Deflate the working matrix by removing the contribution
        # of the eigenpair just computed.
        A_work -= lam * np.outer(u, u)

    return vals, vecs

# Step 14 - orthogonal_iteration
def orthogonal_iteration(A, k, n_iter, seed=0):
    """
    Find the top k eigenpairs of a real symmetric matrix
    using orthogonal iteration (block power method).

    Parameters:
        A      : Real symmetric matrix of shape (n, n).
        k      : Number of eigenpairs to estimate.
        n_iter : Number of orthogonal iterations.
        seed   : Random seed for reproducibility (default=0).

    Returns:
        vals : Rayleigh quotient estimates of shape (k,),
               sorted by decreasing absolute value.
        vecs : Orthonormal eigenvector estimates of shape (n, k),
               with eigenvectors stored as columns.

    Raises:
        ValueError: If A is not square, k is invalid,
                    or n_iter is invalid.
    """
    # Convert A to a floating-point array without modifying
    # the original matrix.
    A = np.asarray(A, dtype=float)

    # Validate that A is square.
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix.")

    n = A.shape[0]

    # Validate k.
    if not isinstance(k, (int, np.integer)) or not 0 <= k <= n:
        raise ValueError("k must be an integer between 0 and n.")

    # Validate the number of iterations.
    if not isinstance(n_iter, (int, np.integer)) or n_iter < 0:
        raise ValueError("n_iter must be a non-negative integer.")

    # Create a reproducible random number generator.
    rng = np.random.default_rng(seed)

    # Generate a random starting matrix of shape (n, k).
    V = rng.standard_normal(size=(n, k))

    # Orthonormalize the initial columns using Householder QR.
    V, _ = householder_qr(V)

    # Perform exactly n_iter orthogonal iterations.
    for _ in range(n_iter):

        # Multiply the current orthonormal block by A.
        Z = A @ V

        # Re-orthonormalize the columns using Householder QR.
        V, _ = householder_qr(Z)

    # Calculate the Rayleigh quotient for each final column.
    AV = A @ V
    vals = np.sum(V * AV, axis=0)

    # Sort eigenvalue estimates by decreasing absolute value.
    order = np.argsort(-np.abs(vals), kind="stable")

    # Rearrange eigenvalues and their corresponding eigenvectors.
    vals = vals[order]
    vecs = V[:, order]

    return vals, vecs

# Step 15 - svd_via_eig
def svd_via_eig(A, n_iter=500, seed=0):
    """
    Compute the thin Singular Value Decomposition (SVD)
    using eigenvectors of A.T @ A or A @ A.T.

    The decomposition satisfies:
        A = (U * s) @ Vt

    Parameters:
        A      : Input matrix of shape (m, n).
        n_iter : Number of orthogonal iterations.
        seed   : Random seed for reproducibility.

    Returns:
        U  : Left singular vectors of shape (m, r).
        s  : Singular values of shape (r,), in decreasing order.
        Vt : Transpose of right singular vectors, shape (r, n).

        where r = min(m, n).
    """
    # Convert A to a floating-point array without modifying the original.
    A = np.asarray(A, dtype=float)

    # Validate the input dimensions.
    if A.ndim != 2:
        raise ValueError("A must be a two-dimensional matrix.")

    m, n = A.shape
    r = min(m, n)

    # Case 1: A has at least as many rows as columns.
    if m >= n:

        # Form the smaller symmetric matrix A.T @ A.
        gram = A.T @ A

        # Compute its eigenvalues and eigenvectors using
        # orthogonal iteration.
        eigvals, V = orthogonal_iteration(
            gram, r, n_iter, seed
        )

        # Clip negative eigenvalues caused by floating-point
        # rounding before taking square roots.
        s = np.sqrt(np.maximum(eigvals, 0.0))

        # Recover the left singular vectors.
        AV = A @ V
        U = np.zeros((m, r), dtype=float)

        for i in range(r):

            if s[i] > 0.0:
                U[:, i] = AV[:, i] / s[i]
            else:
                # Do not divide by a zero singular value.
                U[:, i] = AV[:, i]

        # Right singular vectors are stored as rows in Vt.
        Vt = V.T

    # Case 2: A has more columns than rows.
    else:

        # Form the smaller symmetric matrix A @ A.T.
        gram = A @ A.T

        # Compute its eigenvalues and eigenvectors using
        # orthogonal iteration.
        eigvals, U = orthogonal_iteration(
            gram, r, n_iter, seed
        )

        # Clip negative eigenvalues before taking square roots.
        s = np.sqrt(np.maximum(eigvals, 0.0))

        # Recover the right singular vectors.
        ATU = A.T @ U
        V = np.zeros((n, r), dtype=float)

        for i in range(r):

            if s[i] > 0.0:
                V[:, i] = ATU[:, i] / s[i]
            else:
                # Do not divide by a zero singular value.
                V[:, i] = ATU[:, i]

        # Right singular vectors are stored as rows in Vt.
        Vt = V.T

    return U, s, Vt

# Step 16 - svd_jacobi
def svd_jacobi(A, tol=1e-14, max_sweeps=40):
    """
    Compute the thin Singular Value Decomposition using
    the one-sided Jacobi algorithm.

    The decomposition satisfies:
        A = (U * s) @ Vt

    Parameters:
        A          : Input matrix of shape (m, n).
        tol        : Relative orthogonality tolerance.
        max_sweeps : Maximum number of Jacobi sweeps.

    Returns:
        U  : Left singular vectors of shape (m, r).
        s  : Singular values of shape (r,), sorted in
             decreasing order and non-negative.
        Vt : Transpose of right singular vectors,
             shape (r, n).

        where r = min(m, n).

    Notes:
        - Works directly on A without forming A.T @ A.
        - Handles wide and tall matrices.
        - Singular values at or below 1e-15 * s[0]
          are treated as exactly zero.
        - The original input matrix is not modified.
    """
    # Convert A to a floating-point array.
    A = np.asarray(A, dtype=float)

    # Validate the input dimensions.
    if A.ndim != 2:
        raise ValueError("A must be a two-dimensional matrix.")

    m, n = A.shape

    # For a wide matrix, compute the SVD of A.T and
    # exchange the roles of the left and right factors.
    if m < n:
        U_t, s, Vt_t = svd_jacobi(
            A.T, tol=tol, max_sweeps=max_sweeps
        )

        U = Vt_t.T
        Vt = U_t.T

        return U, s, Vt

    # Work with a copy of A so the original is unchanged.
    B = A.copy()

    # Initialize V as the identity matrix.
    # Throughout the algorithm, B = A @ V.
    V = np.eye(n, dtype=float)

    # Perform Jacobi sweeps over all pairs of columns.
    for sweep in range(max_sweeps):

        rotations_performed = False

        for p in range(n - 1):
            for q in range(p + 1, n):

                # Extract the current pair of columns.
                ap = B[:, p]
                aq = B[:, q]

                # Calculate the squared column norms
                # and the inner product.
                alpha = np.dot(ap, ap)
                beta = np.dot(aq, aq)
                gamma = np.dot(ap, aq)

                # Calculate the column norms.
                norm_p = np.sqrt(alpha)
                norm_q = np.sqrt(beta)

                # Skip pairs that are already sufficiently
                # orthogonal, or contain a zero column.
                if (
                    gamma == 0.0
                    or norm_p == 0.0
                    or norm_q == 0.0
                    or abs(gamma) <= tol * norm_p * norm_q
                ):
                    continue

                # Calculate the Jacobi rotation parameter.
                tau = (beta - alpha) / (2.0 * gamma)

                # Compute t using a numerically stable formula.
                t = np.copysign(1.0, tau) / (
                    abs(tau) + np.hypot(1.0, tau)
                )

                # Calculate the cosine and sine of the rotation.
                c = 1.0 / np.sqrt(1.0 + t * t)
                s = t * c

                # Save the original columns before updating them.
                ap_old = B[:, p].copy()
                aq_old = B[:, q].copy()

                # Apply the Jacobi rotation to the working matrix.
                B[:, p] = c * ap_old - s * aq_old
                B[:, q] = s * ap_old + c * aq_old

                # Apply the same rotation to V.
                vp_old = V[:, p].copy()
                vq_old = V[:, q].copy()

                V[:, p] = c * vp_old - s * vq_old
                V[:, q] = s * vp_old + c * vq_old

                rotations_performed = True

        # Stop when an entire sweep performs no rotations.
        if not rotations_performed:
            break

    # Calculate the singular values from the column lengths
    # of the orthogonalized working matrix.
    singular_values = np.sqrt(np.sum(B * B, axis=0))

    # Sort singular values in decreasing order.
    order = np.argsort(-singular_values, kind="stable")

    singular_values = singular_values[order]
    B = B[:, order]
    V = V[:, order]

    # Initialize U with zeros.
    U = np.zeros((m, n), dtype=float)

    # Apply the numerical zero threshold.
    if n > 0:
        threshold = 1e-15 * singular_values[0]
    else:
        threshold = 0.0

    # Construct the left singular vectors.
    for j in range(n):

        if singular_values[j] > threshold:

            # Normalize the corresponding column of B.
            U[:, j] = B[:, j] / singular_values[j]

        else:
            # Treat sufficiently small singular values as zero.
            singular_values[j] = 0.0

            # The corresponding column of U remains zero.
            U[:, j] = 0.0

    # The right singular vectors are stored as rows in Vt.
    Vt = V.T

    return U, singular_values, Vt

