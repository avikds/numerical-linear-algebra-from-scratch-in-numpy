"""
Numerical Linear Algebra from Scratch in NumPy scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Numerical Linear Algebra from Scratch (Murphy, Probabilistic Machine Learning: An Introduction, chapter 7).
Every decomposition below runs on your own code. np.linalg appears only to check your answers against the real thing."""
import time

import numpy as np
from sklearn.datasets import fetch_olivetti_faces


def hilbert(n):
    return 1.0 / (np.arange(1, n + 1)[:, None] + np.arange(n)[None, :])


def with_spectrum(s, m, n, seed):
    r = np.random.default_rng(seed)
    U = householder_qr(r.standard_normal((m, len(s))))[0]
    V = householder_qr(r.standard_normal((n, len(s))))[0]
    return (U * s) @ V.T


def main():
    t_start = time.time()

    # --- Part 1 ---
    print("Part 1: triangular systems and LU")
    A = np.array([[3.0, 2.0, -1.0], [2.0, -2.0, 4.0], [-1.0, 0.5, -1.0]])
    x = solve_lu(A, np.array([1.0, -2.0, 0.0]))
    print(f"  book's system (eq 7.214): x = {(np.round(x, 6) + 0.0).tolist()}  (book: [1, -2, -2])")
    E = np.array([[1e-20, 1.0], [1.0, 1.0]])
    be = np.array([1.0, 2.0])
    L, U = lu_no_pivot(E)
    print(f"  tiny pivot, no pivoting: x = {(back_substitution(U, forward_substitution(L, be)) + 0.0).tolist()}, partial pivoting: x = {np.round(solve_lu(E, be), 6).tolist()}")
    R = np.random.default_rng(0).standard_normal((300, 300))
    xt = np.ones(300)
    b = R @ xt
    L, U = lu_no_pivot(R)
    e0 = np.linalg.norm(back_substitution(U, forward_substitution(L, b)) - xt) / np.sqrt(300)
    e1 = np.linalg.norm(solve_lu(R, b) - xt) / np.sqrt(300)
    print(f"  random 300 x 300: relative error {e0:.1e} without pivoting, {e1:.1e} with; largest multiplier {np.abs(L).max():.0f} without, 1 with")

    # --- Part 2 ---
    print("Part 2: Cholesky")
    print(f"  0.1 I_100: det {np.linalg.det(0.1 * np.eye(100)):.1e}, log det {logdet_chol(0.1 * np.eye(100)):.4f}")
    print(f"  0.1 I_400: det {np.linalg.det(0.1 * np.eye(400))}, log det {logdet_chol(0.1 * np.eye(400)):.4f}")
    S = np.array([[2.0, 0.8], [0.8, 1.0]])
    y = mvn_sample(np.array([1.0, -1.0]), S, 200000, np.random.default_rng(0))
    print(f"  MVN sampling (eq 7.209): target cov {S.tolist()}, empirical {np.round(np.cov(y.T), 3).tolist()}")

    # --- Part 3 ---
    print("Part 3: QR and least squares")
    print("  loss of orthogonality ||Q^T Q - I||_F on Hilbert matrices:")
    for n in (6, 8, 10, 12):
        H = hilbert(n)
        loss = [np.linalg.norm(f(H)[0].T @ f(H)[0] - np.eye(n)) for f in (lambda a: gram_schmidt_qr(a, False), lambda a: gram_schmidt_qr(a, True), householder_qr)]
        print(f"    n={n:2d} kappa {np.linalg.cond(H):.1e}: classical GS {loss[0]:.1e}, modified GS {loss[1]:.1e}, Householder {loss[2]:.1e}")

    # --- Part 4 ---
    print("Part 4: power method")
    Q = householder_qr(np.random.default_rng(1).standard_normal((6, 6)))[0]
    lam = np.array([5.0, 4.0, 2.0, 1.0, 0.5, 0.1])
    S6 = (Q * lam) @ Q.T
    v0 = np.random.default_rng(2).standard_normal(6)

    def tan_angle(t):
        u = power_method(S6, v0, t)[1]
        c = u @ Q[:, 0]
        return np.linalg.norm(u - c * Q[:, 0]) / abs(c)

    print(f"  measured error reduction per iteration {tan_angle(31) / tan_angle(30):.4f}, |lambda_2 / lambda_1| = {lam[1] / lam[0]:.4f}")
    
    # Load the Olivetti Faces dataset
    d = fetch_olivetti_faces(shuffle=False)

    F = np.asarray(d.data, dtype=float)
    fy = np.asarray(d.target, dtype=int)

    Fc = F - F.mean(axis=0)
    G = Fc @ Fc.T
    w, Ut = np.linalg.eigh(G)
    w, Ut = w[::-1], Ut[:, ::-1]
    print("  faces Gram matrix (400 x 400), top k eigenvectors, 300 iterations (subspace error / orthogonality loss):")
    for k in (5, 20, 50):
        Vd = top_k_deflation(G, k, 300, 0)[1]
        Vo = orthogonal_iteration(G, k, 300, 0)[1]
        U_k = Ut[:, :k]
        err = lambda V: np.linalg.norm(V @ (V.T @ U_k) - U_k) / np.sqrt(k)
        print(f"    k={k:2d}: deflation {err(Vd):.1e} / {np.linalg.norm(Vd.T @ Vd - np.eye(k)):.1e}, orthogonal iteration {err(Vo):.1e} / {np.linalg.norm(Vo.T @ Vo - np.eye(k)):.1e}")

    # --- Part 5 ---
    print("Part 5: the SVD, twice")
    s_t = np.geomspace(1.0, 1e-14, 8)
    Bad = with_spectrum(s_t, 20, 8, 0)
    se = svd_via_eig(Bad)[1]
    sj = svd_jacobi(Bad)[1]
    print("  true sigma  " + " ".join(f"{v:9.1e}" for v in s_t))
    print("  via A^T A   " + " ".join(f"{v:9.1e}" for v in se))
    print("  Jacobi      " + " ".join(f"{v:9.1e}" for v in sj))
    N49 = 0.5 * np.array([[1.0, 1.0], [1.0 + 1e-10, 1.0 - 1e-10]])
    moved = solve_lu(N49, np.array([1.0 + 1e-6, 1.0]))
    print(f"  kappa(0.1 I_100) = {condition_number(0.1 * np.eye(100)):.3f}; eq 7.49: kappa = {condition_number(N49):.3e}, b1 + 1e-6 moves x from (1, 1) to ({moved[0]:.1f}, {moved[1]:.1f})")
    print(f"  rank of [[0, 1], [0, 0]] is {rank_and_null_space(np.array([[0.0, 1.0], [0.0, 0.0]]))[0]} though both of its eigenvalues are 0")
    print("  polynomial fits, relative coefficient error:")
    for r in least_squares_study([5, 8, 10, 12, 14, 16]):
        print(f"    degree {r['degree']:2d} kappa {r['kappa']:.1e}: normal {r['normal']:.1e}, QR {r['qr']:.1e}, pinv {r['pinv']:.1e}")

    # --- Part 6 ---
    print("Part 6: truncated SVD on a 200 x 320 photo")
    img = book_image()
    U, s, Vt = svd_jacobi(img)
    print(f"  parameters and errors (full image has {img.size} numbers, ||A||_F = {np.linalg.norm(img):.2f}):")
    for K in (2, 5, 20, 50):
        A_hat = (U[:, :K] * s[:K]) @ Vt[:K]
        fro, spec, printed = truncation_error(s, K)
        print(f"    K={K:3d}: {K * (200 + 320 + 1):6d} numbers, direct ||A - A_K||_F {np.linalg.norm(img - A_hat):7.3f}, corrected formula {fro:7.3f}, eq 7.201 as printed {printed:7.2f} ({printed / fro:.1f}x too large)")
    so, ss = spectrum_vs_shuffled(img, 0)
    print("  Figure 7.10, log sigma_i photo / shuffled: " + ", ".join(f"i={i} {np.log(so[i - 1]):.2f} / {np.log(ss[i - 1]):.2f}" for i in (1, 2, 20, 100)))
    print(f"  K for 99% of the energy: photo {rank_for_energy(so, 0.99)}, shuffled {rank_for_energy(ss, 0.99)}")

    # --- Part 7 ---
    print("Part 7: block structure")
    M = np.random.default_rng(0).standard_normal((40, 40)) + 5 * np.eye(40)
    M_inv, _ = partitioned_inverse(M, 10)
    print(f"  partitioned inverse (eq 7.109): ||M M^-1 - I||_F = {np.linalg.norm(M @ M_inv - np.eye(40)):.1e}")
    print("  Woodbury (eq 7.123) against a dense LU solve, D = 5:")
    for N in (250, 500, 1000):
        r = np.random.default_rng(N)
        X = r.standard_normal((N, 5))
        sig = 1.0 + r.random(N)
        b = r.standard_normal(N)
        t = time.time()
        x1 = woodbury_solve(sig, X, b)
        t1 = time.time() - t
        t = time.time()
        x2 = solve_lu(np.diag(sig) + X @ X.T, b)
        t2 = time.time() - t
        print(f"    N={N:5d}: Woodbury {1e3 * t1:6.2f} ms, dense LU {1e3 * t2:7.1f} ms, largest difference {np.abs(x1 - x2).max():.1e}")
    r = np.random.default_rng(1)
    print("  Woodbury cost at D = 200 as N grows (a pure O(D^3) cost, as the book states, would stay flat):")
    X = r.standard_normal((16000, 200))
    sig = 1.0 + r.random(16000)
    b = r.standard_normal(16000)
    prev = None
    for N in (2000, 4000, 8000, 16000):
        t1 = float("inf")
        for _ in range(5):
            t = time.time()
            woodbury_solve(sig[:N], X[:N], b[:N])
            t1 = min(t1, time.time() - t)
        print(f"    N={N:5d}: {1e3 * t1:7.2f} ms" + (f" ({t1 / prev:.1f}x)" if prev else ""))
        prev = t1

    # --- Part 8 ---
    print("Part 8: whitening, least norm, faces")
    F16 = F.reshape(400, 64, 64)
    F16 = 0.25 * (F16[:, ::2, ::2] + F16[:, 1::2, ::2] + F16[:, ::2, 1::2] + F16[:, 1::2, 1::2])
    F16 = 0.25 * (F16[:, ::2, ::2] + F16[:, 1::2, ::2] + F16[:, ::2, 1::2] + F16[:, 1::2, 1::2]).reshape(400, -1)
    W_pca, W_zca, mu = whitening_matrices(F16)
    Xc = F16 - mu
    Yp, Yz = Xc @ W_pca.T, Xc @ W_zca.T

    def corr(a, b):
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        return float(np.mean(np.sum(a * b, axis=1) / np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))))

    print(f"  whitening 16 x 16 faces: distance to the data PCA {np.linalg.norm(Yp - Xc):.1f}, ZCA {np.linalg.norm(Yz - Xc):.1f}; mean correlation with the face PCA {corr(Yp, Xc):.3f}, ZCA {corr(Yz, Xc):.3f}")
    print(f"  least-norm line (Figure 7.12a): x1 + x2 = 2 gives {np.round(least_norm(np.array([[1.0, 1.0]]), np.array([2.0])), 6).tolist()}")
    ranks = [5, 10, 20, 50, 100]
    acc = {K: [] for K in ranks}
    mn, nn = [], []
    n_splits = 5
    for seed in range(n_splits):
        rng = np.random.default_rng(seed)
        tr = np.ones(400, bool)
        for p in range(40):
            tr[rng.choice(np.flatnonzero(fy == p), 2, replace=False)] = False
        a = face_accuracy_vs_rank(F[tr], fy[tr], F[~tr], fy[~tr], ranks)
        for K in ranks:
            acc[K].append(a[K])
        m = F[tr].mean(axis=0)
        Y = (fy[tr][:, None] == np.arange(40)).astype(float)
        W = least_norm(F[tr] - m, Y - Y.mean(axis=0))
        mn.append(np.mean(np.argmax((F[~tr] - m) @ W, axis=1) == fy[~tr]))
        d2 = ((F[~tr][:, None, :] - F[tr][None, :, :]) ** 2).sum(axis=-1)
        nn.append(np.mean(fy[tr][np.argmin(d2, axis=1)] == fy[~tr]))
    print(f"  40 people from 320 faces, mean test accuracy over {n_splits} splits:")
    print("    " + ", ".join(f"K={K} {np.mean(acc[K]):.3f}" for K in ranks))
    print(f"    least-norm on all 4096 pixels {np.mean(mn):.3f}, 1-nearest neighbor on pixels {np.mean(nn):.3f}")
    print(f"done in {time.time() - t_start:.1f} s")


if __name__ == "__main__":
    main()

