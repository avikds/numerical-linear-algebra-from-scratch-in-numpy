# Numerical Linear Algebra from Scratch in NumPy

Write your own np.linalg, without calling np.linalg, and find out why the textbook formula is so often the wrong thing to type. Murphy's Probabilistic Machine Learning states the matrix decompositions that the rest of machine learning stands on; here you implement every one of them and measure what they do to your digits. Start with LU, partial pivoting and the book's worked system, then Cholesky for log determinants that do not underflow and for Gaussian sampling. Compare Gram-Schmidt with Householder QR on a Hilbert matrix, where the first loses all orthogonality while the last stays at machine precision, and fit polynomials until the normal equations collapse while QR keeps going. Measure the power method's convergence against the eigenvalue ratio the book predicts, then extend it with deflation and with orthogonal iteration. Build the SVD twice: the book's route through the eigenvectors of A^T A, which cannot see singular values below about 1e-8, and one-sided Jacobi, which can. Use it for pseudo-inverses, null spaces and condition numbers. Compress a 200 x 320 photo to the book's 10,420 numbers and check the truncation error, including the book's misprinted error equation. Finish with Schur complements, the Woodbury identity, whitening, least-norm solutions and a 40-person face classifier that runs entirely on code you wrote.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** forward_substitution
- [x] **2.** lu_no_pivot
- [x] **3.** lu_partial_pivot
- [x] **4.** solve_lu
- [x] **5.** cholesky_factor
- [x] **6.** logdet_chol
- [x] **7.** mvn_sample
- [x] **8.** gram_schmidt_qr
- [x] **9.** householder_qr
- [x] **10.** lstsq_normal
- [x] **11.** lstsq_qr
- [x] **12.** power_method
- [x] **13.** top_k_deflation
- [x] **14.** orthogonal_iteration
- [x] **15.** svd_via_eig
- [x] **16.** svd_jacobi
- [x] **17.** pinv_svd
- [x] **18.** rank_and_null_space
- [x] **19.** condition_number
- [x] **20.** least_squares_study
- [x] **21.** truncated_svd
- [x] **22.** truncation_error
- [x] **23.** spectrum_vs_shuffled
- [x] **24.** partitioned_inverse
- [x] **25.** woodbury_solve
- [x] **26.** rank_one_update
- [x] **27.** whitening_matrices
- [x] **28.** least_norm
- [x] **29.** ridge_dual
- [x] **30.** face_accuracy_vs_rank

## Results

```
Part 1: triangular systems and LU
  book's system (eq 7.214): x = [1.0, -2.0, -2.0]  (book: [1, -2, -2])
  tiny pivot, no pivoting: x = [0.0, 1.0], partial pivoting: x = [1.0, 1.0]
  random 300 x 300: relative error 5.9e-10 without pivoting, 2.4e-13 with; largest multiplier 8445 without, 1 with
Part 2: Cholesky
  0.1 I_100: det 1.0e-100, log det -230.2585
  0.1 I_400: det 0.0, log det -921.0340
  MVN sampling (eq 7.209): target cov [[2.0, 0.8], [0.8, 1.0]], empirical [[2.007, 0.798], [0.798, 0.999]]
Part 3: QR and least squares
  loss of orthogonality ||Q^T Q - I||_F on Hilbert matrices:
    n= 6 kappa 1.5e+07: classical GS 2.8e-04, modified GS 3.5e-10, Householder 2.0e-15
    n= 8 kappa 1.5e+10: classical GS 1.4e+00, modified GS 3.6e-07, Householder 1.6e-15
    n=10 kappa 1.6e+13: classical GS 3.5e+00, modified GS 5.2e-05, Householder 2.0e-15
    n=12 kappa 1.8e+16: classical GS 5.5e+00, modified GS 3.3e-01, Householder 2.3e-15
Part 4: power method
  measured error reduction per iteration 0.8000, |lambda_2 / lambda_1| = 0.8000
  faces Gram matrix (400 x 400), top k eigenvectors, 300 iterations (subspace error / orthogonality loss):
    k= 5: deflation 8.0e-16 / 3.8e-16, orthogonal iteration 9.8e-16 / 1.2e-15
    k=20: deflation 6.1e-04 / 2.7e-03, orthogonal iteration 1.7e-10 / 2.1e-15
    k=50: deflation 5.2e-04 / 3.7e-03, orthogonal iteration 2.8e-07 / 4.1e-15
Part 5: the SVD, twice
  true sigma    1.0e+00   1.0e-02   1.0e-04   1.0e-06   1.0e-08   1.0e-10   1.0e-12   1.0e-14
  via A^T A     1.0e+00   1.0e-02   1.0e-04   1.0e-06   9.0e-09   0.0e+00   3.9e-09   0.0e+00
  Jacobi        1.0e+00   1.0e-02   1.0e-04   1.0e-06   1.0e-08   1.0e-10   1.0e-12   1.0e-14
  kappa(0.1 I_100) = 1.000; eq 7.49: kappa = 2.000e+10, b1 + 1e-6 moves x from (1, 1) to (-9999.0, 10001.0)
  rank of [[0, 1], [0, 0]] is 1 though both of its eigenvalues are 0
  polynomial fits, relative coefficient error:
    degree  5 kappa 3.5e+03: normal 8.7e-11, QR 2.5e-13, pinv 1.8e-13
    degree  8 kappa 6.3e+05: normal 3.0e-06, QR 3.1e-12, pinv 1.0e-11
    degree 10 kappa 2.0e+07: normal 9.0e-03, QR 9.6e-10, pinv 3.3e-09
    degree 12 kappa 6.8e+08: normal 1.6e+00, QR 3.4e-08, pinv 5.1e-07
    degree 14 kappa 2.3e+10: normal inf, QR 3.6e-07, pinv 1.2e-04
    degree 16 kappa 8.2e+11: normal inf, QR 2.1e-05, pinv 1.3e-03
Part 6: truncated SVD on a 200 x 320 photo
  parameters and errors (full image has 64000 numbers, ||A||_F = 169.21):
    K=  2:   1042 numbers, direct ||A - A_K||_F  34.236, corrected formula  34.236, eq 7.201 as printed  271.65 (7.9x too large)
    K=  5:   2605 numbers, direct ||A - A_K||_F  25.187, corrected formula  25.187, eq 7.201 as printed  233.34 (9.3x too large)
    K= 20:  10420 numbers, direct ||A - A_K||_F  15.473, corrected formula  15.473, eq 7.201 as printed  159.28 (10.3x too large)
    K= 50:  26050 numbers, direct ||A - A_K||_F   9.364, corrected formula   9.364, eq 7.201 as printed   93.33 (10.0x too large)
  Figure 7.10, log sigma_i photo / shuffled: i=1 5.09 / 5.00, i=2 3.39 / 2.31, i=20 1.18 / 2.16, i=100 -0.19 / 1.64
  K for 99% of the energy: photo 16, shuffled 152
Part 7: block structure
  partitioned inverse (eq 7.109): ||M M^-1 - I||_F = 9.4e-13
  Woodbury (eq 7.123) against a dense LU solve, D = 5:
    N=  250: Woodbury   0.12 ms, dense LU    52.4 ms, largest difference 1.6e-14
    N=  500: Woodbury   0.12 ms, dense LU   355.7 ms, largest difference 2.3e-14
    N= 1000: Woodbury   0.30 ms, dense LU  1266.9 ms, largest difference 2.1e-14
  Woodbury cost at D = 200 as N grows (a pure O(D^3) cost, as the book states, would stay flat):
    N= 2000:   23.11 ms
    N= 4000:   25.14 ms (1.1x)
    N= 8000:   30.71 ms (1.2x)
    N=16000:   35.80 ms (1.2x)
Part 8: whitening, least norm, faces
  whitening 16 x 16 faces: distance to the data PCA 322.6, ZCA 303.1; mean correlation with the face PCA -0.004, ZCA 0.524
  least-norm line (Figure 7.12a): x1 + x2 = 2 gives [1.0, 1.0]
  40 people from 320 faces, mean test accuracy over 5 splits:
    K=5 0.295, K=10 0.688, K=20 0.872, K=50 0.940, K=100 0.955
    least-norm on all 4096 pixels 0.970, 1-nearest neighbor on pixels 0.938
done in 33.3 s
```
