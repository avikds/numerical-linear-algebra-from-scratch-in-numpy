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

---

Built on Deep-ML.
