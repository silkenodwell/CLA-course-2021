# q4
import numpy as np
from cla_utils import householder_ls, solve_U
from numpy import random

def make_apply_pc(M):
    def apply_pc(b):
        # assuming M is upper triangular
        return solve_U(M, b)
    return apply_pc

def GMRES_precond(A, b, apply_pc, maxit, tol, x0=None, return_residual_norms=False,
          return_residuals=False):
    """
    For a matrix A, solve Ax=b using the preconditioned GMRES algorithm.

    :param A: an mxm numpy array
    :param b: m dimensional numpy array
    :apply_pc : function taking b as in input and returning Minv * b
    :param maxit: integer, the maximum number of iterations
    :param tol: floating point number, the tolerance for termination
    :param x0: the initial guess (if not present, use b)
    :param return_residual_norms: logical
    :param return_residuals: logical

    :return x: an m dimensional numpy array, the solution
    :return nits: if converged, the number of iterations required, otherwise \
    equal to -1
    :return rnorms: nits dimensional numpy array containing the norms of \
    the residuals at each iteration
    :return r: mxnits dimensional numpy array, column k contains residual \
    at iteration k
    """
    m = A.shape[0]

    b_hat = apply_pc(b)
    b_e1 = np.linalg.norm(b_hat) * (np.eye(maxit+1)[:, 0])

    x = np.zeros(m, dtype = A.dtype)
    Q = np.zeros([m, maxit+1], dtype = A.dtype)
    H = np.zeros([maxit+1, maxit], dtype = A.dtype)
    nits = 0
    rnorms = np.zeros(maxit)
    r = np.zeros([m, maxit])

    if x0 is None:
        x0 = b
    x = x0

    Q[:, 0] = b_hat / np.linalg.norm(b_hat)

    for n in range(maxit):
        print(n)
        # Arnoldi with preconditioning
        v = apply_pc(A @ Q[:, n])

        H[:, n] = Q.conj().T @ v
        v -= Q[:, :n+1] @ H[:n+1, n]

        H[n+1, n] = np.linalg.norm(v)
        Q[:, n+1] = v / np.linalg.norm(v)

        # least squares solution
        y = householder_ls(H[:n+1, :n+1], b_e1[:n+1])
        # y = np.linalg.lstsq(H[:n+1, :n+1], b_e1[:n+1])[0]

        # xn = Qn * y
        x = Q[:, :n+1] @ y

        # update nits
        nits += 1

        # norm of residual
        r[:, n] = A @ x - b
        rnorms[n] = np.linalg.norm(r[:n+1, n])
        if rnorms[n] <= tol:
            break

    if rnorms[-1] > tol:
        nits = -1
    else:
        rnorms = rnorms[:nits]

    if return_residual_norms:
        return x, nits, rnorms
    else:
        return x, nits
