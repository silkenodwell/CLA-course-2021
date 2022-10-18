'''Plots for 3e, 3f, 3g '''
import q3
from cla_utils import hessenberg, pure_QR
import numpy as np
from numpy import random
from matplotlib import pyplot as plt

# construct a symmetric matrix A of size m, and reduce to tridiagonal
def get_tri(m):
    A = random.randn(m, m)
    A = 0.5*(A + A.T) # make A symmetric
    hessenberg(A) # reduce A to tridiag
    return A

# construct the 5x5 A as specified, and reduce to tridiagonal
def get_tri_A5():
    A = q3.construct_A()
    hessenberg(A)
    return A

# construct A = D+O and reduce to tridiagonal
def get_DO():
    D = np.diag(np.arange(15, 0, -1))
    O = np.ones([15, 15])
    A = D+O
    hessenberg(A)
    return A

def ev_tri(A, return_err=False, shift=False):
    ''' Apply the qr_alg_tri with deflation in order to find the 
    eigenvalues.

    Parameters
    ----------
    A : mxm numpy array
        tridiagonal matrix
    return_err : Boolean
        if True, return the error
    shift : Boolean
        if True, apply the shifted qr_alg_tri

    Returns
    -------
    ev : mx1 numpy array
        eigenvalues of A
    err : numpy array
        return if return_err

    '''
    m = A.shape[0]
    ev = np.zeros(m)

    if return_err:
        err = q3.qr_alg_tri(A, return_err=True, shift=shift)
    else:
        q3.qr_alg_tri(A, shift=shift)

    ev[-1] = A[-1, -1]

    for k in reversed(range(1, m-1)):
        if return_err:
            err = np.append(
                err, q3.qr_alg_tri(A[:k+1, :k+1], 
                return_err=True, shift=shift))
        else:
            q3.qr_alg_tri(A[:k+1, :k+1], shift=shift)
        print(A[:k+1, :k+1].shape)
        ev[k] = A[k, k]

    ev[0] = A[0, 0]
    if return_err:
        return ev, err
    else:
        return ev


def plot_err(A, shifted=False):
    '''Plot the error for the pure QR algorithm, as well as
    qr_alg_tri with deflation

    Parameters
    ----------
    A : mxm numpy array
        tridiagonal matrix
    shifted : Boolean
        if True, apply the shifted qr_alg_tri
    '''
    A1 = 1.0*A
    
    _, T_err = ev_tri(A, return_err=True, shift=shifted)


    _, pure_qr_err = pure_QR(
        A1, maxit=1000, tol=1.0e-12, return_err=True)
    print(pure_qr_err)

    if shifted:
        lab = 'qr_alg_tri shifted, with deflation'
    else:
        lab = 'qr_alg_tri+deflation'

    plt.semilogy(T_err, 'b.', label=lab)
    plt.semilogy(pure_qr_err, 'r-', label='pure QR')
    plt.xlabel('iterations')
    plt.ylim(bottom=1.0e-13)
    plt.ylabel('error')
    plt.legend()
    plt.show()
