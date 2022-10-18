'''q3'''
import numpy as np
from cla_utils import householder, householder_qr, hessenberg
from numpy import random
from scipy.sparse import diags

# redefine sign, so that it always has norm 1
def sign(x1):
    if x1 != 0:
        return np.sign(x1)
    else:
        return 1.0 

def qr_factor_tri(T):
    '''
    QR factorize a tridiagonal matrix T. The algorithm works inplace, so 
    that T is transformed into R.

    Parameters
    ----------
    T : mxm numpy array
        tridiagonal matrix

    Returns
    -------
    v : (m-1)x2 numpy array
        2x2 Householder reflections used to transform T to R

    :param T
    '''
    m = T.shape[0] 
    v = np.zeros([m-1, 2])

    for k in range(m-1):
        x = T[k:k+2, k]

        #construct e1 - the same length as x
        e1 = np.zeros(2)
        e1[0] = 1

        # rest of algorithm
        v[k] = sign(x[0]) * np.linalg.norm(x) * e1 + x  
        v[k] = v[k]/np.linalg.norm(v[k])

        T[k:k+2, k:k+3] -= 2 * np.outer(v[k], v[k]) @ T[k:k+2, k:k+3]

    return v


def qr_alg_tri(T, shift=False, maxit=1000, return_err=False):
    '''
    Apply the QR algorithm to a tridiagonal matrix T. The algorithm works 
    inplace to transform T iteratively into a diagonal matrix. Note that
    it currently only works for reals.

    Parameters
    ----------
    T : mxm numpy array
        tridiagonal matrix
    shift : Boolean
        if shift, apply the Wilkinson shift (defined below)
    maxit : int
        maximum number of iterations
    return_err : Boolean
        if return_err, return a vector of the errors at each iteration

    Returns
    -------
    T_err : numpy array
        if return_err, return a vector of the errors at each iteration
    '''

    m = T.shape[0]
    k = 0
    if return_err:
        T_err = np.array([])

    while True:
        if shift:
            mu = wilk_shift(T)
            T[np.diag_indices(m)] -= mu
        v = qr_factor_tri(T)
        T[:, :] = RQ_diags(T, v)
        if shift:
            T[np.diag_indices(m)] += mu

        # update k
        k += 1
        if k >= maxit:
            break

        # update the error r
        r = np.abs(T[-1, -2])
        if return_err:
            T_err = np.append(T_err, r)
        if r < 1.0e-12:
            break
    if return_err:
        return T_err

def RQ_diags(T, v):
    '''
    Given T = R and v from qr_factor_tri, constructs RQ_diags. This uses the 
    symmetry and the known tridiagonal form of the solution.

    Parameters
    ----------
    T : mxm numpy array
        Output of qr_factor_tri, an upper triangular matrix.
    v : (m-1)x2 numpy array
        2x2 Householder reflections as outputted by qr_factor_tri


    Returns
    -------
    RQ : mxm numpy array
        tridiagonal matrix equivalent to R @ Q
    '''
    m = T.shape[0]
    Q_diag = np.zeros(m)

    Q_diag[0] = 1 - 2 * (v[0,0]**2)
    Q_diag[-1] = 1 - 2 * (v[-1,-1]**2)
    Q_diag[1:-1] = (1 - 2 * (v[1:,0]**2))* (1 - 2 * (v[:-1, 1]**2))

    Q_subdiag = -2*v[:, 0]*v[:, 1] # don't have to preallocate, ch dims

    # check sub or super diag of R
    RQ_diag = np.diag(T)*Q_diag
    RQ_diag[:-1] += np.diag(T, k=1)*Q_subdiag
    RQ_sudiag = np.diag(T)[1:] * Q_subdiag

    # return RQ_diag, RQ_sudiag
    return diags(
        [RQ_diag, RQ_sudiag, RQ_sudiag], [0, -1, 1]).toarray()

def wilk_shift(T):
    '''Wilkson shift.

    Parameters
    ----------
    T : mxm numpy array
        
    Returns
    -------
    mu : float
        Wilkonson shift
    '''
    d = (T[-2, -2] - T[-1, -1])/2
    b = T[-1, -2]
    mu = (T[-1, -1] - 
        (np.sign(d)*b**2) / (abs(d)+np.sqrt(d**2+b**2)))
    print(mu)
    return mu

def construct_A():
    '''Constructs the 5x5 A as specified by the question
    '''
    A = np.zeros([5, 5])
    for i in range(5):
        for j in range(5):
            A[i, j] = 1/(i + j + 3)
    return A