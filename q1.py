'''Code for 1d '''
import numpy as np

def LU_tridiag_dcd(c, d, m):
    '''LU algorithm for a tridiagonal matrix with c along the diagonal,
    d along the super- and subdiagonal.

    Parameters
    ----------
    c : float
        diagonal of the tridiagonal matrix
    d : float
        sup and super-diagonal of the tridiagonal matrix
    m : int
        size of the (square) tridiagonal matrix

    Returns
    -------
    l : (m-1)x1 numpy array
        subdiagonal of L
    u : mx1 numpy array
        diagonal of U
        
    '''
    l = np.zeros(m-1)
    u = c * np.ones(m)

    for k in range(m-1):
        l[k] = d/u[k]
        u[k+1] = c - (l[k] * d)
    return l, u            


def LU_tridiag_solve(c, d, b):
    ''' Solve the system of equations Tx = b, where T is a tridiagonal matrix,
    with c along the diagonal, d along the super- and subdiagonal.

    Parameters
    ----------
    c : float
        diagonal of the tridiagonal matrix
    d : float
        sup and super-diagonal of the tridiagonal matrix
    b : mx1 numpy array
        b in Tx=b

    Returns
    -------
    x : mx1 numpy array
        solution of Tx=b

    '''
    # LU decomposition and forwards substituion
    m = b.shape[0]
    l = np.zeros(m-1)
    u = c * np.ones(m)

    y = np.zeros(b.shape)
    x = np.zeros(b.shape)

    y[0] = b[0]

    for k in range(m-1):
        l[k] = d/u[k]
        u[k+1] = c - (l[k] * d)
        y[k+1] = b[k+1] - l[k] * y[k]

    # backwards solve Uy = b
    x[m-1] = y[m-1]/u[m-1]
    for k in reversed(range(m-1)):
        x[k] = (y[k] - d*x[k+1]) / u[k]

    return x

        



