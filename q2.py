# answers for q2
# solve T (p | q | y) = (u1 | u2 | b)

import numpy as np
from numpy import random
from cla_utils import solve_LUP
import q1
from scipy.sparse import diags

def construct_A(C1, m):
    '''Construct A with the form given in 2.b.
    
    Parameters
    ----------
    C1 : float
        parameter to construct A
    m : int
        size of A

    Returns
    -------
    A : mxm numpy array
        A as specified in 2.b.
    '''
    diag = 1 + (2 * C1)
    A = diags([-C1, diag, -C1], [-1, 0, 1], shape=(m, m)).toarray() 
    A[-1, 0], A[0, -1] = -C1, -C1
    return A

def LU_solve_tri_update2(C1, b):
    ''' Solve the system of equations Ax = b, where A is a tridiagonal matrix
    with a rank 2 update, as in question 2.b.

    Parameters
    ----------
    C1 : float
        parameter to construct A
    b : mx1 numpy array
        b in Ax=b

    Returns
    -------
    x : mx1 numpy array
        solution of Tx=b

    '''
    m = b.shape[0]

    # construct W
    W = np.zeros([m, 2])
    W[-1, 0], W[0, 1] = -C1, -C1

    # solve T(P|y) = (W|b)
    diag = 1 + (2 * C1)
    y = q1.LU_tridiag_solve(diag, -C1, np.column_stack([W, b]) )
    P = y[:, :2]
    y = y[:, 2]

    # construct Minv
    Minv = np.array([[1 + P[-1, 1], -P[0, 1]], [-P[-1, 0], 1 + P[0, 0]]])
    # divide by det(M)
    Minv /= ((Minv[0, 0] * Minv[1, 1]) - (Minv[0, 1] * Minv[1, 0]))

    x = y - P @ (Minv @ np.array([y[0], y[-1]]) )
    return x



# setup to have the variables in the working space
m = 2000
C1 = random.randn(1)
A = construct_A(C1, m)
A0 = 1.0 * A
b = random.randn(m)

def timeable_LU_solve_tri():
    LU_solve_tri(C1, b)

def timeable_LU_solve():
    solve_LUP(A, b)

def time_solveQ():
    import timeit

    print("Timing for tridiag solve")
    print(timeit.Timer(timeable_LU_solve_tri).timeit(number = 1))
    print("Timing for general solver")
    print(timeit.Timer(timeable_LU_solve).timeit(number = 1))