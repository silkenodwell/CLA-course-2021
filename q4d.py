# q4d
from numpy import random
from scipy.sparse import csgraph
import numpy as np
from numpy import random
from cla_utils import operator_2_norm as o2n, solve_U
import q4
from matplotlib import pyplot as plt

m = 6
I = np.eye(m)

# define A = I + L, as specified in the question
def get_A():
    x = np.array(
        [ 0.40846802,  0.6773707 ,  1.48125888,  0.9212499 ,
         -0.05354195, 0.56436666])
    A = csgraph.laplacian(np.outer(x, x)) + I
    return A

A = get_A()
M = np.triu(A)
MinvA = solve_U(M, A)

def get_c(A):
    c = 1
    for i in range(5000):
        v = np.linspace(0.0001, 1000, 5000, endpoint=False)
        C = o2n(I - (MinvA/v[i]))
        if C < 1:
            print(i)
            c = v[i]
            break
    print('end')
    return c, C

c, C = get_c(A)
b = random.randn(m)

# apply the preconditioned GMRES 
M = c * np.triu(A)
apply_pc = q4.make_apply_pc(M)
y, nits, rnorms = q4.GMRES_precond(
   A, b, apply_pc, 1000, 0.001, return_residual_norms=True)

# plot the error and the error estimate
plt.semilogy(rnorms, 'bx', label='actual error')
plt.ylabel('error')
plt.xlabel('iterations')
j = np.arange(nits)
plt.semilogy(C**j, 'r', label='upper bound')
plt.legend()
plt.show()