'''Plot for 3d'''
from cla_utils import hessenberg
from q3 import construct_A, qr_alg_tri
from matplotlib import pyplot as plt

A = construct_A()
hessenberg(A)
T_err = qr_alg_tri(A, return_err=True)
print('Reduced A: ', A)
plt.semilogy(T_err, 'o-')
plt.xlabel('iterations')
plt.ylabel('error')
plt.show()