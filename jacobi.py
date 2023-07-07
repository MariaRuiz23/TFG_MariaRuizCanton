import numpy as np
from numpy import linalg as la
#import scipy.linalg as la

def Jacobi(A,b,x0,E,norma,n): #x0:punto semilla, E:error o tolerancia, norma: tipo de norma, n:num max iteraciones
    D = np.diag(np.diag(A))
    L = -np.tril(A-D)
    U = -np.triu(A-D)
    M = D
    N = L + U
    B = np.dot(la.inv(M),N)
    c = np.dot(la.inv(M),b)
    val = la.eig(B)[0]
    ro = max(abs(val))
    if ro>=1:
        print('El método no converge')
        return [[],0,ro]
    i=1
    while True:
        if i>=n:
            print('El método no converge en',n,'pasos')
            return [x0,n,ro]
        x1 = np.dot(B,x0) + c
        if la.norm(x1-x0,norma)<E:
            return [x1,i,ro]
        i = i+1
        x0 = x1.copy()









