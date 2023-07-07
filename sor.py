import numpy as np
from numpy import linalg as la

def Sor(A,b,x0,norma,error,k,w):
    D = np.diag(np.diag(A))
    L = -np.tril(A - D)
    U = -np.triu(A - D)
    M = D - w*L
    N = (1-w)*D + w*U
    B = np.dot(la.inv(M), N)
    c = np.dot(la.inv(M), w*b)
    [val,vec] = la.eig(B) # valores propios y vectores propios
    ro = max(abs(val))
    if ro>=1:       #si el método no converge
        print('El método no converge')
        return [[],0,ro]
    i=1
    while True:
        if i>=k:
            print('El método no converge en',k,'pasos')
            return [0,k,ro]
        x1 = c + np.dot(B,x0)
        if la.norm(x1-x0,norma)<error:
            return [x1,i,ro]
        i = i+1
        x0 = x1.copy()


#EJEMPLO
A = np.array([[3,-1,0,0,0,-1],[-1,3,-1,0,-1,0],[0,-1,3,-1,0,0],[0,0,-1,3,-1,0],[0,-1,0,-1,3,-1],[-1,0,0,0,-1,3]],dtype=float)
b = np.array([1,2,3,4,5,6],dtype=float)
x0 = np.zeros(6)

print('w = 0')
[x1,i1,ro1] = Sor(A,b,x0,np.inf,10**(-6),100,0)
print('Radio espectral:',ro1)
print('Solución:',x1)
print('Número de iteraciones:',i1)

print()
print('w = 0.75')
[x2,i2,ro2] = Sor(A,b,x0,np.inf,10**(-6),100,0.75)
print('Radio espectral:',ro2)
print('Solución:',x2)
print('Número de iteraciones:',i2)

print()
print('w = 1.5')
[x3,i3,ro3] = Sor(A,b,x0,np.inf,10**(-6),100,1.5)
print('Radio espectral:',ro3)
print('Solución:',x3)
print('Número de iteraciones:',i3)

print()
print('w = 2.25')
[x4,i4,ro4] = Sor(A,b,x0,np.inf,10**(-6),100,2.25)
print('Radio espectral:',ro4)
print('Solución:',x4)
print('Número de iteraciones:',i4)