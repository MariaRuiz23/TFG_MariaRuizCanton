import numpy as np
from numpy import linalg as la
import scipy.linalg as la
import sys


A = np.array([[4,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0],
              [-1,5,-1,-1,-1,0,0,0,0,0,0,0,0,0,0],
              [-1,-1,5,0,-1,-1,0,0,0,0,0,0,0,0,0],
              [0,-1,0,5,-1,0,-1,-1,0,0,0,0,0,0,0],
              [0,-1,-1,-1,6,-1,0,-1,-1,0,0,0,0,0,0],
              [0,0,-1,0,-1,5,0,0,-1,-1,0,0,0,0,0],
              [0,0,0,-1,0,0,5,-1,0,0,-1,-1,0,0,0],
              [0,0,0,-1,-1,0,-1,6,-1,0,0,-1,-1,0,0],
              [0,0,0,0,-1,-1,0,-1,6,-1,0,0,-1,-1,0],
              [0,0,0,0,0,-1,0,0,-1,5,0,0,0,-1,-1],
              [0,0,0,0,0,0,-1,0,0,0,4,-1,0,0,0],
              [0,0,0,0,0,0,-1,-1,0,0,-1,5,-1,0,0],
              [0,0,0,0,0,0,0,-1,-1,0,0,-1,5,-1,0],
              [0,0,0,0,0,0,0,0,-1,-1,0,0,-1,5,-1],
              [0,0,0,0,0,0,0,0,0,-1,0,0,0,-1,4]])

b = np.zeros(15)
for i in range(10,15):
    b[i] = 1

def Jacobi(A,b,x0,E,norma,n):
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


def Gauss_seidel(A,b,x0,E,norma,n):
    D = np.diag(np.diag(A))
    L = -np.tril(A-D)
    U = -np.triu(A-D)
    M = D - L
    N = U
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
        x1 = c + np.dot(B,x0)
        if la.norm(x1-x0,norma)<error:
            return [x1,i,ro]
        i = i+1
        x0 = x1.copy()


def restriccion(y):  #y son los puntos
    v = []
    n = len(y)+1
    k = int(n/2)
    for j in range(1,k):
        v.append((y[j-1]+2*y[j]+y[j+1])/4)
    return v


def interpolacion(y): #y son los puntos
    v = np.zeros(2*(len(y)+1)-1)
    n = len(y) + 1
    k = int(n / 2)
    for j in range(len(y)):
        v[2*j+1] = y[j]
    v[0] = y[0]/2
    v[2*(len(y)+1)-2] = y[len(y)-1]/2
    for i in range(1,len(v)-len(y)-1):
        v[2*i] = (v[2*i-1] + v[2*i+1])/2
    return v


def restriccionmatrices(Ah):
    n = len(Ah)+1
    Ih2h = np.zeros((int(n / 2 - 1), n - 1))
    for i in range(0, int(n / 2 - 1)):
        Ih2h[i][2 * i] = 1
        Ih2h[i][2 * i + 2] = 1
        Ih2h[i][2 * i + 1] = 2
    A2hf = np.dot(Ih2h, Ah)
    A2h = np.zeros((int(n / 2 - 1), int(n / 2 - 1)))
    for z in range(len(A2hf)):
        for x in range(int(n / 2 - 1)):
            A2h[z][x] = A2hf[z][2 * x + 1]
    return A2h


def multigrid2mallas(A,b,k,x0,E,norma): #n:num subinterv PAR,k:num iteraciones previas
    n = len(A)+1
    #Paso 1, primeras iteraciones con Sor en la malla gruesa
    wh = Sor(A,b,x0,norma,E,k,2/3)
    wh = wh[0]
    rh = b - np.dot(A, wh)
    #Paso 2, cálculo de r_{2h} y A_{2h}
    r2h = restriccion(rh)
    r2h = np.array(r2h,dtype=float)
    Ih2h = np.zeros((int(n / 2 - 1), n - 1))
    for i in range(0, int(n / 2 - 1)):
        Ih2h[i][2 * i] = 1
        Ih2h[i][2 * i + 2] = 1
        Ih2h[i][2 * i + 1] = 2
    A2hf = np.dot(Ih2h,A)
    A2h = np.zeros((int(n / 2 - 1), int(n / 2 - 1)))
    for z in range(len(A2hf)):
        for x in range(int(n / 2 - 1)):
            A2h[z][x] = A2hf[z][2 * x + 1]
    #Paso 3, se resuelve la ecuación en la malla fina  A_{2h} e_{2h} = r_{2h}
    e2h = np.linalg.solve(A2h, r2h)
    #Paso 4, mediante interpolación se traslada e2h a la malla gruesa
    eh = interpolacion(e2h)
    solfinal = wh + eh
    return solfinal


def multigridvciclo(A,b,k,x0,E,norma): #n:num subinterv PAR,k:num iteraciones previas
    n = len(A) + 1
    #Paso 1, primeras iteraciones con Sor en la malla gruesa
    wh = Sor(A,b,x0,norma,E,k,2/3)
    wh = wh[0]
    rh = b - np.dot(A, wh)
    #Paso 2, restricción (descenso de malla gruesa a fina) (sobre los residuos)
    r2h = restriccion(rh)
    r2h = np.array(r2h,dtype=float)
    contador = 1
    while len(r2h) > 2:
        # Aplicar la función de restricción
        r2h = restriccion(r2h)
        contador = contador + 1
    for i in range(contador):
        A = restriccionmatrices(A)
    print('num de mallas', contador)
    #Paso 3, cálculo de e_{2^jh}
    e2h = np.linalg.solve(A, r2h)
    #Paso 4, interpolación (de malla fina a gruesa)  (sobre los errores)
    while len(e2h) < len(x0):
        # Aplicar la función de interpolación
        e2h = interpolacion(e2h)
    #Paso 5, cálculo de la solución final
    solfinal = wh + e2h
    return solfinal



x0 = np.zeros(15)

print('jacobi',Jacobi(A,b,x0,10**(-6),np.inf,100))

print('gauss-seidel',Gauss_seidel(A,b,x0,10**(-6),np.inf,100))

print('sor',Sor(A,b,x0,np.inf,10**(-6),100,1.293030505283))

print('multigrid 2 mallas',multigrid2mallas(A,b,10,x0,10**(-6),np.inf))

print('multigrid v-ciclo',multigridvciclo(A,b,10,x0,10**(-6),np.inf))

