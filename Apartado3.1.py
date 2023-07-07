import numpy as np
from numpy import linalg as la
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style

def Jacobi(A,b,x0,E,norma,n):
    D = np.diag(np.diag(A))
    L = -np.tril(A-D)
    U = -np.triu(A-D)
    M = D
    N = L + U
    B = np.dot(la.inv(M),N)
    c = np.dot(la.inv(M),b)
    val = la.eig(B)[0]  #valores propios
    ro = max(abs(val))
    if ro>=1:       #si el método no converge
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



def Gauss_seidel(A,b,x0,E,norma,n):    #x0 es el punto semilla, E el error o tolerncia, n el num max de iter
    D = np.diag(np.diag(A))
    L = -np.tril(A-D)
    U = -np.triu(A-D)
    M = D - L
    N = U
    B = np.dot(la.inv(M),N)
    c = np.dot(la.inv(M),b)
    val = la.eig(B)[0]  #valores propios
    ro = max(abs(val))
    if ro>=1:       #si el método no converge
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
        if i>=k:
            print('El método no converge en',k,'pasos')
            return [0,k,ro]
        x1 = c + np.dot(B,x0)
        if la.norm(x1-x0,norma)<error:
            return [x1,i,ro]
        i = i+1
        x0 = x1.copy()


def adi(N,r,sigma,f,n,x0,norma,E):  #N num de puntos en Omega,  n num max pasos
    #Paso 1: construcción de las matrices H, V, SIGMA, (A = H+V+SIGMA)
    H = 2 * np.eye(N ** 2) + np.diagflat(-1 * np.ones((1, N ** 2 - 1)), k=-1) + np.diagflat(-1 * np.ones((1, N ** 2 - 1)), k=1)
    for k in range(1, N):
        H[N * k, N * k - 1] = 0
        H[N * k - 1, N * k] = 0
    V = 2 * np.eye(N ** 2) + np.diagflat(-1 * np.ones((1, N ** 2 - N)), k=-N) + np.diagflat(-1 * np.ones((1, N ** 2 - N)), k=N)
    h = 1 / (N + 1)
    SIGMA = sigma * h ** 2 * np.eye(N ** 2)
    #Paso 2: construcción de las matrices H1 y V1
    H1 = H + 0.5*SIGMA
    V1 = V + 0.5*SIGMA
    #Paso 3: construcción de la matriz T_r
    M1 = np.dot(la.inv(V1+r*np.eye(N**2)),r*np.eye(N**2)-H1)
    M2 = np.dot(la.inv(H1 + r*np.eye(N**2)),r*np.eye(N**2)-V1)
    Tr = np.dot(M1,M2)
    #Paso 4: construcción de la matriz g_r(b)
    puntos = []
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            puntos.append([j / (N + 1), i / (N + 1)])
    b = []
    for k in range(0, len(puntos)):
        b.append(h**2 * f(puntos[k][0], puntos[k][1]))
    N1 = np.eye(N**2) + np.dot(r*np.eye(N**2)-H1,la.inv(H1+r*np.eye(N**2)))
    N2 = np.dot(la.inv(V1+r*np.eye(N**2)),N1)
    gr = np.dot(N2,b)
    #Paso 5: iteraciones
    val = la.eig(Tr)[0]  #valores propios
    ro = max(abs(val))
    if ro>=1:       #si el método no converge
        print('El método no converge')
        return [[],0,ro]
    i=1
    while True:
        if i>=n:
            print('El método no converge en',n,'pasos')
            return [0,n,ro]
        x1 = np.dot(Tr,x0) + gr
        if la.norm(x1-x0,norma)<E:
            return [x1,i,ro]
        i = i+1
        x0 = x1.copy()


def f(x, y):
    return 2*np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y)


#CONSTRUCCIÓN DE LAS MATRICES DE ITERACIÓN Y VECTORES INDEPENDIENTES
N1 = 25
H1 = 2 * np.eye(N1 ** 2) + np.diagflat(-1 * np.ones((1, N1 ** 2 - 1)), k=-1) + np.diagflat(-1 * np.ones((1, N1 ** 2 - 1)), k=1)
for k in range(1, N1):
    H1[N1 * k, N1 * k - 1] = 0
    H1[N1 * k - 1, N1 * k] = 0
V1 = 2 * np.eye(N1 ** 2) + np.diagflat(-1 * np.ones((1, N1 ** 2 - N1)), k=-N1) + np.diagflat(-1 * np.ones((1, N1 ** 2 - N1)), k=N1)
A1 = H1 + V1
puntos1 = []
h1 = 1 / (N1 + 1)
for i in range(1, N1 + 1):
    for j in range(1, N1 + 1):
        puntos1.append([j / (N1 + 1), i / (N1 + 1)])
b1 = []
for k in range(0, len(puntos1)):
    b1.append(h1**2 * f(puntos1[k][0], puntos1[k][1]))


N2 = 50
H2 = 2 * np.eye(N2 ** 2) + np.diagflat(-1 * np.ones((1, N2 ** 2 - 1)), k=-1) + np.diagflat(-1 * np.ones((1, N2 ** 2 - 1)), k=1)
for k in range(1, N2):
    H2[N2 * k, N2 * k - 1] = 0
    H2[N2 * k - 1, N2 * k] = 0
V2 = 2 * np.eye(N2 ** 2) + np.diagflat(-1 * np.ones((1, N2 ** 2 - N2)), k=-N2) + np.diagflat(-1 * np.ones((1, N2 ** 2 - N2)), k=N2)
A2 = H2 + V2
puntos2 = []
h2 = 1 / (N2 + 1)
for i in range(1, N2 + 1):
    for j in range(1, N2 + 1):
        puntos2.append([j / (N2 + 1), i / (N2 + 1)])
b2 = []
for k in range(0, len(puntos2)):
    b2.append(h2**2 * f(puntos2[k][0], puntos2[k][1]))



#RESULTADOS PARA N1 = 25
x01 = np.zeros(N1**2)
#print('Jacobi:',Jacobi(A1,b1,x01,10**(-6),np.inf,2000))
#print('Gauss-Seidel:',Gauss_seidel(A1,b1,x01,10**(-6),np.inf,1000))
#print('ADI:',adi(N1,2,0,f,500,x01,np.inf,10**(-6)))

#RESULTADOS PARA N2 = 50
x02 = np.zeros(N2**2)
#print('Jacobi:',Jacobi(A2,b2,x02,10**(-6),np.inf,4000))
#print('Gauss-Seidel:',Gauss_seidel(A2,b2,x02,10**(-6),np.inf,4000))
#print('ADI:',adi(N2,2,0,f,1500,x02,np.inf,10**(-6)))