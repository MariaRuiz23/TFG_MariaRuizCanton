import numpy as np
from scipy.interpolate import interp1d
from numpy import linalg as la
import sys
import matplotlib.pyplot as plt

def restriccion(y):  #y son los puntos
    v = []
    n = len(y)+1
    k = int(n/2)
    for j in range(1,k):
        v.append((y[j-1]+2*y[j]+y[j+1])/4)
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

def interpolacion(y): #y son los puntos
    v = np.zeros(2*(len(y)+1)-1)
    n = len(y) + 1
    k = int(n / 2)
    for j in range(len(y)):
        v[2*j+1] = y[j]
    v[0] = y[0]/2
    v[2*(len(y)+1)-2] = y[len(y)-1]/2
    for i in range(1,int((len(v)-3)/2)):
        v[2*i] = (v[2*i-1] + v[2*i+1])/2
    return v

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


def multigridvciclo(n,f,k,x0,E,norma): #n:num subinterv PAR,k:num iteraciones previas
    h = np.pi/n
    malla = []
    fh = []
    for j in range(1,n):
        malla.append(j*h)
    Ah = (2*np.eye(n-1) + np.diagflat(-1*np.ones((1,n-2)),k=-1) + np.diagflat(-1*np.ones((1,n-2)),k=1))/h**2
    for x in malla:
        fh.append(f(x))
    fh = np.array(fh)
    #Paso 1, primeras iteraciones con Sor en la malla gruesa
    wh = Sor(Ah,fh,x0,norma,E,k,2/3)
    if wh[1] == k:
        print('Aumenta el número de iteraciones previas')
        sys.exit()
    wh = wh[0]
    rh = fh - np.dot(Ah, wh)
    #Paso 2, restricción (descenso de malla gruesa a fina) (sobre los residuos)
    r2h = restriccion(rh)
    r2h = np.array(r2h,dtype=float)
    contador = 1
    while len(r2h) > 2:
        # Aplicar la función de restricción
        r2h = restriccion(r2h)
        contador = contador + 1
    for i in range(contador):
        Ah = restriccionmatrices(Ah)
    print('num de mallas', contador)
    #Paso 3, cálculo de e_{2^jh}
    e2h = np.linalg.solve(Ah, r2h)
    #Paso 4, interpolación (de malla fina a gruesa)  (sobre los errores)
    while len(e2h) < len(x0):
        # Aplicar la función de interpolación
        e2h = interpolacion(e2h)
    #Paso 5, cálculo de la solución final
    solfinal = wh + e2h
    return solfinal



#EJEMPLO

def f(x):
    return np.sin(x)

x01 = np.zeros(7)
x02 = np.zeros(15)
x03 = np.zeros(31)
x11 = np.linspace(np.pi/8,7*np.pi/8,1000)
x22 = np.linspace(np.pi/16,15*np.pi/16,1000)
x33 = np.linspace(np.pi/32,31*np.pi/32,1000)
x1 = np.linspace(np.pi/8, 7*np.pi/8,7)
x2 = np.linspace(np.pi/16, 15*np.pi/16,15)
x3 = np.linspace(np.pi/32, 31*np.pi/32,31)

result1 = multigridvciclo(8,f,2000,x01,10**(-7),np.inf)
result2 = multigridvciclo(16,f,2000,x02,10**(-7),np.inf)
result3 = multigridvciclo(32,f,3000,x03,10**(-7),np.inf)

#Aproximaciones vs solución exacta
#n = 8
"""plt.plot(x1,result1, x11, f(x11))
plt.title('Aproximación 1')
plt.legend(('Solución aproximada','sen(x)'),loc='upper right')
plt.xlabel('x')
plt.ylabel('sen(x)')
plt.show()"""

#n = 16
"""plt.plot(x2,result2, x22, f(x22))
plt.title('Aproximación 2')
plt.legend(('Solución aproximada','sen(x)'),loc='upper right')
plt.xlabel('x')
plt.ylabel('sen(x)')
plt.show()"""

#n = 24
"""plt.plot(x3,result3, x33, f(x33))
plt.title('Aproximación 3')
plt.legend(('Solución aproximada','sen(x)'),loc='upper right')
plt.xlabel('x')
plt.ylabel('sen(x)')
plt.show()"""


#ERRORES
errores1 = []
for i in range(len(result1)):
    errores1.append(result1[i]-f(x1[i]))

errores2 = []
for i in range(len(result2)):
    errores2.append(result2[i]-f(x2[i]))

errores3 = []
for i in range(len(result3)):
    errores3.append(result3[i]-f(x3[i]))

#Representación errores
"""plt.plot(x1,errores1, x2,errores2, x3, errores3)
plt.title('Errores cometidos')
plt.legend(('$\Omega_{h1}$','$\Omega_{h2}$','$\Omega_{h3}$'),loc='upper right')
plt.xlabel('x')
plt.ylabel('Errores')
plt.show()"""




