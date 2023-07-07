import numpy as np
from numpy import linalg as la
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style

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


#EJEMPLO

def f(x, y):
    return 2*y - 2*y**2 + 2*x - 2*x**2 + x*y - x**2*y - x*y**2 + x**2*y**2

N1 = 3
N2 = 6
N3 = 9
x01 = np.zeros(N1**2)
x02 = np.zeros(N2**2)
x03 = np.zeros(N3**2)
result1 = adi(N1,2,0,f,400,x01,np.inf,10**(-6))[0]
result2 = adi(N2,2,0,f,400,x02,np.inf,10**(-6))[0]
result3 = adi(N3,2,0,f,400,x03,np.inf,10**(-6))[0]


def solexacta(x,y):
    return x*y - x**2*y - x*y**2 + x**2*y**2


x1 = np.array([0.25,0.5,0.75,0.25,0.5,0.75,0.25,0.5,0.75])
y1 = np.array([0.25,0.25,0.25,0.5,0.5,0.5,0.75,0.75,0.75])

#Representación de la primera malla
"""plt.scatter(x1,y1)
plt.title("$\Omega_{h1}$")
plt.xlabel("X")
plt.ylabel("Y")

plt.show()"""

#Aproximación vs solución exacta N=3
"""fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

curve1 = ax.plot(x1, y1,result1)
curve2 = ax.plot(x1, y1, np.array(solexacta(x1,y1)))

ax.legend([curve1[0], curve2[0]], ['Solución aproximada', 'Solución exacta'])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Aproximación con N=3')

plt.show()"""



x2 = np.array([1/7, 2/7, 3/7, 4/7, 5/7, 6/7,1/7, 2/7, 3/7, 4/7, 5/7, 6/7,1/7, 2/7, 3/7, 4/7, 5/7, 6/7,1/7, 2/7, 3/7, 4/7, 5/7, 6/7,1/7, 2/7, 3/7, 4/7, 5/7, 6/7,1/7, 2/7, 3/7, 4/7, 5/7, 6/7])
y2 = np.array([1/7,1/7,1/7,1/7,1/7,1/7,2/7,2/7,2/7,2/7,2/7,2/7,3/7,3/7,3/7,3/7,3/7,3/7,4/7,4/7,4/7,4/7,4/7,4/7,5/7,5/7,5/7,5/7,5/7,5/7,6/7,6/7,6/7,6/7,6/7,6/7])

#Representación de la segunda malla
"""plt.scatter(x2,y2)
plt.title("$\Omega_{h2}$")
plt.xlabel("X")
plt.ylabel("Y")

plt.show()"""

#Aproximación vs solución exacta N=6
"""fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

curve1 = ax.plot(x2, y2, result2)
curve2 = ax.plot(x2, y2, np.array(solexacta(x2,y2)))

ax.legend([curve1[0], curve2[0]], ['Solución aproximada', 'Solución exacta'])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Aproximación con N=6')

plt.show()"""



x3 = np.array([1/10,2/10,3/10,4/10,5/10,6/10,7/10,8/10,9/10,1/10,2/10,3/10,4/10,5/10,6/10,7/10,8/10,9/10,1/10,2/10,3/10,4/10,5/10,6/10,7/10,8/10,9/10,1/10,2/10,3/10,4/10,5/10,6/10,7/10,8/10,9/10,1/10,2/10,3/10,4/10,5/10,6/10,7/10,8/10,9/10,1/10,2/10,3/10,4/10,5/10,6/10,7/10,8/10,9/10,1/10,2/10,3/10,4/10,5/10,6/10,7/10,8/10,9/10,1/10,2/10,3/10,4/10,5/10,6/10,7/10,8/10,9/10,1/10,2/10,3/10,4/10,5/10,6/10,7/10,8/10,9/10])
y3 = np.array([1/10,1/10,1/10,1/10,1/10,1/10,1/10,1/10,1/10,2/10,2/10,2/10,2/10,2/10,2/10,2/10,2/10,2/10,3/10,3/10,3/10,3/10,3/10,3/10,3/10,3/10,3/10,4/10,4/10,4/10,4/10,4/10,4/10,4/10,4/10,4/10,5/10,5/10,5/10,5/10,5/10,5/10,5/10,5/10,5/10,6/10,6/10,6/10,6/10,6/10,6/10,6/10,6/10,6/10,7/10,7/10,7/10,7/10,7/10,7/10,7/10,7/10,7/10,8/10,8/10,8/10,8/10,8/10,8/10,8/10,8/10,8/10,9/10,9/10,9/10,9/10,9/10,9/10,9/10,9/10,9/10])

#Representación de la tercera malla
"""plt.scatter(x3,y3)
plt.title("$\Omega_{h3}$")
plt.xlabel("X")
plt.ylabel("Y")

plt.show()"""

#Aproximación vs solución exacta N=9
"""
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

curve1 = ax.plot(x3, y3, result3)
curve2 = ax.plot(x3, y3, np.array(solexacta(x3,y3)))

ax.legend([curve1[0], curve2[0]], ['Solución aproximada', 'Solución exacta'])

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Aproximación con N=9')

plt.show()"""



#Gráfico solución exacta

"""
x = np.linspace(0, 1, 200)
y = np.linspace(0, 1, 200)
X, Y = np.meshgrid(x, y)
Z = solexacta(X,Y)  

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()"""



#ERRORES

#N = 3
errores1 = []
sol1 = []
for i in range(len(x1)):
    sol1.append(solexacta(x1[i],y1[i]))
for j in range(len(sol1)):
    errores1.append(result1[j]-sol1[j])

y11 = []
y12 = []
y13 = []
for k in range(3):
    y11.append(errores1[3*k])
    y12.append(errores1[3 * k +1])
    y13.append(errores1[3 * k + 2])

sum11 = 0
for n in range(3):
    sum11 = sum11 + y11[n]

sum12 = 0
for n in range(3):
    sum12 = sum12 + y12[n]

sum13 = 0
for n in range(3):
    sum13 = sum13 + y13[n]


#Representación errores N=3
"""plt.scatter([0.25,0.5,0.75],[sum11/3,sum12/3,sum13/3])
plt.title("Media errores con N=3")
plt.xlabel("X")

plt.show()"""


#N = 6
errores2 = []
sol2 = []
for i in range(len(x2)):
    sol2.append(solexacta(x2[i],y2[i]))
for j in range(len(sol2)):
    errores2.append(np.abs(result2[j]-sol2[j]))

sum21 = 0
for n in range(6):
    sum21 = sum21 + errores2[6*n]

sum22 = 0
for n in range(6):
    sum22 = sum22 + errores2[6*n+1]

sum23 = 0
for n in range(6):
    sum23 = sum23 + errores2[6*n+2]

sum24 = 0
for n in range(6):
    sum24 = sum24 + errores2[6*n+3]

sum25 = 0
for n in range(6):
    sum25 = sum25 + errores2[6*n+4]

sum26 = 0
for n in range(6):
    sum26 = sum26 + errores2[6*n+5]

#Representación errores N=6
"""plt.scatter([1/7,2/7,3/7,4/7,5/7,6/7],[sum21/6,sum22/6,sum23/6,sum24/6,sum25/6,sum26/6])
plt.title("Media errores con N=6")
plt.xlabel("X")

plt.show()
"""


#N = 9
errores3 = []
sol3 = []
for i in range(len(x3)):
    sol3.append(solexacta(x3[i],y3[i]))
for j in range(len(sol3)):
    errores3.append(np.abs(result3[j]-sol3[j]))

sum31 = 0
for n in range(9):
    sum31 = sum31 + errores3[9*n]

sum32 = 0
for n in range(9):
    sum32 = sum32 + errores3[9*n+1]

sum33 = 0
for n in range(9):
    sum33 = sum33 + errores3[9*n+2]

sum34 = 0
for n in range(9):
    sum34 = sum34 + errores3[9*n+3]

sum35 = 0
for n in range(9):
    sum35 = sum35 + errores3[9*n+4]

sum36 = 0
for n in range(9):
    sum36 = sum36 + errores3[9*n+5]

sum37 = 0
for n in range(9):
    sum37 = sum37 + errores3[9*n+6]

sum38 = 0
for n in range(9):
    sum38 = sum38 + errores3[9*n+7]

sum39 = 0
for n in range(9):
    sum39 = sum39 + errores3[9*n+8]

#Representación errores N=9
"""plt.scatter([1/10,2/10,3/10,4/10,5/10,6/10,7/10,8/10,9/10],[sum31/9,sum32/9,sum33/9,sum34/9,sum35/9,sum36/9,sum37/9,sum38/9,sum39/9])
plt.title("Media errores con N=9")
plt.xlabel("X")

plt.show()"""

