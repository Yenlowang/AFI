# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 19:40:55 2019

@author: E051692
"""

# Metodo de diferencias finitas

# Librerias
import numpy as np
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.sparse import diags

#%%
############################
#   Ecuación del calor
############################
# du/dt = d2u/dx2
# du/dt = d2u/dx2 -> du/dt - d2u/dx2 = 0 (Ecuación Forward - Necesita Condicion inicial)

# Condiciones iniciales y finales
# Espaciales
x_ini = 0
x_fin = 5
# Temporales
t_0 = 0
T = 0.1

# Pasos
nt = 100
nx = 50

# Definimos deltax y deltat
dt = (T - t_0)/nt
dx = (x_fin - x_ini)/nx

# alpha
alpha = dt/(dx*dx)

# Condición inicial - Necesaria porque es ecu forward (0 ---> T)

def cond_ini(x):
    
    if ((x < 2) or (x > 3)):
        t = 0
    else:
        t = 1
    return t
    
# Definimos el mallado
malla = np.zeros((nx +2, nt + 2))
malla[0,0] = 0
malla[1:, 0] = np.linspace(x_ini, x_fin, num = nx+1)
malla[0,1:] = np.linspace(t_0, T, num = nt+1)


# Rellenamos la malla con la condición inicial y las condiciones de contorno
# Rellenamos el primer elemento de la malla - Con la condición inicial
for i in range(1, nx + 2):
    
    malla[i, 1] = cond_ini(malla[i,0])

# Rellenamos las condiciones de contorno en este caso son Dirichlet 
    # Por que he fijado un valor (podría ser una funcion que me de un valor)
    # en los extremos.
malla[1,1:] = 0
malla[nx+1,1:] = 1.5 * malla[0,1:] # Condición tipo dirichlet que con una funcion

#%%
############################
#   METODO EXPLICITO
############################
# u_i+1(j) = alpha*u_i(j+1) + (1-2*alpha)*u_i(j) + alpha * u_i(j-1)

for i in range(1, nt+1): # i - columnas
    
    for j in range(2, nx+1): # j - filas (Empiezo en fila 2 por que la 0 es el tiempo, y la 1 es el condición inicial)

        malla[j,i+1] = alpha * malla[j-1, i] + (1-2*alpha) * malla[j,i] +\
        alpha * malla[j+1, i]

fig = plt.figure(figsize=(12,7))
#ax = fig.gca(projection = "3d")
ax = fig.add_subplot(111, projection='3d')

##############
#   Plotting
##############

# Make data
x, y = malla[1:,1:].shape
x = np.linspace(0, 5, x)
y = np.linspace(0, 0.01, y)
xv, yv = np.meshgrid(y,x)

# Surface plot
z = ax.plot_surface(yv, xv, malla[1:,1:], rstride=1, cstride=1,cmap=plt.cm.jet,
                     linewidth=0, antialiased=False, alpha = 0.5)

fig.colorbar(z, shrink=0.5, aspect=5)
ax.set_title('Heat Equation -> Explicito')
ax.view_init(elev = 45, azim = 35)
plt.show()
# Para ver como sería el perfil
#plt.plot(np.linspace(0, 5, malla[1:,1:].shape[0]), malla[1:,101]);
#%%
############################
#   METODO IMPLICITO
############################
# Mu_i+1 = b_i
# Hay que hacer la inversa de M y obtener u_i+1 = M(inv)*b_i
# b_i = u_i+CondIni

# Creamos matriz M
diagonals =  diags([-alpha, 1+2*alpha, -alpha], [-1, 0, 1], shape=(nx-2, nx-2)).toarray()

for i in range(1, nt+1): # i - columnas
    
    for j in range(2, nx+1): # j - filas
#        malla[j+1,i] = 12
        malla[2:50,i+1] = np.matmul(np.linalg.inv(diagonals),malla[2:50,i])

##############
#   Plotting
##############

fig = plt.figure(figsize=(12,7))
#ax = fig.gca(projection = "3d")
ax = fig.add_subplot(111, projection='3d')

# Make data
x, y = malla[1:,1:].shape
x = np.linspace(0, 5, x)
y = np.linspace(0, 0.01, y)
xv, yv = np.meshgrid(y,x)

# Surface plot
z = ax.plot_surface(yv, xv, malla[1:,1:], rstride=1, cstride=1,cmap=plt.cm.jet,
                     linewidth=0, antialiased=False, alpha = 0.5)

fig.colorbar(z, shrink=0.5, aspect=5, fraction = 0.2)
ax.set_title('Heat Equation -> Implicito')
ax.view_init(45, 35)
plt.show()

#%%
############################
#   METODO CRANK - NICOLSON (IMPLICITO) # LA MATRIZ M ESTÁ MAL EN LOS EXCEL 
############################
# M u_i+1 = b_i
# Hay que hacer la inversa de M y obtener u_i+1 = M(inv)*b_i
# b_i = N u_i + CondIni

# Creamos matriz M
diagonal_M = diags([-0.5 * alpha, 1 + alpha, -0.5 * alpha], [-1, 0, 1], 
                   shape=(nx-2, nx-2)).toarray()
# Creamos matriz N
diagonal_N =  diags([0.5 * alpha, 1 - alpha, 0.5 * alpha], [-1, 0, 1], 
                   shape=(nx-2, nx-2)).toarray()

for i in range(1, nt+1): # i - columnas
    
    for j in range(2, nx+1): # j - filas
#        malla[j+1,i] = 12
        malla[2:50,i+1] = np.matmul(np.linalg.inv(diagonal_M), 
             np.matmul(diagonal_N, malla[2:50,i]))

##############
#   Plotting
##############
        
fig = plt.figure(figsize=(12,7))
#ax = fig.gca(projection = "3d")
ax = fig.add_subplot(111, projection='3d')

# Make data
x, y = malla[1:,1:].shape
x = np.linspace(0, 5, x)
y = np.linspace(0, 0.01, y)
xv, yv = np.meshgrid(y,x)

# Surface plot
z = ax.plot_surface(yv, xv, malla[1:,1:], rstride=1, cstride=1,cmap=plt.cm.jet,
                     linewidth=0, antialiased=False, alpha = 0.5)

fig.colorbar(z, shrink=0.5, aspect=5, fraction = 0.2)
ax.set_title('Heat Equation -> Crank Nicolson')
ax.view_init(45, 35)
plt.show()