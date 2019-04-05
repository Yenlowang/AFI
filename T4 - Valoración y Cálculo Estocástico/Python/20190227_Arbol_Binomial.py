# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""

# Valoración por Arbol Binomial de opciones Europeas y Americanas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.stats as stat

# Inputs
'''
Datos de mercado
    vol_anual
    r_anual
    S0
    fd = Factor de descuento

Datos del modelo
    DeltaT = 1/N
    N = Número de pasos
    NSim = Número de simulaciones para el MC
    
Vencimiento
    T - Numero de periodos
'''

vol_anual = 0.3
r_anual = 0.05
S0 = 5000
N = 12
DeltaT = 1/N
T = 6
k = 5900
fd = np.exp(-r_anual*DeltaT)
NSim = 10000000
########################################################
# Funciones iniciales
########################################################
'''
PROXY
'''
# U
def proxy_u(vol_anual, delta_t):
    
    u = np.exp(vol_anual * np.sqrt(delta_t))
    return(u)


# P
def proxy_p (r_anual, delta_t, u):
    
    d = 1/u
    p = (np.exp(r_anual * delta_t) - (d))/(u - d)
    return(p)

'''
CONDICIONES DEL MODELO
'''
def condicion_martingala(r_anual, delta_t, p, u):
    
    if(p<=0 or p>=1):
        print("is wrong!!, p should be (0,1)")
    else:
        d = 1/u
        return(np.exp(r_anual * delta_t) - p * u - (1-p)*d)

def ajus_vola(vol_anual, delta_t, p, u):
    
    if(p<=0 or p>=1):
        print("is wrong!!, p should be (0,1)")
    else:
        return(vol_anual * np.sqrt(delta_t) - 2*np.sqrt(p*(1-p))*np.log(u))
    
'''
Funciones a optimizar para obtener parametros p/u
'''
# Deprecated - Usaremos optim_funII
def optim_fun(u, r_anual, vol_anual, delta_t, p):
    
    a = ajus_vola(vol_anual, delta_t, p, u) 
    b = condicion_martingala(r_anual, delta_t, p, u)
    
    #sum_error = a*a + b*b
    return(a*a + b*b)


def optim_funII(x, r_anual, vol_anual, delta_t):
    
    a = ajus_vola(vol_anual, delta_t, x[0], x[1]) 
    b = condicion_martingala(r_anual, delta_t,  x[0], x[1])
    
    #sum_error = a*a + b*b
    return(a*a + b*b)

'''
Definición de Call y Put - lambda funct 
'''
# Call y Put definition
call = lambda x,k: max((x-k), 0)
put = lambda x,k: max((k-x), 0)

########################################################
# END - Funciones iniciales
########################################################
    
########################################################
# Prueba - Funciones iniciales
########################################################

# Utilizando datos de Excel para p y u funciona correctamente
condicion_martingala(r_anual, DeltaT, p = 0.50243922780365, u = 1.09046317849212)
ajus_vola(vol_anual, DeltaT, p = 0.50243922780365, u = 1.09046317849212)

optim_fun(u = 1.09046317849212, r_anual= r_anual, vol_anual= vol_anual, delta_t=DeltaT,
          p = 0.50243922780365) 

########################################################
# Obtención de parametros p/u
########################################################

# Proxy
u = proxy_u(vol_anual=vol_anual, delta_t=DeltaT)
p = proxy_p(r_anual=r_anual, delta_t=DeltaT, u = u)

# Deprecated function
#opt.minimize(optim_fun, 
#             args=(r_anual, vol_anual, DeltaT, p),
#             x0 = u,
#             method='BFGS',
#             options={'disp': True})  

# Optimización para obtener valores definitivos de u y p
prox = np.array([p, u])

optim_values = opt.minimize(optim_funII, 
             args=(r_anual, vol_anual, DeltaT),
             x0 = prox,
             method='BFGS',
             options={'disp': True}) 
p = optim_values.x[0]
u = optim_values.x[1]

########################################################
# Evolución del Subyacente
######################################################## 

###########
# Matriz de evolución del subyacente
###########
    # i - filas
    # j - columnas   
# Con T = 6 replica la hoja de excel
 
# Inicialización de matriz de subyacente
subyacente = np.matrix(np.zeros([T+1,T+1]))
l = subyacente.shape[1]

# Begin of for cycle - Creación de evolución del Subyacente
for i in range(-1,l-1):
    i +=1
    
    if i == 0: 
        for j in range(0,l):
            #j -=1
            subyacente[i, j] = S0*np.power(u, j)
    else:
        for j in range(1,l):
            subyacente[i,j] = subyacente[i-1, j-1]/u
# End of for cycle - Creación de evolución del Subyacente
print(subyacente)

# Plot
for i in range(subyacente.shape[0]):
    plt.plot(list(range(subyacente.shape[0])), subyacente[i,:].tolist()[0],"+")
plt.show()
 

########################################################
# Valor por Arbol Binomial + Payoff (Columna final)
######################################################## 
###########
# Matriz del arbol
###########
    # i - filas
    # j - columnas  

# Inicialización de matriz de arbol
arbol = np.matrix(np.zeros([T+1,T+1]))

# Begin of for cycle - Creación de arbol de precios
for j in range(-T,1)[::-1]:
    j-=1
    # Payoff de la opción 
    if j == -1:
        for i in range(0,T+1):
            arbol[i, j] = call(subyacente[i, j], k)
#            b[i, j] = max((subyacente[i, j]-k), 0) # Call
    else:
        for i in range(0,T):
            
            arbol[i,j]= fd * (p*arbol[i,j+1] + (1-p)*arbol[i+1,j+1])
# End of for cycle - Creación de evolución del Subyacente

# Resultado final           
print("Precio de la opcion call mediante arbol binomial: ", arbol[0,0])

########################################################
# Valor por Formula cerrada
########################################################
# Valores de payoff
payoff = np.asarray(arbol[:,T][::-1])
# Probabilidades
prob = [stat.binom.pmf(x,T,p) for x in range(T+1)]

formu_value = np.sum(np.squeeze(payoff)*prob) * np.power(fd,T)
# Este valor hay que descontarlo - Usamos el factor de descuento y el vto
print("Precio de la opcion call mediante formula: ", formu_value)

########################################################
# Valor por Monte Carlo
########################################################
# Para MC - hay que simular trayectorias, sumamos 1 al numero de valores por encima de p encontrados
# Descontamos el promedio de las simulaciones
# Traduciendo del excel - si aleatorio()<p 0
# Nosotros buscamos entonces los >p
# EJEMPLO np.sum([[0, 1], [0, 5], [0, 5], [0, 5]], axis=1)
#  b[np.sum([[0, 1], [0, 5], [0, 1], [0, 1]], axis=1),6]
mc_value = np.mean(arbol[np.sum(np.random.rand(NSim,T)>p, axis=1),T]) * np.power(fd,T)
print("Precio de la opcion call mediante MC: ", mc_value)
