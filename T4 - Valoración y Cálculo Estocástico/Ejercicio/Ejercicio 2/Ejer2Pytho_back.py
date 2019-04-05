import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')

from scipy.stats import norm

N = norm.cdf

from scipy.optimize import fsolve
from scipy import optimize

TTM = 0.5 

beta = 0.7

Spot = 9

Strike = 8

S_min = 0
S_max = 4*Spot
rate = 0.01
vol = 50/100


n_time = 100
n_spot = 50



drift_ftor = lambda t, x: rate*x
var_ftor_back = lambda t, x: 1/2*vol*vol*x**(2*beta)
disc = lambda t, x : -rate

def compute_M_matrix_back(time_grid, x_grid, time_grid_index, drift_functor, var_functor, disc_functor):
  
  # Assumes a regular x_grid
  
  M = np.zeros((len(x_grid)-2, len(x_grid)-2))
  
  inv_delta_x = 1/(x_grid[1]-x_grid[0])
  
  inv_delta_x_squared = inv_delta_x * inv_delta_x 
  
  
  delta_t = time_grid[1]-time_grid[0]
  
  rho = delta_t * inv_delta_x
  alpha = delta_t *  inv_delta_x_squared
  
  t = time_grid[time_grid_index]
  
  
  
  for i in range(len(x_grid)-2):
    
    x = x_grid[i+1]
    
    var = var_functor(t, x)
    drift =  drift_functor(t, x)
    disc = disc_functor(t, x)
    
    if i == 0: # equivale a j=1 en la memoria
      
      M[i,i] = 1 - delta_t * disc + rho * drift      
      M[i,i+1] = -rho * drift  


    elif i == len(x_grid)-3:  # equivale a j=m-1 en la memoria

      M[i,i] = 1 - delta_t * disc - rho * drift
      M[i,i-1] = + rho * drift
      
    else:  # equivale a 1<j<m-1 en la memoria
    
      M[i,i] = 1 + 2 * alpha * var - delta_t * disc
      M[i,i+1] = -alpha * var - rho/2 * drift
      M[i,i-1] = -alpha * var + rho/2 * drift
      
    
    
  return M

def Backward_PDE(Strike, TTM, n_time, S_min, S_max, n_spot, sigma, drift_ftor, var_ftor_back, disc):
  
  time_grid = np.linspace(0,TTM, n_time + 1, True)
  spot_grid = np.linspace(S_min, S_max, n_spot + 1, True)

  option = np.zeros((n_strike + 1,n_time + 1))

  option[:,-1] = np.maximum(spot_grid - Strike, 0)
  
  delta_T = TTM/n_time
  
  for i in range(n_time-1,-1,-1):
  
    M_i = compute_M_matrix_back(time_grid, spot_grid, i, 
                           drift_ftor, var_ftor_back, disc)
    
    M_inv = np.linalg.inv(M_i)

    option[1:n_spot, i] = np.matmul(M_inv, option[1:n_spot,i+1])

    option[0, i] = 2* option[1, i] - option[2, i]
    option[n_spot, i] = 2* option[n_spot-1, i] - option[n_spot-2, i]
  
  return option 


ba = Backward_PDE(Strike, TTM, n_time, S_min, S_max, n_strike, vol, drift_ftor, var_ftor_back, disc)

Backward_PDE(Strike, TTM, n_time, S_min, S_max, n_strike, vol, drift_ftor, var_ftor_back, disc)