# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 12:19:55 2019

@author: E051692
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')

from scipy.stats import norm

N = norm.cdf

from scipy.optimize import fsolve
from scipy import optimize

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)


def enable_plotly_in_cell():
  import IPython
  from plotly.offline import init_notebook_mode
  display(IPython.core.display.HTML('''<script src="/static/components/requirejs/require.js"></script>'''))
  init_notebook_mode(connected=False)
  
def plot_3D(data):

  enable_plotly_in_cell()


  data = [
      go.Surface(
          z=data.T
      )
  ]
  layout = go.Layout(
      title='Option price',
      autosize=False,
      width=500,
      height=500,
      margin=dict(
          l=65,
          r=50,
          b=65,
          t=90
      )
  )
  fig = go.Figure(data=data, layout=layout)
  iplot(fig, filename='elevations-3d-surface')
  
def compute_M_matrix(time_grid, x_grid, time_grid_index, drift_functor, var_functor, disc_functor):
  
  # Assumes a regular x_grid
  
  M = np.zeros((len(x_grid)-2, len(x_grid)-2))
  
  inv_delta_x = 1/(x_grid[1]-x_grid[0])
  
  inv_delta_x_squared = inv_delta_x * inv_delta_x 
  
  
  delta_t = time_grid[1]-time_grid[0]
  
  rho = delta_t * inv_delta_x
  alpha = delta_t *  inv_delta_x_squared
  
  t = time_grid[time_grid_index]
  
  
  for i in range(len(x_grid)-2):
    
    x = x_grid[i]
    
    var = var_functor(t, x)
    drift =  drift_functor(t, x)
    disc = disc_functor(t, x)
    
    if i == 0: # equivale a j=1 en la memoria
      
      M[i,i] = 1 + delta_t * disc - rho * drift      
      M[i,i+1] = rho * drift  


    elif i == len(x_grid)-3:  # equivale a j=m-1 en la memoria

      M[i,i] = 1 + delta_t * disc + rho * drift
      M[i,i-1] = - rho * drift
      
    else:  # equivale a 1<j<m-1 en la memoria
    
      M[i,i] = 1 - 2 * alpha * var + delta_t * disc
      M[i,i+1] = alpha * var + rho/2 * drift
      M[i,i-1] = alpha * var - rho/2 * drift
      
    
    
  return M

def Forward_PDE(Spot, TTM, n_time, K_min, K_max, n_strike, sigma, drift_ftor, var_ftor, disc):
  
  time_grid = np.linspace(0,TTM, n_time + 1, True)
  strike_grid = np.linspace(K_min, K_max, n_strike + 1, True)
  option = np.zeros((n_strike + 1, n_time + 1))
  option[:,0] = np.maximum(Spot-strike_grid,0)
  
  delta_T = TTM/n_time
  
  for i in range(0,n_time):
  
    M_i = compute_M_matrix(time_grid, strike_grid, i, 
                           drift_ftor, var_ftor, disc)
    
    M_inv = np.linalg.inv(M_i)

    option[1:n_strike, i+1] = np.matmul(M_inv, option[1:n_strike,i])

    # Cond. Neumann
    option[0, i+1] = 2* option[1, i+1] - option[2, i+1]
    option[n_strike, i+1] = 2* option[n_strike-1, i+1] - option[n_strike-2, i+1]
  
  return option 
  

TTM = 0.5 #6 meses # representa último vencimiento

beta = 0.7

Spot = 9

K_min = 0
K_max = 4*Spot
rate = 0.01
vol = 50/100


n_time = 100
n_strike = 50



drift_ftor = lambda t, x: rate*x
var_ftor = lambda t, x: -1/2*vol*vol*x**(2*beta)
disc = lambda t, x : -rate
option=Forward_PDE(Spot, TTM, n_time, K_min, K_max, n_strike, vol, drift_ftor, var_ftor, disc)