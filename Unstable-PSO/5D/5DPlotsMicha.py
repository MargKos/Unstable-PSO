#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 17:10:57 2023

@author: bzfkostr
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib as mpl

# plot the Michalewicz function in 2D for all pairs of dimensions

def fct_michalewicz(z, k, l): # z is a vector of length 2, k and l are the dimensions that are varied
    y = 0
    dim = 5
    m=1
    x=np.array(np.array([2.07169496, 1.57079963, 1.30466642, 1.91628545, 1.71824045])) # global minimum
    x=np.zeros(5)
    x[k]=z[0]
    x[l]=z[1]
    for j in range(dim):
       
        y = y +np.sin(x[j])*(np.sin(((j+1)*x[j]**2)/(np.pi)))**(2*m)
    
    return -y

# Plot

cols=4
rows=4
Bdl=0
Bdu=np.pi

Vmax=0.5
Vmin=-0.5
fig, axs = plt.subplots(rows, cols, figsize=(7.5 * cols, 5.5 * rows))

k=0
l=0

T_Orbit=30
T_Classic=1000
T_Damping=1000

for i in range(5):
    for j in range(5):
        if i!=j:
            
            axsi=axs[k,l]
            x = np.linspace(Bdl,Bdu, 100)
            y = np.linspace(Bdu,Bdl, 100)
            
            zl=np.zeros((100,100))
            for ry in range(100):
                for rx in range(100):
                    zl[ry, rx]=fct_michalewicz(np.array([x[rx], y[ry]]), i, j)
            
            cmap = mpl.cm.rainbow
            norm = mpl.colors.Normalize(vmin=Vmin, vmax=Vmax)
            norm = mpl.colors.Normalize()
            axsi.imshow(zl,  interpolation='nearest',cmap=cmap, norm=norm, extent=[Bdl,Bdu,Bdl,Bdu])
            cbar=fig.colorbar(mpl.cm.ScalarMappable( norm=norm, cmap=cmap), ax=axsi)
            cbar.ax.tick_params(labelsize=20)
            axsi.set_ylim(Bdl, Bdu)
            axsi.set_xlim(Bdl, Bdu)
            axsi.tick_params(axis='x', labelsize=20)
            axsi.tick_params(axis='y', labelsize=20)
            
            
            l=l+1
        
            if l==4:
                l=0
                k=k+1
                
    if k==4:
        break


for i in range(4):
    axsi=axs[0,i]
    axsi.set_title(str(i+1), fontsize=20, weight='bold', y=1.05)      
    
# Add numbers on the left side
for ax, num in zip(axs[:, 0], range(2, 6)):
    ax.set_ylabel(str(num), fontsize=20, weight='bold', rotation=0, ha='right', va='center')

plt.savefig('./Plots/Fig9.eps')

    
                