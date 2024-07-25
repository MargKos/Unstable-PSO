#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 17:52:43 2024

@author: kostre
"""



import numpy as np
import multiprocessing
from multiprocessing import Pool
from functools import partial
import math
from multiprocessing import Pool
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap


# for Fig1 and Fig2: calculates the spectral radius for different parameters and the evolution of the first and second moments

#%%
def fct_abs(c): # complex absolute value
    
    r=c.real
    i=c.imag
    
    return np.sqrt(r**2+i**2)
#%% calculate different operators M and their eigenvalues

Mu=np.linspace(0.01,3.99,1000)
W=np.linspace(-1,1.1,1000)
WMu=np.zeros((1000,1000))
WMu1=np.zeros((1000,1000))
WMuReal=np.zeros((1000,1000))
WMuImag=np.zeros((1000,1000))
WMuSign=np.zeros((1000,1000))
Theta=np.zeros((1000,1000))

D=np.zeros((1000,1000))

Convergent=[]
Convergent1=[]

p=0
for k in range(1000):
    w=W[999-k]
    for j in range(1000):

        mu=Mu[j]
        M=np.array([[ (1+w)**2-4*mu/2*(1+w)+2*(mu**2)/3+2*(mu/2)**2,  2*w*(2*(mu/2)-(1+w)),w**2],
                [(1+w)-mu,-w,0],
                [1,0,0]],
                )
        ev, ew=np.linalg.eig(M)
        
        NormEv=np.zeros(len(ev))
        for i in range(len(ev)):
            NormEv[i]=fct_abs(ev[i])
        
        WMu[k,j]=np.max(NormEv)
        if WMu[k,j]<1:
            Convergent.append([w,mu])
            
        
        M1=np.array([[1+w-mu,-w],[1,0]])
        ev1, ew1=np.linalg.eig(M1)
        NormEv1=np.zeros(len(ev1))
        for i in range(len(ev1)):
            NormEv1[i]=fct_abs(ev1[i])
        WMu1[k,j]=np.max(NormEv1)
        if WMu1[k,j]<1:
            Convergent1.append([w,mu])
        
        #if (1+w-mu)**2<4*w:
        M1=np.array([[1+w-mu,-w],[1,0]])
        ev1, ew1=np.linalg.eig(M1)
        NormEv1=np.zeros(len(ev1))
        for i in range(len(ev1)):
            NormEv1[i]=fct_abs(ev1[i])
        WMu1[k,j]=np.max(NormEv1)
        if WMu1[k,j]<1:
            
            
            alpha=1+w-mu
            Theta[k,j]=np.abs(np.arctan(np.sqrt(np.abs((1+w-mu)**2-4*w))/alpha))
            D[k,j]=(1+w-mu)**2-4*w
                
        
#%% Realtion between theta and rB

# Extract theta and R values from the matrices
# Extract all theta and R values from the matrices
#theta_values = Theta.flatten()
theta_values = WMu.flatten()
r_values = D.flatten()

# Discretize the theta axis into 10 parts
#theta_bins = np.linspace(np.min(theta_values), np.max(theta_values), 30)
theta_bins = np.linspace(np.min(theta_values), 1, 10)

# Initialize lists to store bar heights and positions
bar_heights = []
bar_positions = []

# Iterate over the intervals and find corresponding r values
for i in range(len(theta_bins) - 1):
    theta_start, theta_end = theta_bins[i], theta_bins[i + 1]

    # Extract r values within the current theta interval
    r_values_interval = r_values[(theta_values >= theta_start) & (theta_values <= theta_end)]

    # Store the min and max r values for the bar height
    if len(r_values_interval) > 0:
        bar_heights.append([min(r_values_interval), max(r_values_interval)])
        bar_positions.append((theta_start + theta_end) / 2)

#%%fig=plt.figure()
# Create the bar plot
for i in range(len(bar_positions)):
    plt.bar(bar_positions[i], bar_heights[i][1] - bar_heights[i][0], bottom=bar_heights[i][0], width=theta_bins[1] - theta_bins[0], color='lightblue', alpha=0.8)



plt.ylabel(r'$D$', fontsize=16)
#plt.ylim(0,7)
plt.xlabel(r'$\rho(B)$', fontsize=16)

plt.xlim(0,1)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

plt.savefig('Rho_BD.pdf')
#sns.catplot(x="Medal", y="Year", hue="Gender",kind="box", data=df)

#%%
target_value = 0.645

# Flatten the matrix to a 1D array and find the index of the closest value
flattened_matrix = WMu.flatten()
closest_index = np.abs(flattened_matrix - target_value).argmin()

# Convert the 1D index back to 2D index
rows, cols = np.unravel_index(closest_index, WMu.shape)

print("Closest value:", WMu[rows, cols])
print("Index of closest value (row, column):", (rows, cols))

w=W[999-rows]
mu=Mu[371]
r=WMu[rows, cols]
print(w,mu,r, D[rows, cols])

  
#%%

# first order stable region

Convergent=np.array(Convergent)
hullF=ConvexHull(Convergent)


Convergent1=np.array(Convergent1)
hullF1=ConvexHull(Convergent1)

#%% get all first order stable pairs 

# Define custom colors for your colormap
colors = [(0.0, 'red'),        # Color for values < 0
          (0.1, 'darkorange'),     # Color transition from 0 to 1
          (0.2, 'yellow'),      # Color transition from 0 to 1
          (0.3, 'lime'),       # Color transition from 1 to 3
          (0.4, 'cyan'),    # Color transition from 1 to 3
          (0.6, 'blue'),
          (0.85, 'blueviolet'),
          (1.0, 'fuchsia')]        # Color for values >=3 

# Create the custom colormap
Cmap = LinearSegmentedColormap.from_list('custom_RdYlGn', colors)
Vmax=2.5
Vmin=0






fig=plt.figure()

plt.imshow(WMu,  interpolation='nearest',cmap=Cmap, vmin=Vmin, vmax=Vmax, extent=[0.01, 3.99, -1, 1.1])


# first order stable region


legend_fontsize=26

for simplex in hullF.simplices:
    plt.plot(Convergent[simplex, 1], Convergent[simplex, 0],linewidth=2, color='black' )

plt.title('Spectral radius of B', fontsize=20)


# shade the area inside first order stable region

x_hullF = Convergent[hullF.vertices, 1]
y_hullF = Convergent[hullF.vertices, 0]

# Append the first data point at the end to create a closed path for filling
x_hullF_closed = np.append(x_hullF, x_hullF[0])
y_hullF_closed = np.append(y_hullF, y_hullF[0])

polygon = Polygon(np.column_stack((x_hullF_closed, y_hullF_closed)), closed=True, edgecolor=None, facecolor='grey', alpha=0.5)
#plt.add_patch(polygon)

# make a box only under the last row  in the centre with latex label \mu
fig.text(0.49, 0.2, r'$\mu$', ha='center', fontsize=20)

# y-axis label next to the left plot
fig.text(0, 0.53, 'w', va='center', rotation='vertical', fontsize=20)
#cbar_ax = fig.add_axes([0.89, 0.14, 0.03, 0.7])  # [left, bottom, width, height]
cbar = plt.colorbar( shrink=0.82)  # Experiment with the shrink value

# Increase the size of colorbar ticks
cbar.ax.tick_params(labelsize=16) 
#cbar.ax.tick_params(labelsize=26)
plt.xticks(fontsize=12)  # Adjust the fontsize parameter as needed
plt.yticks(fontsize=12)
#plt.legend(bbox_to_anchor=(-1.41, 2.8, 1, 0.6), loc="upper left",ncol=3,  prop={'size': legend_fontsize})

# Save the figure
plt.subplots_adjust(top=0.85, bottom=0.25, left=0.1, right=0.88)
plt.savefig('Fig01.pdf',  bbox_inches="tight")

# 
fig=plt.figure()

plt.imshow(WMu,  interpolation='nearest',cmap=Cmap, vmin=Vmin, vmax=Vmax, extent=[0.01, 3.99, -1, 1.1])


# first order stable region


legend_fontsize=26

for simplex in hullF.simplices:
    plt.plot(Convergent[simplex, 1], Convergent[simplex, 0],linewidth=2, color='black' )

plt.title('Spectral radius of B', fontsize=20)


# shade the area inside first order stable region

x_hullF = Convergent[hullF.vertices, 1]
y_hullF = Convergent[hullF.vertices, 0]

# Append the first data point at the end to create a closed path for filling
x_hullF_closed = np.append(x_hullF, x_hullF[0])
y_hullF_closed = np.append(y_hullF, y_hullF[0])

polygon = Polygon(np.column_stack((x_hullF_closed, y_hullF_closed)), closed=True, edgecolor=None, facecolor='grey', alpha=0.5)
#plt.add_patch(polygon)

# make a box only under the last row  in the centre with latex label \mu
fig.text(0.49, 0.2, r'$\mu$', ha='center', fontsize=20)

# y-axis label next to the left plot
fig.text(0, 0.53, 'w', va='center', rotation='vertical', fontsize=20)
#cbar_ax = fig.add_axes([0.89, 0.14, 0.03, 0.7])  # [left, bottom, width, height]
cbar = plt.colorbar( shrink=0.82)  # Experiment with the shrink value

# Increase the size of colorbar ticks
cbar.ax.tick_params(labelsize=16) 
#cbar.ax.tick_params(labelsize=26)
plt.xticks(fontsize=12)  # Adjust the fontsize parameter as needed
plt.yticks(fontsize=12)
#plt.legend(bbox_to_anchor=(-1.41, 2.8, 1, 0.6), loc="upper left",ncol=3,  prop={'size': legend_fontsize})

# Save the figure
plt.subplots_adjust(top=0.85, bottom=0.25, left=0.1, right=0.88)
plt.savefig('Fig01.pdf',  bbox_inches="tight")


#%% plot rhos(A) and rho(B) next to each other



# Create a subplot with 1 row and 2 columns
fig, axs = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1]})


# 


im1=axs[0].imshow(WMu1,  interpolation='nearest',cmap=Cmap, vmin=Vmin, vmax=Vmax, extent=[0.01, 3.99, -1, 1.1], aspect='auto')



legend_fontsize=26

for simplex in hullF1.simplices:
    axs[0].plot(Convergent1[simplex, 1], Convergent1[simplex, 0],linewidth=2, color='black' )

axs[0].set_title('Spectral radius of $\mathcal{A}$', fontsize=20)


# shade the area inside first order stable region

x_hullF1 = Convergent1[hullF1.vertices, 1]
y_hullF1 = Convergent1[hullF1.vertices, 0]

# Append the first data point at the end to create a closed path for filling
x_hullF_closed = np.append(x_hullF, x_hullF[0])
y_hullF_closed = np.append(y_hullF, y_hullF[0])

polygon = Polygon(np.column_stack((x_hullF_closed, y_hullF_closed)), closed=True, edgecolor=None, facecolor='grey', alpha=0.5)
#plt.add_patch(polygon)

# make a box only under the last row  in the centre with latex label \mu
axs[0].text(4.5, -1.3, r'$\mu$', ha='center', fontsize=20)

# y-axis label next to the left plot
axs[0].text(-0.8, 0, 'w', va='center', rotation='vertical', fontsize=20)
#cbar_ax = fig.add_axes([0.89, 0.14, 0.03, 0.7])  # [left, bottom, width, height]
#cbar = plt.colorbar(im, shrink=0.82)  # Experiment with the shrink value

# Increase the size of colorbar ticks
cbar.ax.tick_params(labelsize=20) 
#cbar.ax.tick_params(labelsize=26)
axs[0].tick_params(axis='both', labelsize=20)

im=axs[1].imshow(WMu,  interpolation='nearest',cmap=Cmap, vmin=Vmin, vmax=Vmax, extent=[0.01, 3.99, -1, 1.1],  aspect='auto')



#legend_fontsize=26

for simplex in hullF.simplices:
    axs[1].plot(Convergent[simplex, 1], Convergent[simplex, 0],linewidth=2, color='black' )

# on top
for simplex in hullF1.simplices:
    axs[1].plot(Convergent1[simplex, 1], Convergent1[simplex, 0],linewidth=2, color='black' )

    

axs[1].set_title('Spectral radius of $\mathcal{B}$', fontsize=20)



# shade the area inside first order stable region

x_hullF = Convergent[hullF.vertices, 1]
y_hullF = Convergent[hullF.vertices, 0]

# Append the first data point at the end to create a closed path for filling
x_hullF_closed = np.append(x_hullF, x_hullF[0])
y_hullF_closed = np.append(y_hullF, y_hullF[0])

polygon = Polygon(np.column_stack((x_hullF_closed, y_hullF_closed)), closed=True, edgecolor=None, facecolor='grey', alpha=0.5)
#plt.add_patch(polygon)

# make a box only under the last row  in the centre with latex label \mu
#axs[1].text(0.49, 0.2, r'$\mu$', ha='center', fontsize=20)

# y-axis label next to the left plot
#axs[1].text(0, 0.53, 'w', va='center', rotation='vertical', fontsize=20)
#cbar_ax = fig.add_axes([0.89, 0.14, 0.03, 0.7])  # [left, bottom, width, height]
cbar = plt.colorbar(im, shrink=1)  # Experiment with the shrink value
axs[1].text(-0.1, -1.5, "b)", fontsize=20, weight='bold')
axs[0].text(-0.1, -1.5, "a)", fontsize=20, weight='bold')
# Increase the size of colorbar ticks
cbar.ax.tick_params(labelsize=20) 
#cbar.ax.tick_params(labelsize=26)
axs[1].tick_params(axis='both', labelsize=20)
#plt.legend(bbox_to_anchor=(-1.41, 2.8, 1, 0.6), loc="upper left",ncol=3,  prop={'size': legend_fontsize})

# Save the figure
#plt.subplots_adjust(top=0.85, bottom=0.25, left=0.1, right=0.88)
plt.subplots_adjust(top=0.8, bottom=0.25, left=0.1, right=0.85, wspace=0.3)
#plt.savefig('Fig1.png',  bbox_inches="tight")


