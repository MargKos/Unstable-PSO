#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 10:29:45 2023

@author: bzfkostr
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
WMuReal=np.zeros((1000,1000))
WMuImag=np.zeros((1000,1000))
WMuSign=np.zeros((1000,1000))
p=0
for k in range(1000):
    w=W[999-k]
    for j in range(1000):

        mu=Mu[j]
        M=np.array([[1+w-mu, -w,0,0,0],
                [1,0,0,0,0],
                [p*4*(mu/2*(1+w)-(mu**2)/3-(mu/2)**2), -4*mu/2*w*p, (1+w)**2-4*mu/2*(1+w)+2*(mu**2)/3+2*(mu/2)**2,  2*w*(2*(mu/2)-(1+w)),w**2],
                [mu*p,0,(1+w)-mu,-w,0],
                [0,0,1,0,0],
                ])
        ev, ew=np.linalg.eig(M)
        
        NormEv=np.zeros(len(ev))
        for i in range(len(ev)):
            NormEv[i]=fct_abs(ev[i])
        
        WMu[k,j]=np.max(NormEv)
        
        M1=np.array([[1+w-mu,-w],[1,0]])
        ev1, ew1=np.linalg.eig(M1)
        WMuReal[k,j]=np.abs(ev1[0].real)
        WMuImag[k,j]=np.abs(ev1[0].imag)

#%%
def eigenvalue_test(w,mu):
    
    
    M=np.array([[1+w-mu, -w,0,0,0],
            [1,0,0,0,0],
            [p*4*(mu/2*(1+w)-(mu**2)/3-(mu/2)**2), -4*mu/2*w*p, (1+w)**2-4*mu/2*(1+w)+2*(mu**2)/3+2*(mu/2)**2,  2*w*(2*(mu/2)-(1+w)),w**2],
            [mu*p,0,(1+w)-mu,-w,0],
            [0,0,1,0,0],
            ])
    ev, ew=np.linalg.eig(M)
    
    NormEv=np.zeros(len(ev))
    for i in range(len(ev)):
        NormEv[i]=fct_abs(ev[i])
    
    radius=np.max(NormEv)

    M1=np.array([[1+w-mu,-w],[1,0]])
    ev1, ew1=np.linalg.eig(M1)
    real=np.abs(ev1[0].real)
    imag=np.abs(ev1[0].imag)
    
    
    return radius, real, imag
    
#%% Get all points where the spectral radius is smaller than 0.95
SpectralSmallerThan7=[]
for k in range(1000):
    for j in range(1000):
        if WMu[k,j]<0.95 and WMuImag[k,j]>0:
            SpectralSmallerThan7.append([W[999-k],Mu[j]])
            
            

#%% Get all points where the spectral radius is bigger than 0.95
SpectralBiggerThan7=[]
for k in range(1000):
    for j in range(1000):
        if 1>WMu[k,j]>0.95 and WMuImag[k,j]>0:
            SpectralBiggerThan7.append([W[999-k],Mu[j]])


radius, real,imag=eigenvalue_test(SpectralBiggerThan7[-1][0],SpectralBiggerThan7[-1][1])

#%% Get all points where imaginary part of the eigenvalues is almost one but still second order stable
ImaginaryOne=[]
for k in range(1000):
    for j in range(1000):
        if 1>WMuImag[k,j]>0.8 and WMu[k,j]<1:
            ImaginaryOne.append([W[999-k],Mu[j]])
        
#%% Harmonic oscillatory region: get all paris of w, mu that satisfy w**2+mu**2-2*w*mu-2*w+2*mu+1<0
Harmonic=[]
for k in range(1000):
    for j in range(1000):
        if W[k]**2+Mu[j]**2-2*W[k]*Mu[j]-2*W[k]-2*Mu[j]+1<0:
            Harmonic.append([W[k],Mu[j]])


#%% get all second order stable pairs 

SecondOrder=[]
for k in range(1000):
    for j in range(1000):
        if Mu[j]<(12*(1-W[k]**2))/(7-5*W[k]) and -1<W[k]<1:  
            SecondOrder.append([W[k],Mu[j]])
            
#%%
FirstOrder=[]
for k in range(1000):
    for j in range(1000):
        if 2*W[k]-Mu[j]+2>0 and -1<W[k]<1:  
            FirstOrder.append([W[k],Mu[j]])
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


fig, axs = plt.subplots(3, 2, figsize=(12, 14), sharex='col', gridspec_kw={'hspace': -0.3})

axs[0,1].imshow(WMu,  interpolation='nearest',cmap=Cmap, vmin=Vmin, vmax=Vmax, extent=[0.01, 3.99, -1, 1.1])
axs[0,1].tick_params(labelsize=26)
axs[0,1].set_title('Spectral Radius', fontsize=30)

# plot one stable point as a dot in all plots
axs[0,1].plot(1.4,0.7, 's', markersize=6, color='black', label='Overdamped')
axs[0,1].plot(0.3215,0.9, '^', markersize=6, color='black', label='Damped')
axs[0,1].plot(0.14,1, 'o', markersize=6, color='black')
axs[0,1].plot(0.03,1, 'o', markersize=6, color='black', label='Divergent')

# Plot other points
axs[1,0].plot(1.4, 0.7, 's', markersize=6, color='black')
axs[1,0].plot(0.3215, 0.9, '^', markersize=6, color='black')
axs[1,0].plot(0.14, 1, 'o', markersize=6, color='black')
axs[1,0].plot(0.03, 1, 'o', markersize=6, color='black')

# plot one stable point as a dot in all plots
axs[1,1].plot(1.4,0.7, 's', markersize=6, color='black', label='Overdamped')
axs[1,1].plot(0.3215,0.9, '^', markersize=6, color='black', label='Damped')
axs[1,1].plot(0.14,1, 'o', markersize=6, color='black')
axs[1,1].plot(0.03,1, 'o', markersize=6, color='black', label='Divergent')

# plot all points in the next row
axs[2,0].plot(1.4,0.7, 's', markersize=6, color='black', label='Overdamped')
axs[2,0].plot(0.3215,0.9, '^', markersize=6, color='black', label='Damped')
axs[2,0].plot(0.14,1, 'o', markersize=6, color='black', label='Divergent')
axs[2,0].plot(0.03,1, 'o', markersize=6, color='black')




# first order stable region

FirstOrder=np.array(FirstOrder)
hullF=ConvexHull(FirstOrder)
legend_fontsize=26
axs[1,0].imshow(WMu,  interpolation='nearest',cmap=Cmap, vmin=Vmin, vmax=Vmax, extent=[0.01, 3.99, -1, 1.1])
for simplex in hullF.simplices:
    axs[1,0].plot(FirstOrder[simplex, 1], FirstOrder[simplex, 0],linewidth=2, color='black' )
axs[1,0].tick_params(labelsize=26)
axs[1,0].set_title('1-order Stable', fontsize=30)


# second order stable region

SecondOrder=np.array(SecondOrder)
hullS=ConvexHull(SecondOrder)
im=axs[2,0].imshow(WMu,  interpolation='nearest',cmap=Cmap, vmin=Vmin, vmax=Vmax, extent=[0.01, 3.99, -1, 1.1])
axs[2,0].tick_params(labelsize=26)

axs[2,0].set_title('2-order Stable', fontsize=30)

for simplex in hullS.simplices:
    axs[2,0].plot(SecondOrder[simplex, 1], SecondOrder[simplex, 0],linewidth=2, color='black')

axs[1,1].imshow(WMu,  interpolation='nearest',cmap=Cmap, vmin=Vmin, vmax=Vmax, extent=[0.01, 3.99, -1, 1.1])
axs[1,1].tick_params(labelsize=26)
axs[1,1].tick_params(labelsize=26)
axs[1,1].set_title('Harmonic Oscillatory', fontsize=30)


Harmonic=np.array(Harmonic)
hullH=ConvexHull(Harmonic)
for simplex in hullH.simplices:
    axs[1,1].plot(Harmonic[simplex, 1], Harmonic[simplex, 0],linewidth=2, color='black')


# write a) b) c) under each subplot
axs[0,0].text(0.1, -0.2, '(a)', transform=axs[0,0].transAxes, fontsize=20, va='top', ha='right',weight='bold')
axs[0,1].text(0.1, -0.2, '(b)', transform=axs[0,1].transAxes, fontsize=20, va='top', ha='right',weight='bold')
axs[1,0].text(0.1, -0.3, '(c)', transform=axs[1,0].transAxes, fontsize=20, va='top', ha='right',weight='bold')
axs[1,1].text(0.1, -0.3, '(d)', transform=axs[1,1].transAxes, fontsize=20, va='top', ha='right',weight='bold')
axs[2,0].text(0.1, -0.4, '(e)', transform=axs[2,0].transAxes, fontsize=20, va='top', ha='right',weight='bold')
axs[2,1].text(0.1, -0.4, '(f)', transform=axs[2,1].transAxes, fontsize=20, va='top', ha='right',weight='bold')

# shade the area inside first order stable region

x_hullF = FirstOrder[hullF.vertices, 1]
y_hullF = FirstOrder[hullF.vertices, 0]

# Append the first data point at the end to create a closed path for filling
x_hullF_closed = np.append(x_hullF, x_hullF[0])
y_hullF_closed = np.append(y_hullF, y_hullF[0])

polygon = Polygon(np.column_stack((x_hullF_closed, y_hullF_closed)), closed=True, edgecolor=None, facecolor='grey', alpha=0.5)
axs[1, 0].add_patch(polygon)

# shade the area inside second order stable region
x_hullS = SecondOrder[hullS.vertices, 1]
y_hullS = SecondOrder[hullS.vertices, 0]

# Append the first data point at the end to create a closed path
x_hullS_closed = np.append(x_hullS, x_hullS[0])
y_hullS_closed = np.append(y_hullS, y_hullS[0])

polygon = Polygon(np.column_stack((x_hullS_closed, y_hullS_closed)), closed=True, edgecolor=None, facecolor='grey', alpha=0.5)
axs[2, 0].add_patch(polygon)

# shade the area inside harmonic oscillatory region

x_hullH = Harmonic[hullH.vertices, 1]
y_hullH = Harmonic[hullH.vertices, 0]

# Append the first data point at the end to create a closed path
x_hullH_closed = np.append(x_hullH, x_hullH[0])
y_hullH_closed = np.append(y_hullH, y_hullH[0])

polygon = Polygon(np.column_stack((x_hullH_closed, y_hullH_closed)), closed=True, edgecolor=None, facecolor='grey', alpha=0.5)
axs[1, 1].add_patch(polygon)

# plot the  WMuImag int the first row


axs[0,0].imshow(WMuImag,  interpolation='nearest',cmap=Cmap, vmin=Vmin, vmax=Vmax, extent=[0.01, 3.99, -1, 1.1])
axs[0,0].tick_params(labelsize=26)
axs[0,0].set_title('Imaginary Part', fontsize=30)



# Plot Damped Region

axs[2,1].set_title('Damped', fontsize=30)
axs[2,1].imshow(WMu, interpolation='nearest', cmap=Cmap, vmin=Vmin, vmax=Vmax, extent=[0.01, 3.99, -1, 1.1])
axs[2,1].tick_params(labelsize=30)



# plot all points from SpectralBiggerThan7

for i in range(len(SpectralBiggerThan7)):
    axs[2,1].plot(SpectralBiggerThan7[i][1], SpectralBiggerThan7[i][0], 's', markersize=1, alpha=0.2, color='grey')


# plot all points in the damped region
axs[2,1].plot(1.4,0.68, 's', markersize=6, color='black', label='Overdamped')
axs[2,1].plot(0.3215,0.9+0.05, '^', markersize=6, color='black', label='Damped')
axs[2,1].plot(0.14,1, 'o', markersize=6, color='black')
axs[2,1].plot(0.03,1, 'o', markersize=6, color='black', label='Divergent')

# make a box only under the last row  in the centre with latex label \mu
fig.text(0.04, 0.5, r'$w$', fontsize=20, va='center', rotation='vertical', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
fig.text(0.48, 0.09, r'$\mu$', fontsize=20, va='center', rotation='horizontal', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

cbar_ax = fig.add_axes([0.89, 0.14, 0.03, 0.7])  # [left, bottom, width, height]
cbar = plt.colorbar(im, cax=cbar_ax)
cbar.ax.tick_params(labelsize=26)

axs[1,1].legend(bbox_to_anchor=(-1.41, 2.8, 1, 0.6), loc="upper left",ncol=3,  prop={'size': legend_fontsize})

# Save the figure
plt.subplots_adjust(top=0.95, bottom=0.03, left=0.1, right=0.88)
plt.savefig('./Fig4.png')

#%% New Figure 4

fig = plt.figure(figsize=(15,10))

# Plot Damped Region

plt.imshow(WMu, interpolation='nearest', cmap=Cmap, vmin=Vmin, vmax=Vmax, extent=[0.01, 3.99, -1, 1.1])
plt.tick_params(labelsize=30)

# Plot all points from SpectralBiggerThan7
for i in range(len(SpectralBiggerThan7)):
    plt.plot(SpectralBiggerThan7[i][1], SpectralBiggerThan7[i][0], 's', markersize=1, alpha=0.2, color='grey')

# Plot all points in the damped region
plt.plot(1.4, 0.68, 's', markersize=20, color='black', label='Overdamped')
plt.plot(0.3215, 0.9 + 0.05, '^', markersize=20, color='black', label='Damped')
plt.plot(0.14, 1, 'o', markersize=20, color='black')
plt.plot(0.03, 1, 'o', markersize=20, color='black', label='Divergent')

plt.text(0.75, 0.5, 'Damped Region', color='grey', fontsize=25, va='center', ha='left', fontweight='bold')
# Make a box only under the last row in the center with latex label \mu
plt.ylabel( r'$w$', fontsize=30)
plt.xlabel( r'$\mu$', fontsize=30)

# Adding legend
plt.legend(bbox_to_anchor=(-0.06, 0.9, 1, 0.6), loc="center", ncol=3, prop={'size': legend_fontsize})

# Adjusting subplot layout
plt.subplots_adjust(top=0.95, bottom=0.03, left=0.1, right=0.88)

# Save the figure
plt.savefig('./Fig4.png')




ddd
#%%
# calculate numerically first and second moments

def fct_std(T,mu,w,l,g, omega): # T: number of timesteps, mu: social learning rate, w: intertial factor, l=local best, g=global best, domain=[-omega, omega]
    p=(l+g)/2
    
    # operator: Z_t+1=MZ_t+b
    
    M=np.array([[1+w-mu, -w,0,0,0],
                [1,0,0,0,0],
                [p*4*(mu/2*(1+w)-(mu**2)/3-(mu/2)**2), -4*mu/2*w*p, (1+w)**2-4*mu/2*(1+w)+2*(mu**2)/3+2*(mu/2)**2,  2*w*(2*(mu/2)-(1+w)),w**2],
                [mu*p,0,(1+w)-mu,-w,0],
                [0,0,1,0,0],
                ])
    
    
    
    b=np.array([mu*p, 
                0, 
                ((mu**2)/3)*(l**2)+2*((mu/2)**2)*l*g+((mu**2)/3)*(g**2),
                0,
                0])
   
    
  
    #
    Z=np.zeros((5,T)) # vector of moments
    
    Z0=np.array([mu*p, 0, (7*mu**2-12*mu+6*w**2+6/18)*(omega)**2+(mu)*mu*((p*2)**2)/3, 1/3*(1-mu)*omega**2, 1/3*omega**2]) # starting point
    Z[:,0]=Z0
    
    
    # iteration
    for t in range(T-1):
        Z[:,t+1]=M.dot(Z[:,t])+b

    # calculate standart deviation from calculated moments
    
    STD=np.zeros(T)
    for t in range(T-1):
        STD[t+1]=np.sqrt(Z[2,t+1]-(Z[0,t+1])**2)
    

    zstar=np.linalg.inv(np.identity(5)-M).dot(b)
    
    return Z,STD


#%%
T=50 # number of time steps
l=0.5 # local best position in stagnation
g=0.5 # global best position in stagnation   
omega=0.5 # range of initialiconditons of particles and velocity: x-Unif[-omega,omega], v-Unif[-omega,omega


w_overdamped, mu_overdamped=0.7, 1.4 # overdamped PSO parameters
phi_overdamped=mu_overdamped


p_overdamped=(mu_overdamped*l+phi_overdamped*g)/(mu_overdamped+phi_overdamped)
Z_overdamped, STD_overdamped=fct_std(T,mu_overdamped, w_overdamped, l,g, omega)
pst_overdamped=0.5*np.sqrt((mu_overdamped*(w_overdamped+1))/(mu_overdamped*(5*w_overdamped-7)-12*w_overdamped**2+12))*np.abs(l-g)

w_divergent, mu_divergent=1, 0.14 # divergent oscillating PSO parameters
phi_divergent=mu_divergent


p_divergent=(mu_divergent*l+phi_divergent*g)/(mu_divergent+phi_divergent)
Z_divergent, STD_divergent=fct_std(T,mu_divergent, w_divergent, l,g,omega)
pst_O=0.5*np.sqrt((mu_divergent*(w_divergent+1))/(mu_divergent*(5*w_divergent-7)-12*w_divergent**2+12))*np.abs(l-g)


w_damped, mu_damped=0.958058, 0.3215 # damped PSO parameters
phi_damped=mu_damped


p_damped=(mu_damped*l+phi_damped*g)/(mu_damped+phi_damped)
Z_damped, STD_damped=fct_std(T,mu_damped, w_damped,l,g, omega)
pst_damped=0.5*np.sqrt((mu_damped*(w_damped+1))/(mu_damped*(5*w_damped-7)-12*w_damped**2+12))*np.abs(l-g)

#%% create a figure with 3 columns and 1 row
legend_fontsize=30
fig, axs = plt.subplots(1, 3, figsize=(20,7))
plt.subplots_adjust(top=0.78, bottom=0.22,left=0.05, right=0.95)

# overdamped statistics

axs[0].set_title('Overdamped', fontsize=30)
axs[0].plot(np.linspace(0,T-1, T), Z_overdamped[0,:], linewidth=5, label='mean')
axs[0].plot(np.linspace(0,T-1, T), np.ones(T)*p_overdamped,color='red', linewidth=5, label='p')
axs[0].plot(np.linspace(0,T-1, T), STD_overdamped**2, linewidth=5)
#axs[0].plot(np.linspace(0,T-1, T), Z_overdamped[2,:])
axs[0].tick_params(axis='both', which='major', labelsize=30)
axs[0].set_xlabel('time', fontsize=30)
axs[0].set_ylabel('position', fontsize=30)
axs[0].set_ylim(-0.5,1)

# diverg. oscillating statistics
axs[1].set_title('Damped', fontsize=30)
axs[1].plot(np.linspace(0,T-1, T), Z_damped[0,:],linewidth=5)
axs[1].plot(np.linspace(0,T-1, T), np.ones(T)*p_damped,linewidth=5,color='red')
axs[1].plot(np.linspace(0,T-1, T), STD_damped**2,linewidth=5, label='var')
#axs[1].plot(np.linspace(0,T-1, T), Z_damped[2,:])
axs[1].set_xlabel('time', fontsize=30)
axs[1].set_ylabel('position', fontsize=30)
axs[1].tick_params(axis='both', which='major', labelsize=30)
axs[1].set_ylim(-0.5,5)

# damped statistics
axs[2].set_title('Divergent', fontsize=30)
axs[2].set_ylabel('position', fontsize=30)
axs[2].plot(np.linspace(0,T-1, T), Z_divergent[0,:], linewidth=5,label='mean')
axs[2].plot(np.linspace(0,T-1, T), STD_divergent**2, linewidth=5,label='var')
#axs[2].plot(np.linspace(0,T-1, T), Z_divergent[2,:], label='second moment')
axs[2].plot(np.linspace(0,T-1, T), np.ones(T)*p_divergent,color='red',linewidth=5, label='p')
axs[2].set_xlabel('time', fontsize=30)
axs[2].tick_params(axis='both', which='major', labelsize=30)
axs[2].set_ylim(-0.5,10)
axs[2].legend(bbox_to_anchor=(-1.5, 1.02, 1, 0.35), loc="upper left",
               ncol=4,  prop={'size': legend_fontsize})

axs[0].text(0.1, -0.3, '(a)', transform=axs[0].transAxes, fontsize=30, va='top', ha='right', weight='bold')
axs[1].text(0.1, -0.3, '(b)', transform=axs[1].transAxes, fontsize=30, va='top', ha='right',weight='bold')
axs[2].text(0.1, -0.3, '(c)', transform=axs[2].transAxes, fontsize=30, va='top', ha='right', weight='bold')
plt.savefig('Fig5.png')



