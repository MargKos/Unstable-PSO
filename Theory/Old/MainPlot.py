#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 11:55:13 2024

@author: bzfkostr
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def fct_abs(c): # complex absolute value
    r=c.real
    i=c.imag
    return np.sqrt(r**2+i**2)

discr=1200
Mu = np.linspace(0.01, 3.99, discr)
W = np.linspace(-1, 1.1, discr)
WMuB = np.zeros((discr, discr))
WMuA = np.zeros((discr, discr))

BDDamped = []
BDStable = []
BDDivergent = []
BDComplex = []
for k in range(discr):
    w = W[discr-1-k]
    for j in range(discr):
        mu = Mu[j]
        M = np.array([[ (1+w)**2-4*mu/2*(1+w)+2*(mu**2)/3+2*(mu/2)**2,  2*w*(2*(mu/2)-(1+w)),w**2],
                [(1+w)-mu,-w,0],
                [1,0,0]])
        ev, ew = np.linalg.eig(M)
        
        NormEv = np.zeros(len(ev))
        for i in range(len(ev)):
            NormEv[i] = fct_abs(ev[i])
        
        WMuB[k,j] = np.max(NormEv)
        if WMuB[k,j] < 1:
            if (1+w-mu)**2 < 4*w: # check if compley
                
                if 0.945< WMuB[k,j] < 0.955:
                    BDDamped.append([mu,w])
        
        
        M1 = np.array([[1+w-mu,-w],[1,0]])
        ev1, ew1 = np.linalg.eig(M1)
        NormEv1 = np.zeros(len(ev1))
        for i in range(len(ev1)):
            NormEv1[i] = fct_abs(ev1[i])
        WMuA[k,j] = np.max(NormEv1)
        
        if 0.999<WMuB[k,j] <= 1.01 and (1+w-mu)**2 < 4*w:
            BDDivergent.append([mu,w])
        
        if 1<WMuA[k,j] < 1.005 and (1+w-mu)**2 <=4*w:
            BDStable.append([mu,w])
        
        if w>0 and -0.01<(1+w-mu)**2-4*w<0.05 and 1.005>WMuA[k,j]  :
            BDStable.append([mu,w])

#%%
discr=50
Mu = np.linspace(0.01, 3.99, discr)
W = np.linspace(-1, 1.1, discr)
WMuB = np.zeros((discr, discr))
WMuA = np.zeros((discr, discr))

Damped = []
Overdamped = []
Divergent = []

for k in range(discr):
    w = W[discr-1-k]
    for j in range(discr):
        mu = Mu[j]
        M = np.array([[ (1+w)**2-4*mu/2*(1+w)+2*(mu**2)/3+2*(mu/2)**2,  2*w*(2*(mu/2)-(1+w)),w**2],
                [(1+w)-mu,-w,0],
                [1,0,0]])
        ev, ew = np.linalg.eig(M)
        
        NormEv = np.zeros(len(ev))
        for i in range(len(ev)):
            NormEv[i] = fct_abs(ev[i])
        
        WMuB[k,j] = np.max(NormEv)
        if WMuB[k,j] < 1:
            if (1+w-mu)**2 < 4*w: # check if compley
                if WMuB[k,j] < 0.95:
                    Overdamped.append([mu,w])
                if WMuB[k,j] >= 0.95:
                    Damped.append([mu,w])
                
                if 0.94< WMuB[k,j] < 0.96:
                    BDDamped.append([mu,w])
            
            if 4*w-0.05<(1+w-mu)**2 < 4*w+0.05:
                BDComplex.append([mu,w])
                    
        
        M1 = np.array([[1+w-mu,-w],[1,0]])
        ev1, ew1 = np.linalg.eig(M1)
        NormEv1 = np.zeros(len(ev1))
        for i in range(len(ev1)):
            NormEv1[i] = fct_abs(ev1[i])
        WMuA[k,j] = np.max(NormEv1)
        if WMuA[k,j] <= 1 and WMuB[k,j] > 1 and (1+w-mu)**2 < 4*w:
            Divergent.append([mu,w])
        
        

# Sorting the points
#%%
BDStable=list(BDStable)
BDComplex=list(BDComplex)
All=Divergent+Damped+Overdamped+BDStable+BDComplex
All=np.array(All)
AllHull = ConvexHull(All)

BDDamped=np.array(BDDamped)
BDDivergent=np.array(BDDivergent)
BDComplex=np.array(BDComplex)
BDStable=np.array(BDStable)

Overdamped_sorted = sorted(Overdamped, key=lambda x: (x[0], x[1]))
Divergent_sorted = sorted(Divergent, key=lambda x: (x[0], x[1]))
Damped_sorted = sorted(Damped, key=lambda x: (x[0], x[1]))

Overdamped_sorted = np.array(Overdamped_sorted)
Divergent_sorted = np.array(Divergent_sorted)
Damped_sorted = np.array(Damped_sorted)

OverdampedHull = ConvexHull(Overdamped_sorted)
DivergentHull = ConvexHull(Divergent_sorted)

from matplotlib.patches import PathPatch
from matplotlib.path import Path

# Define hatch patterns
overdamped_hatch = 'X'
divergent_hatch = 'O'
damped_hatch = '**'
fig, ax = plt.subplots(figsize=(20,10))

# Turn off x and y axis
plt.axis('off')



x_overdamped=Overdamped_sorted[:, 0]
y_overdamped=Overdamped_sorted[:, 1]

x_divergent=Divergent_sorted[:, 0]
y_divergent=Divergent_sorted[:, 1]

x_damped=Damped_sorted[:, 0]
y_damped=Damped_sorted[:, 1]
'''
plt.plot(Overdamped_sorted[:, 0], Overdamped_sorted[:, 1], marker='s', markersize=5, color='orange', linestyle='None', fillstyle='none', hatch='X')

plt.plot(Divergent_sorted[:, 0], Divergent_sorted[:, 1], marker='o', markersize=5, color='blue', linestyle='None', fillstyle='none', hatch='O')

plt.plot(Damped_sorted[:, 0], Damped_sorted[:, 1], marker='^', markersize=5, color='green', linestyle='None', fillstyle='none', hatch='**')
'''
# Plot points for overdamped with hatch style
for xi, yi in zip(x_overdamped, y_overdamped):
    polygon = np.array([[xi-0.02, yi-0.02], [xi-0.06, yi+0.06], [xi+0.05, yi+0.05], [xi+0.05, yi-0.05], [xi-0.03, yi-0.03]])
    path = Path(polygon)
    patch = PathPatch(path, facecolor='none', edgecolor='grey', linewidth=0.5, linestyle='None', hatch=overdamped_hatch)
    ax.add_patch(patch)

# Plot points for divergent with hatch style
for xi, yi in zip(x_divergent, y_divergent):
    polygon = np.array([[xi-0.02, yi-0.02], [xi-0.02, yi+0.02], [xi+0.02, yi+0.02], [xi+0.02, yi-0.02], [xi-0.02, yi-0.02]])
    path = Path(polygon)
    patch = PathPatch(path, facecolor='none', edgecolor='grey', linewidth=0.5, linestyle='None', hatch=divergent_hatch)
    ax.add_patch(patch)

# Plot points for damped with hatch style
for xi, yi in zip(x_damped, y_damped):
    polygon = np.array([[xi-0.02, yi-0.02], [xi-0.05, yi+0.05], [xi+0.03, yi+0.03], [xi+0.03, yi-0.03], [xi-0.02, yi-0.02]])
    path = Path(polygon)
    patch = PathPatch(path, facecolor='none', edgecolor='grey',fill=True, linewidth=0.5, linestyle='None', hatch=damped_hatch)
    ax.add_patch(patch)

# plot boundaries

plt.plot(BDDamped[:, 0], BDDamped[:, 1], 'o', markersize=0.5, color='black')

plt.plot(BDDivergent[:, 0], BDDivergent[:, 1], 'o', markersize=0.5, color='black')

#plt.plot(BDStable[:, 0], BDStable[:, 1], 'o', markersize=0.5, color='black')

#plt.plot(BDComplex[:, 0], BDComplex[:, 1], 'o', markersize=0.5, color='black')
# Plot the boundary of the concave hull for the overdamped region
for simplex in AllHull.simplices:
    plt.plot(All[simplex, 0], All[simplex, 1], 'k-', color='black',linewidth=4)

# Save the figure
plt.savefig('Main.png', bbox_inches='tight', dpi=300)
plt.show()


#%%

# boundary of complex region
hull_verticesAll = AllHull.vertices
hull_pointsAll = All[hull_verticesAll]

# boundary od damped region BDDamped

# boundary od divegent region BDDivergent

# get all points of ompley boundary
from matplotlib.patches import Polygon
Complex=[]
for k in range(discr):
    w = W[discr-1-k]
    for j in range(discr):
        mu = Mu[j]
       
        if (1+w-mu)**2 < 4*w:
            Complex.append([mu,w])
Complex1=[]
for k in range(discr):
    w = W[discr-1-k]
    
    for j in range(discr):
        mu = Mu[j]
        if mu<1.7:
            if (1+w-mu)**2 < 4*w:
                Complex1.append([mu,w])
# do convex hull aroudn overdamped region

# boundary of complex region
hullOverdamped_vertices = OverdampedHull.vertices
hullOverdamped_points = Overdamped_sorted[hullOverdamped_vertices]
XY=list(Complex1)+list(BDDamped)
# Create a figure and axis
fig, ax = plt.subplots()

# Create a polygon patch
polygon = Polygon(XY)

# Add the polygon patch to the axis
ax.add_patch(polygon)


# Show the plot
plt.show()


        
