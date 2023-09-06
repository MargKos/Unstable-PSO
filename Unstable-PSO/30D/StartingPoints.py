#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 11:34:08 2023

@author: bzfkostr
"""

import numpy as np


n=20
dim=30
sim=100

BdXlow=-5.12
BdXUp=5.12

StartWerte=np.empty([ dim,n, sim])
for s in range(sim):
    for i in range(n):
        candidate=np.random.uniform(BdXlow, BdXUp, dim)

        StartWerte[:,i, s]=candidate
   

np.save('Uniform30D.npy', StartWerte)

#%%

a = -5.12
b = 5.12

# Define the range to exclude
exclude_range = [-2, 2]

# Define the dimensionality of the points
dim = 30

# Generate random points in the range [-5.12, -1)
points1 = np.random.uniform(a, exclude_range[0], size=(dim, 10000))

# Generate random points in the range [1, 5.12]
points2 = np.random.uniform(exclude_range[1], b, size=(dim, 10000))

# Concatenate the two sets of points along the specified axis (default is 0)
points = np.concatenate((points1, points2), axis=1)

# Transpose the points array to get the points in the desired shape
points = points.T

# Reshape the points array into groups of 20 points along the last axis
points = np.reshape(points, (30, 20, 1000))

print(points.shape)

np.save('Boundary30D.npy', points)