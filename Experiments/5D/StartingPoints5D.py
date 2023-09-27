#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 11:34:08 2023

@author: bzfkostr
"""

import numpy as np
from Variables import *

'''Generates starting points, which are uniformly distributed '''

# set domain range

BdXlow=0 
BdXUp=np.pi

StartWerte=np.empty([ dim,n, sim])
for s in range(sim):
    for i in range(n):
        StartWerte[:,i, s]=np.random.uniform(BdXlow, BdXUp, dim)
 

np.save('Uniform5D.npy', StartWerte)

