#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:23:57 2023

@author: bzfkostr
"""

''' Checklist

Set in Variable.py:

- number of particles
- number of simulations in -sh file
- number of iteration-steps
- where to save the files
- select starting points (eventually generate startings points in StartingPoints.py )

0) generate starting points in StartingPoint1D.py
1) Run simulations (in -sh file)
1.1 Run PSO simulations PSO1D.py with multiprocessing
1.2 Run Exploration1D.py to calculate average exploration
1.3 Run Measures1D.py to calculate  mean function values and average exploitation 

2) Generate Figures

'''
