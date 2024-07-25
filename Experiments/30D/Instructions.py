#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 10:23:57 2023

@author: bzfkostr
"""

''' Checklist

Set in Variable.py:

- number of particles 'n'
- number of simulations in -sh file
- number of timesteps 'T_PSO'
- where to save the files 'path'
- select starting points (eventually generate startings points in StartingPoints.py )

0) Calculate Waiting times
    0.1 Run PSO with a small number of time steps T_PSO_short
    0.2 Run WaitingTimes.py
    0.3 write down the waiting times in Variables.py

1) Run simulations (in -sh file)
1.1 Run PSO simulations PSO.py with multiprocessing
1.2 Run short PSO simultions that stores the particle Positions 
1.3 Run ExplorationMultiProcessing.py with multiprocessing to calculate for each simulation the number of found local minima with CG
1.4 Run Exploration.py, which averages exploration over all simulations calculated in 1.3, save all found local minima and calculated their functions values
1.5 Run Measures.py to calculate mean function values and exploitation 

2) Generate Figures
Run Plot.py to generate exploitation, exploration, mean function value, and local minima values 
distribution plot
'''
