# Unstable-PSO


This project involves running particle simulations and analyzing various performance measures. Below are the steps and configuration details to guide you through the process.

## 1. Set Parameters in `Variable.py`
Before running the simulations, configure the following parameters in the `Variable.py` file:

- **Number of particles:** Set the total number of particles involved in the simulations.
- **Number of simulations:** Define the number of simulation runs in the `-sh` file.
- **Number of iteration steps:** Specify the total number of iteration steps for each simulation.
- **Output directory:** Choose where to save the simulation results and generated figures.
- **Starting points:** Select or generate starting points. You can use predefined starting points or generate them using the `StartingPoints.py` script.

## 2. Workflow Overview

### Step 0: Generate Starting Points
Generate the initial starting points for the simulations by running the `StartingPoint1D.py` script. This script creates or loads initial positions for the particles.


python StartingPoint1D.py

0) generate starting points in StartingPoint1D.py
2) Run simulations (in -sh file)
3) Run PSO simulations PSO_(example).py with multiprocessing
4) Run Exploration_(example).py to calculate average exploration
5) Run Measures_(example).py to calculate  mean function values and average exploitation 
6) Generate Figures with Plots_(example).py

Folder Structure

In the folder 'Theory' one can find the code to figures to support the figures 1-6 in the thoretical section.
The folder 'Experiemnts' contains three subfolders for each dimension: '1D', '5D', '30D'
Each of them is containing following folders': 'Data', which stores all the given data for the experiments (here the starting positions of the swarm), 'LocalMinima', which 
contains the local minima of the given benchmarc functions, 'Plots' with figures for the paper and 'Results', which contains all necessary .npy files to generate the figures

Other files

Functions.py defines the energy landscape, which PSO is minimizing

Getting Started
Python version with 



This project is licensed under the [License Name] - see the LICENSE.md file for details.
Acknowledgments

If your project uses third-party libraries, tools, or other resources, acknowledge them here. You can also thank individuals or teams who contributed to the project.
