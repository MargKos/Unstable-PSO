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

# Workflow
1. **Generate Starting Points:** Use `StartingPoint1D.py` to create initial data and set PSO parameters in `Variables.py`.
2. **Run PSO Simulations:** Perform PSO simulations with `run_python_(example).sh` using multiprocessing.
3. **Calculte number of discovered local minima:** Perform `ExplorationMultiProcessing.py` for every PSO simulation with `run_python_Exploration.sh` using multiprocessing.
4. **Calculate Average Exploration:** Use `run_python_ExplorationMulti.py` to compute the average exploration.
5. **Calculate Mean Function Values and Average Exploitation:** Execute `Measures_(example).py` for these calculations.
6. **Generate Figures:** Create figures using `Fig().py` or `Fig().ipynb`.

To calculate the waiting times, run short PSO simulations and evaluate how long a bird in PSO did not change the position in `WaitingTimesFig13.ipynb`.


# Folder Structure

- **'Theory':** Contains code for generating figures that support figures 1-6 in the theoretical section.
- **'Experiments':** This folder is organized into three subfolders for each dimension:
  - **'1D'**
  - **'5D'**
  - **'30D'**
  Each subfolder includes:
  - **'Data':** Stores the experimental data, including the starting positions of the swarm.
  - **'LocalMinima':** Contains the local minima of the benchmark functions.
  - **'Plots':** Holds figures intended for the paper.
  - **'Results':** Contains all necessary `.npy` files for generating the figures.

# Other Files

- **`Functions.py` or `(example)_fct.sh`:** Defines the energy landscape that PSO minimizes.

# Getting Started

Ensure you have the required Python version and dependencies. [Details to be provided here.]

# License

This project is licensed under the [License Name] - see the `LICENSE.md` file for details.

# Acknowledgments

If your project uses third-party libraries, tools, or other resources, please acknowledge them here. You can also thank individuals or teams who contributed to the project.
