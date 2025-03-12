# active_sensing

Python implementation of **"IGBIPP Towards Field Deployment"**, an extension of **"Multi-UAV active sensing for precision agriculture via Bayesian fusion"** by Luca Pierdicca, Dimitri Ognibene, and Vito Trianni. This extended version includes advanced simulation, data aggregation, and visualization tools for precision agriculture applications.



<!-- ## Overview

This repository provides a comprehensive simulation framework for multi-UAV active sensing in precision agriculture. The project extends previous work by integrating:
- **UAV Trajectory Planning:** Generate efficient flight paths using active sensing strategies.
- **Camera and Sensor Simulation:** Model camera observations, including field-of-view, overlap, and noise.
- **Data Aggregation and Visualization:** Process and visualize data collected during simulated flights.

The implementation leverages Bayesian fusion techniques to enhance data quality and decision-making in field deployment scenarios. -->

<!-- ## Features

- **Multi-UAV Coordination:** Simulate and plan trajectories for multiple UAVs.
- **Active Sensing:** Optimize UAV paths to collect high-value information.
- **Camera Simulation:** Detailed modeling of camera parameters and noise.
- **Precision Agriculture:** Tools and datasets for agricultural monitoring.
- **Visualization:** Generate animations and plots to assess UAV performance. -->

## Requirements

- Python 3.7+
- NumPy
- PyTorch
- Torchvision
- Pillow
- GDAL
- Matplotlib

All additional dependencies can be found in the `requirements.txt` file.

## Installation

Clone the repository and install the required packages:

```bash
git clone <repository-url>
cd active_sensing
pip install -r requirements.txt
```

## Usage

python src/main.py --config config.json

## Structure

active_sensing/
├── config.json           # Configuration file for simulation parameters
├── plots_/               # Folder for storing generated plots and animations
├── src/
│   ├── main.py           # Main entry point for the simulation
│   ├── uav_camera.py     # UAV and camera model (configuration space and observations)
│   ├── mapper_LBP.py     # Mapper (Bayesian Update + Loopy Belief Propagation)
│   ├── planner.py        # Planner (Informtion Gain based and Sweep)
│   ├── orthomap.py       # Field (both simulated Gaussian correlated field and orthomosaic wheat field)
│   └── ...               # Additional modules and utilities
├── binary_classfier/     # Training and usage of classifier model
│   ├── models/           # Contains trained models
│   ├── predictor.py      # Model prediction functions
│   ├── train.ipynb       # Training
│   └── ...               # Other useful functionalities
├── README.md             # Project documentation (this file)
├── plotter.py            # Collect experiments and generate final plots for each scenario
└── requirements.txt      # List of project dependencies

## Example
An example map generated during simulation using Gaussian field (r=4), error margin=0.3 and equal weights:
![Gaussian Animation](plots_/ig_gaussian_equal_e0.3.gif)

An example sweep animation generated during simulation using Orthomap field and adaptive weights:
![Sweep Animation](plots_/sweep_adaptive_orthomap.gif)
