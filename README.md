# active_sensing

Python implementation of **"IGBIPP Towards Field Deployment"**, an extension of **"Multi-UAV active sensing for precision agriculture via Bayesian fusion"** by Luca Pierdicca, Dimitri Ognibene, and Vito Trianni. This extended version includes advanced simulation, data aggregation, and visualization tools for precision agriculture applications.

![Sweep Animation](plots_/sweep_adaptive_orthomap.gif)

## Overview

This repository provides a comprehensive simulation framework for multi-UAV active sensing in precision agriculture. The project extends previous work by integrating:
- **UAV Trajectory Planning:** Generate efficient flight paths using active sensing strategies.
- **Camera and Sensor Simulation:** Model camera observations, including field-of-view, overlap, and noise.
- **Data Aggregation and Visualization:** Process and visualize data collected during simulated flights.

The implementation leverages Bayesian fusion techniques to enhance data quality and decision-making in field deployment scenarios.

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

## Usage

python src/main.py --config config.json

