# CERN_project
GANs for simulation of electromagnetic showers in the ATLAS calorimeter 

1) MNIST.py

First implementation of Improved Wasserstein GAN (WGAN-gp), on the MNIST dataset.
See the article : https://arxiv.org/abs/1704.00028

2) WGAN for electromagnetic showers in the ATLAS calorimeter
A first script - PreProcessing.py - applies transformations to the former dataset (noise cuts, remove events>300 GeV), as describe in the file report.pdf.

Three Python files that work together :
- config.py (contains training parameters, path to files, ...)
- plot_functions.py (where all plot functions are defined)
- training.py (main file to train the WGAN)

To launch a training, you should write in the terminal : python training.py Name (Name = name of your folder in which all plots and weights will be saved. Will create the folder if it doens't already exist)

Plots are generated automatically each 250 epochs, a folder is created each time.

3) A Jupyter notebook

Interactive plots 
